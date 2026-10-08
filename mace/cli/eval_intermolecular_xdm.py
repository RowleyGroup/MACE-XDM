###########################################################################################
# Evaluate a combined short-range-MACE + XDM-dispersion potential (MACEXDMDispersion) on
# intermolecular complex/monomer data: HDF5 files with one leaf group per complex and one
# leaf group per monomer fragment (e.g. as written by a postg/Gaussian-parsing script in
# the style of parse_tmpdir.py -- complex groups like "<prefix>_<i>", fragment groups like
# "<prefix>_<i>.frag_1"/"<prefix>_<i>.frag_2").
#
# Reports how well the model's predicted interaction energies (complex - frag_1 - frag_2)
# match the QM reference: PBE0 alone (short-range model only), PBE0+XDM (full combined
# model), and XDM alone (dispersion_energy output only) -- the same three quantities
# parse_tmpdir.py writes to its own CSV (deltaE_pbe0, deltaE_pbe0+deltaE_xdm, deltaE_xdm).
###########################################################################################

import argparse
import glob
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

from mace.data import build_xdm_atomic_data, mlxdm_2x_polarizability_reference
from mace.data.xdm import find_leaf_groups, species_to_atomic_numbers
from mace.modules import MACEXDMDispersion, XDMDispersionEnergy, load_xdm_model
from mace.tools import AtomicNumberTable, init_device, load_full_model, set_default_dtype, setup_logger
from mace.tools.torch_geometric.batch import Batch
from mace.tools.torch_geometric.dataloader import DataLoader

HARTREE_TO_KCAL_MOL = 627.5094740631
HARTREE_TO_EV = 27.211386245988
EV_TO_KCAL_MOL = HARTREE_TO_KCAL_MOL / HARTREE_TO_EV

# Matches a group name ending in "_<index>" (a complex) or "_<index>.frag_1"/".frag_2" (a
# fragment), regardless of whatever prefix text comes before the index -- deliberately
# robust to a complex/fragment prefix mismatch like parse_tmpdir.py's own
# "deshaw370k_<i>" vs "deshaw15k_<i>.frag_1" naming.
_INDEX_RE = re.compile(r"_(\d+)(?:\.frag_([12]))?$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--short_range_model",
        required=True,
        help="Path to the short-range energy MACE .model file (e.g. trained on PBE0 "
        "total energies, no dispersion).",
    )
    parser.add_argument(
        "--xdm_model",
        required=True,
        help="Path to the AtomicXDMMACE model -- either a full .model or a "
        "mace_run_train_xdm checkpoint (<name>_{latest,best}.pt).",
    )
    parser.add_argument("--data_files", required=True, nargs="+")
    parser.add_argument("--species_key", default="atomic_numbers")
    parser.add_argument("--coordinates_key", default="coordinates")
    parser.add_argument(
        "--energy_key",
        default="energies",
        help="Dataset holding the QM reference total energy (PBE0, no dispersion), "
        "in Hartree.",
    )
    parser.add_argument(
        "--exdm_key",
        default="e_xdm",
        help="Dataset holding the QM reference XDM dispersion energy, in Hartree.",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "xpu"])
    parser.add_argument("--default_dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dispersion_cutoff", type=float, default=14.0)
    parser.add_argument("--bj_a1", type=float, default=0.4186)
    parser.add_argument("--bj_a2", type=float, default=2.6791)
    parser.add_argument(
        "--max_triplets",
        type=int,
        default=None,
        help="If set, randomly subsample to at most this many complexes for a quick look.",
    )
    parser.add_argument(
        "--min_fragment_atoms",
        type=int,
        default=1,
        help="Skip a triplet if its complex, frag_1, or frag_2 has fewer than this many "
        "atoms. Default 1 (no filtering). Fragments this small are usually a minimal "
        "diatomic/monatomic species (e.g. a bare H2) rather than the kind of organic "
        "fragment the short-range model's training data was actually built from -- the "
        "model can extrapolate wildly (potentially 100+ kcal/mol errors) on them, which "
        "dominates aggregate metrics out of proportion to how often such fragments "
        "actually occur. Try --min_fragment_atoms 3 to exclude diatomics like H2.",
    )
    parser.add_argument(
        "--skip_homonuclear_fragments",
        action="store_true",
        help="Skip a triplet if any of its structures consist of a single element only "
        "(e.g. H2, O2, N2, Cl2) -- a more targeted version of --min_fragment_atoms for "
        "exactly the failure mode of isolated single-element diatomics/small clusters "
        "that organic-molecule-derived training data (e.g. ANI-2x-style) typically never "
        "includes as a standalone fragment.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="intermolecular_results")
    parser.add_argument("--no_plots", action="store_true")
    return parser


def expand_file_patterns(patterns: List[str]) -> List[str]:
    files: List[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No files matched '{pattern}'")
        files.extend(matches)
    return files


def discover_triplets(file_paths: List[str]) -> List[Dict[str, Tuple[str, str]]]:
    """Group every leaf group across file_paths into {"complex", "frag1", "frag2"}
    triplets by trailing numeric index, returning {role: (file_path, leaf_path)}.
    Raises if any triplet is incomplete (a complex with only one fragment, etc.).
    """
    by_index: Dict[str, Dict[str, Tuple[str, str]]] = defaultdict(dict)
    for path in file_paths:
        with h5py.File(path, "r") as f:
            for leaf in find_leaf_groups(f):
                name = leaf.rsplit("/", 1)[-1]
                m = _INDEX_RE.search(name)
                if m is None:
                    logging.warning(f"Skipping group {leaf!r}: no trailing _<index> found.")
                    continue
                index, frag = m.group(1), m.group(2)
                role = "complex" if frag is None else f"frag{frag}"
                if role in by_index[index]:
                    raise ValueError(
                        f"Duplicate {role!r} for index {index!r}: "
                        f"{by_index[index][role]} and {(path, leaf)}"
                    )
                by_index[index][role] = (path, leaf)

    triplets = []
    incomplete = []
    for index, roles in by_index.items():
        if {"complex", "frag1", "frag2"} <= roles.keys():
            triplets.append(roles)
        else:
            incomplete.append((index, sorted(roles.keys())))
    if incomplete:
        logging.warning(
            f"{len(incomplete)} indices had an incomplete complex/frag1/frag2 set and "
            f"were skipped, e.g. {incomplete[:5]}"
        )
    if not triplets:
        raise ValueError(
            "No complete complex/frag1/frag2 triplets found. Expected group names ending "
            "in '_<index>' (complex) and '_<index>.frag_1'/'_<index>.frag_2' (fragments); "
            "pass --species_key/--coordinates_key if your dataset keys differ, or check "
            "that this file actually follows that naming convention."
        )
    return triplets


def _read_structure(
    f: h5py.File, leaf: str, species_key: str, coordinates_key: str, energy_key: str, exdm_key: str
):
    grp = f[leaf]
    atomic_numbers = species_to_atomic_numbers(grp[species_key][()])
    coords = np.asarray(grp[coordinates_key][()], dtype=np.float64)
    if coords.ndim == 2:
        coords = coords[None]
    energy = float(np.asarray(grp[energy_key][()]).reshape(-1)[0])
    e_xdm = float(np.asarray(grp[exdm_key][()]).reshape(-1)[0])
    return atomic_numbers, coords[0], energy, e_xdm


def main():
    args = build_parser().parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    setup_logger(level=logging.INFO)
    set_default_dtype(args.default_dtype)
    device = init_device(args.device)

    short_range_model = load_full_model(args.short_range_model, device=device)
    xdm_model = load_xdm_model(args.xdm_model, device=device)
    xdm_z_table = AtomicNumberTable(xdm_model.atomic_numbers.tolist())
    # The one graph built below (keyed to xdm_z_table) is shared by both
    # sub-models -- cut it off at whichever model needs the wider cutoff, or
    # the short-range model silently loses real neighbors beyond its own
    # r_max (see the matching note in MACEXDMDispersion.__init__).
    r_max = max(float(xdm_model.r_max.item()), float(short_range_model.r_max.item()))
    logging.info(
        f"Short-range model elements: {short_range_model.atomic_numbers.tolist()}; "
        f"XDM model elements: {xdm_z_table.zs}, r_max={r_max}"
    )

    ref = mlxdm_2x_polarizability_reference(xdm_z_table)
    dispersion_energy = XDMDispersionEnergy(
        alpha_free=ref["alpha_free"],
        v_free=ref["v_free"],
        cutoff=args.dispersion_cutoff,
        a1=args.bj_a1,
        a2=args.bj_a2,
    )
    model = MACEXDMDispersion(
        short_range_model=short_range_model,
        xdm_model=xdm_model,
        dispersion_energy=dispersion_energy,
    ).to(device)
    model.eval()

    data_files = expand_file_patterns(args.data_files)
    triplets = discover_triplets(data_files)
    logging.info(f"Found {len(triplets)} complete complex/frag1/frag2 triplets")

    if args.max_triplets is not None and len(triplets) > args.max_triplets:
        rng = np.random.RandomState(args.seed)
        triplets = [triplets[i] for i in rng.choice(len(triplets), args.max_triplets, replace=False)]
        logging.info(f"Subsampled to {len(triplets)} triplets")

    sr_supported = set(short_range_model.atomic_numbers.tolist())
    xdm_supported = set(xdm_z_table.zs)
    supported = sr_supported & xdm_supported

    # Build one graph per structure (3 per triplet), tagged with (triplet_idx, role) so
    # predictions can be recombined into interaction energies afterwards. A triplet with
    # any atom outside what both models support (e.g. a stray/anomalous atom in one
    # system's data) is skipped entirely rather than crashing the whole run.
    graphs = []
    tags: List[Tuple[int, str]] = []
    true_energy: Dict[Tuple[int, str], float] = {}
    true_exdm: Dict[Tuple[int, str], float] = {}
    kept_triplets = []
    n_skipped = 0
    n_skipped_fragment_filter = 0
    open_files = {path: h5py.File(path, "r") for path in data_files}
    try:
        for roles in triplets:
            structures = {}
            for role in ("complex", "frag1", "frag2"):
                path, leaf = roles[role]
                structures[role] = _read_structure(
                    open_files[path], leaf, args.species_key, args.coordinates_key,
                    args.energy_key, args.exdm_key,
                )

            unsupported = {
                role: sorted(set(atomic_numbers.tolist()) - supported)
                for role, (atomic_numbers, *_rest) in structures.items()
            }
            unsupported = {role: zs for role, zs in unsupported.items() if zs}
            if unsupported:
                n_skipped += 1
                logging.warning(
                    f"Skipping {roles['complex'][1]!r}: unsupported atomic number(s) "
                    f"{unsupported} (not covered by both the short-range and XDM models)."
                )
                continue

            too_small = {
                role: len(atomic_numbers)
                for role, (atomic_numbers, *_rest) in structures.items()
                if len(atomic_numbers) < args.min_fragment_atoms
            }
            homonuclear = {
                role: int(atomic_numbers[0])
                for role, (atomic_numbers, *_rest) in structures.items()
                if args.skip_homonuclear_fragments and len(set(atomic_numbers.tolist())) == 1
            }
            if too_small or homonuclear:
                n_skipped_fragment_filter += 1
                reason = []
                if too_small:
                    reason.append(f"below --min_fragment_atoms={args.min_fragment_atoms}: {too_small}")
                if homonuclear:
                    reason.append(f"single-element structure(s): {homonuclear}")
                logging.warning(f"Skipping {roles['complex'][1]!r}: " + "; ".join(reason))
                continue

            t_idx = len(kept_triplets)
            kept_triplets.append(roles)
            for role, (atomic_numbers, positions, energy, e_xdm) in structures.items():
                dummy_targets = np.zeros((len(atomic_numbers), 4))
                graphs.append(
                    build_xdm_atomic_data(
                        atomic_numbers, positions, dummy_targets, xdm_z_table, cutoff=r_max
                    )
                )
                tags.append((t_idx, role))
                true_energy[(t_idx, role)] = energy
                true_exdm[(t_idx, role)] = e_xdm
    finally:
        for fh in open_files.values():
            fh.close()
    if n_skipped:
        logging.warning(f"Skipped {n_skipped}/{len(triplets)} triplet(s) with unsupported elements.")
    if n_skipped_fragment_filter:
        logging.warning(
            f"Skipped {n_skipped_fragment_filter}/{len(triplets)} triplet(s) via "
            f"--min_fragment_atoms/--skip_homonuclear_fragments."
        )
    triplets = kept_triplets

    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    pred_short_range: Dict[Tuple[int, str], float] = {}
    pred_dispersion: Dict[Tuple[int, str], float] = {}
    idx = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.to_dict(), training=False, compute_force=False)
            n = int(batch.ptr.numel() - 1)
            sr = out["short_range_energy"].cpu().numpy()
            disp = out["dispersion_energy"].cpu().numpy()
            for i in range(n):
                pred_short_range[tags[idx]] = float(sr[i])
                pred_dispersion[tags[idx]] = float(disp[i])
                idx += 1

    rows = []
    for t_idx, roles in enumerate(triplets):
        c, f1, f2 = (t_idx, "complex"), (t_idx, "frag1"), (t_idx, "frag2")
        true_pbe0 = (true_energy[c] - true_energy[f1] - true_energy[f2]) * HARTREE_TO_KCAL_MOL
        true_xdm_i = (true_exdm[c] - true_exdm[f1] - true_exdm[f2]) * HARTREE_TO_KCAL_MOL
        pred_pbe0 = (
            pred_short_range[c] - pred_short_range[f1] - pred_short_range[f2]
        ) * EV_TO_KCAL_MOL
        pred_xdm_i = (
            pred_dispersion[c] - pred_dispersion[f1] - pred_dispersion[f2]
        ) * EV_TO_KCAL_MOL
        rows.append(
            {
                "index": roles["complex"][1],
                "true_pbe0": true_pbe0,
                "true_xdm": true_xdm_i,
                "true_pbe0xdm": true_pbe0 + true_xdm_i,
                "pred_pbe0": pred_pbe0,
                "pred_xdm": pred_xdm_i,
                "pred_pbe0xdm": pred_pbe0 + pred_xdm_i,
            }
        )

    def metrics(true_key: str, pred_key: str) -> Dict[str, float]:
        true = np.array([r[true_key] for r in rows])
        pred = np.array([r[pred_key] for r in rows])
        error = pred - true
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(error**2)))
        ss_res = float(np.sum(error**2))
        ss_tot = float(np.sum((true - true.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        pearson_r = float(np.corrcoef(pred, true)[0, 1]) if len(true) > 1 and true.std() > 0 else float("nan")
        return {"n": len(true), "mae_kcal_mol": mae, "rmse_kcal_mol": rmse, "r2": r2, "pearson_r": pearson_r}

    report = {
        "PBE0": metrics("true_pbe0", "pred_pbe0"),
        "XDM": metrics("true_xdm", "pred_xdm"),
        "PBE0+XDM": metrics("true_pbe0xdm", "pred_pbe0xdm"),
    }

    print(f"\n=== Intermolecular interaction energy (n={len(rows)} complexes) ===")
    print(f"{'term':<10}{'n':>8}{'MAE(kcal/mol)':>16}{'RMSE(kcal/mol)':>17}{'R2':>8}{'pearson_r':>11}")
    for key, m in report.items():
        print(f"{key:<10}{m['n']:>8}{m['mae_kcal_mol']:>16.4f}{m['rmse_kcal_mol']:>17.4f}{m['r2']:>8.3f}{m['pearson_r']:>11.3f}")

    report_path = Path(args.output_dir) / "intermolecular_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nWrote {report_path}")

    csv_path = Path(args.output_dir) / "intermolecular_predictions.csv"
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("index,true_pbe0,true_xdm,true_pbe0xdm,pred_pbe0,pred_xdm,pred_pbe0xdm\n")
        for r in rows:
            fh.write(
                f"{r['index']},{r['true_pbe0']},{r['true_xdm']},{r['true_pbe0xdm']},"
                f"{r['pred_pbe0']},{r['pred_xdm']},{r['pred_pbe0xdm']}\n"
            )
    print(f"Wrote {csv_path}")

    if not args.no_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        for ax, (label, true_key, pred_key) in zip(
            axes,
            (("PBE0", "true_pbe0", "pred_pbe0"), ("XDM", "true_xdm", "pred_xdm"), ("PBE0+XDM", "true_pbe0xdm", "pred_pbe0xdm")),
        ):
            true = np.array([r[true_key] for r in rows])
            pred = np.array([r[pred_key] for r in rows])
            ax.scatter(true, pred, s=6, alpha=0.4, linewidths=0)
            lo, hi = min(true.min(), pred.min()), max(true.max(), pred.max())
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
            ax.set_xlabel(f"true {label} (kcal/mol)")
            ax.set_ylabel(f"predicted {label} (kcal/mol)")
            ax.set_title(f"{label} (R2={report[label]['r2']:.3f})")
        fig.tight_layout()
        plot_path = Path(args.output_dir) / "intermolecular_scatter.png"
        fig.savefig(plot_path, dpi=150)
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
