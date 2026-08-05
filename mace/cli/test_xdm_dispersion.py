###########################################################################################
# Judge an AtomicXDMMACE model by the physical quantity it actually exists to
# predict well: the XDM dispersion energy (and its C6/C8/C10 components), not
# just raw M1/M2/M3/Veff regression error. Computes both from the model's
# predicted per-atom moments and from the dataset's true per-atom moments
# (via the same differentiable XDMDispersionEnergy formula used in training/
# production), on a per-molecule basis, and reports how well predicted
# dispersion energy tracks the true one -- overall and split by C6/C8/C10, so
# it's clear which term is driving any remaining error.
###########################################################################################

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from mace.cli.run_train_xdm import expand_file_patterns
from mace.data import XDMHDF5Dataset, mlxdm_2x_polarizability_reference
from mace.modules import load_xdm_model
from mace.modules.xdm_dispersion import XDMDispersionEnergy
from mace.tools import AtomicNumberTable, init_device, set_default_dtype, setup_logger
from mace.tools.torch_geometric.dataloader import DataLoader

DEFAULT_TARGET_KEYS = ("M1", "M2", "M3", "Veff")
HARTREE_TO_KCAL_MOL = 627.5094740631


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Path to either a .model file or a training checkpoint "
        "(<name>_{latest,best}.pt).",
    )
    parser.add_argument("--test_files", required=True, nargs="+")
    parser.add_argument(
        "--split_file",
        default=None,
        help="Path to the <name>_split.json saved by mace_run_train_xdm; "
        "evaluates on its 'test' molecule list. Strongly recommended.",
    )
    parser.add_argument("--molecule_names", default=None, nargs="+")
    parser.add_argument("--target_keys", nargs=4, default=list(DEFAULT_TARGET_KEYS))
    parser.add_argument("--species_key", default="atomic_numbers")
    parser.add_argument("--coordinates_key", default="coordinates")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "xpu"])
    parser.add_argument("--default_dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Molecules per batch. The dispersion energy computation is "
        "O(atoms_per_molecule^2), so unlike mace_test_xdm this stays modest.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--max_conformers",
        type=int,
        default=None,
        help="If set, randomly subsample to at most this many conformers "
        "(fixed seed) for a quick look instead of the full test set.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dispersion_cutoff", type=float, default=14.0, help="XDM pairwise cutoff, Angstrom."
    )
    parser.add_argument("--bj_a1", type=float, default=0.4186)
    parser.add_argument("--bj_a2", type=float, default=2.6791)
    parser.add_argument("--output_dir", default="test_dispersion_results")
    parser.add_argument("--no_plots", action="store_true")
    return parser


def collect_dispersion_energies(
    model, dispersion_energy, loader, device
) -> Dict[str, np.ndarray]:
    keys = ("total", "e6", "e8", "e10")
    pred: Dict[str, List[np.ndarray]] = {k: [] for k in keys}
    true: Dict[str, List[np.ndarray]] = {k: [] for k in keys}
    n_atoms: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            data = batch.to_dict()
            num_graphs = int(data["ptr"].numel() - 1)

            pred_xdm_atomic = model(data)["xdm_atomic"]
            true_xdm_atomic = data["xdm_targets"]

            pred_e = dispersion_energy(
                positions=data["positions"],
                node_attrs=data["node_attrs"],
                batch=data["batch"],
                num_graphs=num_graphs,
                xdm_atomic=pred_xdm_atomic,
                return_components=True,
            )
            true_e = dispersion_energy(
                positions=data["positions"],
                node_attrs=data["node_attrs"],
                batch=data["batch"],
                num_graphs=num_graphs,
                xdm_atomic=true_xdm_atomic,
                return_components=True,
            )
            for k in keys:
                pred[k].append(pred_e[k].cpu().numpy())
                true[k].append(true_e[k].cpu().numpy())
            counts = torch.bincount(data["batch"], minlength=num_graphs)
            n_atoms.append(counts.cpu().numpy())

    out = {f"pred_{k}": np.concatenate(pred[k]) for k in keys}
    out.update({f"true_{k}": np.concatenate(true[k]) for k in keys})
    out["n_atoms"] = np.concatenate(n_atoms)
    return out


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    error = pred - true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    pearson_r = (
        float(np.corrcoef(pred, true)[0, 1]) if len(pred) > 1 and pred.std() > 0 else float("nan")
    )
    return {
        "n": int(len(pred)),
        "mae_hartree": mae,
        "rmse_hartree": rmse,
        "mae_kcal_mol": mae * HARTREE_TO_KCAL_MOL,
        "rmse_kcal_mol": rmse * HARTREE_TO_KCAL_MOL,
        "r2": r2,
        "pearson_r": pearson_r,
    }


def main():
    args = build_parser().parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    setup_logger(level=logging.INFO)

    set_default_dtype(args.default_dtype)
    device = init_device(args.device)

    target_keys = list(args.target_keys)
    model = load_xdm_model(args.model, device=device)
    z_table_zs = model.atomic_numbers.tolist()
    r_max = float(model.r_max.item())
    logging.info(f"Loaded model with elements {z_table_zs}, r_max={r_max}")
    z_table = AtomicNumberTable(z_table_zs)

    polar_ref = mlxdm_2x_polarizability_reference(z_table)
    dispersion_energy = XDMDispersionEnergy(
        alpha_free=polar_ref["alpha_free"],
        v_free=polar_ref["v_free"],
        cutoff=args.dispersion_cutoff,
        a1=args.bj_a1,
        a2=args.bj_a2,
    ).to(device)

    test_files = expand_file_patterns(args.test_files)

    molecule_names: Optional[List[str]] = None
    if args.split_file is not None:
        with open(args.split_file, "r", encoding="utf-8") as f:
            split = json.load(f)
        molecule_names = split["test"]
        logging.info(f"Using {len(molecule_names)} molecules from --split_file's 'test' set")
    elif args.molecule_names is not None:
        molecule_names = args.molecule_names
    else:
        logging.warning(
            "No --split_file given: evaluating on ALL molecules in --test_files, "
            "which may include molecules the model was trained on."
        )

    dataset = XDMHDF5Dataset(
        test_files,
        z_table=z_table,
        r_max=r_max,
        target_keys=target_keys,
        species_key=args.species_key,
        coordinates_key=args.coordinates_key,
        molecule_names=molecule_names,
    )
    logging.info(f"Evaluating dispersion energy on {len(dataset)} conformers")

    if args.max_conformers is not None and len(dataset) > args.max_conformers:
        rng = np.random.RandomState(args.seed)
        indices = rng.choice(len(dataset), size=args.max_conformers, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices.tolist())
        logging.info(f"Subsampled to {len(dataset)} conformers")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    energies = collect_dispersion_energies(model, dispersion_energy, loader, device)

    # C6/C8/C10's typical share of the total for nearest-neighbour intermolecular
    # interactions, for context alongside the per-component errors below.
    energy_share = {"e6": 0.60, "e8": 0.30, "e10": 0.10}

    report: Dict[str, Dict] = {}
    print(f"\n=== Dispersion energy per molecule (n={len(energies['n_atoms'])}) ===")
    print(f"{'term':<8}{'share':>7}{'n':>8}{'MAE(kcal/mol)':>16}{'RMSE(kcal/mol)':>17}{'R2':>8}{'pearson_r':>11}")
    for key, label in (("total", "Total"), ("e6", "C6"), ("e8", "C8"), ("e10", "C10")):
        m = compute_metrics(energies[f"pred_{key}"], energies[f"true_{key}"])
        report[key] = m
        share = f"{energy_share[key]*100:.0f}%" if key in energy_share else "--"
        print(
            f"{label:<8}{share:>7}{m['n']:>8}{m['mae_kcal_mol']:>16.4f}"
            f"{m['rmse_kcal_mol']:>17.4f}{m['r2']:>8.3f}{m['pearson_r']:>11.3f}"
        )

    report_path = Path(args.output_dir) / "dispersion_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {report_path}")

    if not args.no_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
        for ax, (key, label) in zip(
            axes, (("total", "Total"), ("e6", "C6"), ("e8", "C8"), ("e10", "C10"))
        ):
            true_kcal = energies[f"true_{key}"] * HARTREE_TO_KCAL_MOL
            pred_kcal = energies[f"pred_{key}"] * HARTREE_TO_KCAL_MOL
            ax.scatter(true_kcal, pred_kcal, s=4, alpha=0.3, linewidths=0)
            lo = min(true_kcal.min(), pred_kcal.min())
            hi = max(true_kcal.max(), pred_kcal.max())
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
            ax.set_xlabel(f"true {label} (kcal/mol)")
            ax.set_ylabel(f"predicted {label} (kcal/mol)")
            ax.set_title(f"{label} (R2={report[key]['r2']:.3f})")
        fig.tight_layout()
        plot_path = Path(args.output_dir) / "dispersion_scatter.png"
        fig.savefig(plot_path, dpi=150)
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
