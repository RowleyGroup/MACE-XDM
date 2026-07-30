###########################################################################################
# Diagnostic evaluation for a trained/in-progress AtomicXDMMACE: how good are its
# M1, M2, M3, Veff predictions on held-out data, right now -- not just the single
# scalar loss line already printed during training.
###########################################################################################

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from ase.data import chemical_symbols

from mace.cli.run_train_xdm import expand_file_patterns
from mace.data import XDMHDF5Dataset
from mace.modules import load_xdm_model
from mace.tools import AtomicNumberTable, init_device, set_default_dtype, setup_logger
from mace.tools.torch_geometric.dataloader import DataLoader

DEFAULT_TARGET_KEYS = ("M1", "M2", "M3", "Veff")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained (or still-training) AtomicXDMMACE on "
        "held-out data: overall and per-element MAE/RMSE/R^2 for M1, M2, M3, "
        "Veff, a comparison against the naive per-element-reference-mean "
        "baseline, and scatter plots of predicted vs. true values."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to either a .model file or a training checkpoint "
        "(<name>_{latest,best}.pt) -- e.g. the current best checkpoint of a "
        "still-running training job.",
    )
    parser.add_argument(
        "--test_files",
        required=True,
        nargs="+",
        help="HDF5 file(s)/glob(s) to evaluate on. Typically the same "
        "--train_files given to mace_run_train_xdm.",
    )
    parser.add_argument(
        "--split_file",
        default=None,
        help="Path to the <name>_split.json saved by mace_run_train_xdm. If "
        "given, evaluates on exactly its 'test' molecule list (the true "
        "held-out set never seen during training or model selection). "
        "Strongly recommended -- without this, --test_files is evaluated in "
        "full, which may include training molecules.",
    )
    parser.add_argument(
        "--molecule_names",
        default=None,
        nargs="+",
        help="Explicit list of molecule names to evaluate, as an alternative "
        "to --split_file. Ignored if --split_file is given.",
    )
    parser.add_argument("--target_keys", nargs=4, default=list(DEFAULT_TARGET_KEYS))
    parser.add_argument("--species_key", default="atomic_numbers")
    parser.add_argument("--coordinates_key", default="coordinates")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "xpu"])
    parser.add_argument(
        "--default_dtype", default="float64", choices=["float32", "float64"]
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--max_conformers",
        type=int,
        default=None,
        help="If set, randomly subsample to at most this many conformers "
        "(fixed seed) for a quick look instead of the full test set.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="test_results")
    parser.add_argument(
        "--no_plots", action="store_true", help="Skip saving scatter plots."
    )
    return parser


def collect_predictions(model, loader, device):
    element_idx_all: List[np.ndarray] = []
    pred_all: List[np.ndarray] = []
    true_all: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            data = batch.to_dict()
            out = model(data)
            element_idx_all.append(
                torch.argmax(data["node_attrs"], dim=-1).cpu().numpy()
            )
            pred_all.append(out["xdm_atomic"].cpu().numpy())
            true_all.append(data["xdm_targets"].cpu().numpy())
    return (
        np.concatenate(element_idx_all),
        np.concatenate(pred_all, axis=0),
        np.concatenate(true_all, axis=0),
    )


def compute_metrics(pred: np.ndarray, true: np.ndarray, baseline: np.ndarray) -> Dict[str, float]:
    error = pred - true
    baseline_error = baseline - true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    baseline_mae = float(np.mean(np.abs(baseline_error)))
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    skill_vs_baseline = (
        1.0 - mae / baseline_mae if baseline_mae > 0 else float("nan")
    )  # fraction of naive-baseline error removed by the model; ~0 = no better than baseline
    pearson_r = (
        float(np.corrcoef(pred, true)[0, 1]) if len(pred) > 1 and pred.std() > 0 else float("nan")
    )
    return {
        "n": int(len(pred)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": pearson_r,
        "baseline_mae": baseline_mae,
        "skill_vs_baseline": skill_vs_baseline,
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
    logging.info(f"Evaluating on {len(dataset)} conformers")

    if args.max_conformers is not None and len(dataset) > args.max_conformers:
        rng = np.random.RandomState(args.seed)
        indices = rng.choice(len(dataset), size=args.max_conformers, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices.tolist())
        logging.info(f"Subsampled to {len(dataset)} conformers")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    element_idx, pred, true = collect_predictions(model, loader, device)

    reference_mean = model.xdm_reference.mean.detach().cpu().numpy()  # [n_elements, 4]
    baseline = reference_mean[element_idx]  # [n_atoms, 4], naive per-element-mean prediction

    report: Dict[str, Dict] = {"overall": {}, "per_element": {}}

    print("\n=== Overall ===")
    print(f"{'property':<8}{'n':>10}{'MAE':>12}{'RMSE':>12}{'R2':>8}{'pearson_r':>11}{'skill':>8}")
    for j, key in enumerate(target_keys):
        m = compute_metrics(pred[:, j], true[:, j], baseline[:, j])
        report["overall"][key] = m
        print(
            f"{key:<8}{m['n']:>10}{m['mae']:>12.4f}{m['rmse']:>12.4f}"
            f"{m['r2']:>8.3f}{m['pearson_r']:>11.3f}{m['skill_vs_baseline']:>8.3f}"
        )

    print("\n=== Per element ===")
    for z_idx, z in enumerate(z_table.zs):
        symbol = chemical_symbols[z]
        mask = element_idx == z_idx
        if not mask.any():
            continue
        report["per_element"][symbol] = {}
        print(f"\n-- {symbol} (Z={z}, n_atoms={int(mask.sum())}) --")
        print(f"{'property':<8}{'n':>10}{'MAE':>12}{'RMSE':>12}{'R2':>8}{'pearson_r':>11}{'skill':>8}")
        for j, key in enumerate(target_keys):
            m = compute_metrics(pred[mask, j], true[mask, j], baseline[mask, j])
            report["per_element"][symbol][key] = m
            print(
                f"{key:<8}{m['n']:>10}{m['mae']:>12.4f}{m['rmse']:>12.4f}"
                f"{m['r2']:>8.3f}{m['pearson_r']:>11.3f}{m['skill_vs_baseline']:>8.3f}"
            )

    report_path = Path(args.output_dir) / "test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {report_path}")

    if not args.no_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(target_keys), figsize=(4.5 * len(target_keys), 4.5))
        if len(target_keys) == 1:
            axes = [axes]
        for j, (key, ax) in enumerate(zip(target_keys, axes)):
            ax.scatter(true[:, j], pred[:, j], s=2, alpha=0.2, linewidths=0)
            lo = min(true[:, j].min(), pred[:, j].min())
            hi = max(true[:, j].max(), pred[:, j].max())
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
            ax.set_xlabel(f"true {key}")
            ax.set_ylabel(f"predicted {key}")
            ax.set_title(f"{key} (R2={report['overall'][key]['r2']:.3f})")
        fig.tight_layout()
        plot_path = Path(args.output_dir) / "scatter.png"
        fig.savefig(plot_path, dpi=150)
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
