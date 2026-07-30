###########################################################################################
# Training script for AtomicXDMMACE: predicting per-atom XDM dispersion coefficients
# (M1, M2, M3, Veff) from an ANI-style HDF5 dataset of labeled molecules.
###########################################################################################

import argparse
import glob
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from mace.data import (
    XDMHDF5Dataset,
    compute_xdm_element_statistics,
    default_mlxdm_2x_atomic_number_table,
    discover_atomic_number_table,
    discover_molecule_names,
    mlxdm_2x_reference_stats,
)
from mace.modules import (
    AtomicXDMMACE,
    build_atomic_xdm_mace_from_args,
    gate_dict,
    interaction_classes,
)
from mace.modules.utils import compute_avg_num_neighbors
from mace.tools import (
    AtomicNumberTable,
    compute_mae,
    compute_rmse,
    count_parameters,
    init_device,
    set_default_dtype,
    set_seeds,
    setup_logger,
)
from mace.tools.torch_geometric.dataloader import DataLoader
from mace.tools.torch_tools import to_numpy

DEFAULT_TARGET_KEYS = ("M1", "M2", "M3", "Veff")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a MACE model to predict per-atom XDM dispersion "
        "coefficients (M1, M2, M3, Veff) from an ANI-style HDF5 dataset."
    )

    # Data
    parser.add_argument(
        "--train_files",
        required=True,
        nargs="+",
        help="One or more ANI-style HDF5 files and/or glob patterns (e.g. "
        "'data/pbe0xdm-ani2x_*.hdf5'), covering the training pool. The same "
        "molecule name may appear in multiple files (e.g. successive "
        "active-learning batches); their conformers are pooled together.",
    )
    parser.add_argument(
        "--valid_files",
        default=None,
        nargs="+",
        help="Optional separate HDF5 file(s)/glob(s) for validation. If "
        "omitted, a fraction of the molecule names found across --train_files "
        "is held out (by molecule identity, not by individual conformer, so "
        "no molecule's conformers are split across train and valid).",
    )
    parser.add_argument(
        "--valid_fraction",
        type=float,
        default=0.1,
        help="Fraction of molecule names held out for validation when "
        "--valid_files is not given.",
    )
    parser.add_argument(
        "--test_files",
        default=None,
        nargs="+",
        help="Optional separate HDF5 file(s)/glob(s) for a final held-out "
        "test set. If omitted, a further fraction of the molecule names is "
        "held out for testing (disjoint from both training and validation).",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.1,
        help="Fraction of molecule names held out for testing when "
        "--test_files is not given. This is a single merged file's typical "
        "use case: pass one --train_files master file and let "
        "--valid_fraction/--test_fraction carve out validation and test "
        "molecules from it.",
    )
    parser.add_argument(
        "--element_stats",
        default="mlxdm_2x",
        choices=["mlxdm_2x", "dataset"],
        help="Source of the per-element mean/std used to standardize "
        "targets. 'mlxdm_2x' (default) uses the fixed reference statistics "
        "for H, C, N, O, S, F, Cl from RowleyGroup/MLXDM's ANI-2x dispersion "
        "model (not computed from your training data). 'dataset' computes "
        "mean/std from the training molecules instead, and is required for "
        "elements outside that set of 7.",
    )
    parser.add_argument(
        "--target_keys",
        nargs=4,
        default=list(DEFAULT_TARGET_KEYS),
        metavar=("M1_KEY", "M2_KEY", "M3_KEY", "VEFF_KEY"),
        help="HDF5 dataset keys (within each molecule group) for the four "
        "per-atom XDM targets.",
    )
    parser.add_argument(
        "--species_key",
        default="atomic_numbers",
        help="HDF5 key for atomic species: an integer atomic-number array "
        "(default 'atomic_numbers'), or an array of element symbols "
        "(e.g. 'species').",
    )
    parser.add_argument(
        "--coordinates_key",
        default="coordinates",
        help="HDF5 key for atomic coordinates (Angstrom).",
    )
    parser.add_argument(
        "--atomic_numbers",
        default=None,
        help="Comma-separated list of atomic numbers, e.g. '1,6,7,8'. If omitted, "
        "the set of elements is discovered from the training files.",
    )

    # Model hyperparameters
    parser.add_argument("--r_max", type=float, default=5.0)
    parser.add_argument("--num_bessel", type=int, default=8)
    parser.add_argument("--num_polynomial_cutoff", type=int, default=5)
    parser.add_argument("--max_ell", type=int, default=3)
    parser.add_argument(
        "--interaction",
        default="RealAgnosticResidualInteractionBlock",
        choices=list(interaction_classes.keys()),
    )
    parser.add_argument(
        "--interaction_first",
        default="RealAgnosticInteractionBlock",
        choices=list(interaction_classes.keys()),
    )
    parser.add_argument("--num_interactions", type=int, default=2)
    parser.add_argument("--hidden_irreps", default="128x0e + 128x1o")
    parser.add_argument("--MLP_irreps", default="16x0e")
    parser.add_argument("--correlation", type=int, default=3)
    parser.add_argument("--gate", default="silu", choices=list(gate_dict.keys()))
    parser.add_argument(
        "--avg_num_neighbors",
        type=float,
        default=None,
        help="If omitted, computed from the training set.",
    )

    # Optimization
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--valid_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=5e-7)
    parser.add_argument("--max_num_epochs", type=int, default=200)
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Epochs without validation improvement before early stopping.",
    )
    parser.add_argument(
        "--lr_factor",
        type=float,
        default=0.5,
        help="Factor for ReduceLROnPlateau LR scheduler.",
    )
    parser.add_argument(
        "--lr_scheduler_patience",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--loss_weights",
        nargs=4,
        type=float,
        default=[1.0, 1.0, 1.0, 1.0],
        metavar=("W_M1", "W_M2", "W_M3", "W_VEFF"),
        help="Per-property weights for the standardized-space MSE loss.",
    )

    # Misc / bookkeeping
    parser.add_argument("--name", default="xdm_model")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "xpu"])
    parser.add_argument(
        "--default_dtype", default="float64", choices=["float32", "float64"]
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--checkpoints_dir", default="checkpoints")
    parser.add_argument("--model_dir", default="models")
    parser.add_argument(
        "--results_dir",
        default="results",
        help="Where to save the molecule-name train/valid/test split and "
        "final test-set metrics, as JSON.",
    )
    parser.add_argument(
        "--restart_latest",
        action="store_true",
        help="Resume from checkpoints_dir/<name>_latest.pt if it exists.",
    )
    parser.add_argument("--eval_interval", type=int, default=1)
    return parser


def expand_file_patterns(patterns: List[str]) -> List[str]:
    files = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No files matched '{pattern}'")
        files.extend(matches)
    return files


def resolve_z_table(args: argparse.Namespace, train_files: List[str]) -> AtomicNumberTable:
    if args.atomic_numbers is not None:
        zs = [int(z) for z in args.atomic_numbers.split(",")]
        return AtomicNumberTable(sorted(zs))
    if args.element_stats == "mlxdm_2x":
        # Fixed to MLXDM's 7 supported elements; skips scanning every
        # molecule in (potentially very large) --train_files for its elements.
        return default_mlxdm_2x_atomic_number_table()
    return discover_atomic_number_table(train_files, species_key=args.species_key)


def split_molecule_names(
    args: argparse.Namespace,
    train_files: List[str],
    valid_files: Optional[List[str]],
    test_files: Optional[List[str]],
) -> Dict[str, List[str]]:
    """Split molecule names into disjoint train/valid/test sets.

    Explicit --valid_files/--test_files are used as-is (and removed from the
    training pool); everything else is drawn by molecule identity from
    --train_files, so a single merged file can be split into all three parts
    at once via --valid_fraction/--test_fraction.
    """
    all_names = discover_molecule_names(train_files)
    remaining = set(all_names)

    valid_names: Optional[List[str]] = None
    if args.valid_files is not None:
        valid_names = discover_molecule_names(valid_files)
        remaining -= set(valid_names)

    test_names: Optional[List[str]] = None
    if args.test_files is not None:
        test_names = discover_molecule_names(test_files)
        remaining -= set(test_names)

    rng = np.random.RandomState(args.seed)
    shuffled = sorted(remaining)
    rng.shuffle(shuffled)

    if valid_names is None:
        n_valid = max(1, int(round(len(all_names) * args.valid_fraction)))
        valid_names = shuffled[:n_valid]
        shuffled = shuffled[n_valid:]
    if test_names is None:
        n_test = max(1, int(round(len(all_names) * args.test_fraction)))
        test_names = shuffled[:n_test]
        shuffled = shuffled[n_test:]

    train_names = shuffled
    return {"train": train_names, "valid": valid_names, "test": test_names}


def batch_loss_and_metrics(
    model: AtomicXDMMACE,
    batch,
    device: torch.device,
    loss_weights: torch.Tensor,
):
    batch = batch.to(device)
    output = model(batch.to_dict())
    target_standardized = model.xdm_reference.standardize(
        batch.xdm_targets, batch.node_attrs
    )
    error = output["xdm_standardized"] - target_standardized
    loss = torch.mean(loss_weights * error**2)
    return loss, output, target_standardized


def evaluate(
    model: AtomicXDMMACE,
    data_loader: DataLoader,
    device: torch.device,
    loss_weights: torch.Tensor,
    num_properties: int,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_atoms = 0
    physical_errors = [[] for _ in range(num_properties)]
    with torch.no_grad():
        for batch in data_loader:
            loss, output, _ = batch_loss_and_metrics(model, batch, device, loss_weights)
            n_atoms = batch.node_attrs.shape[0]
            total_loss += loss.item() * n_atoms
            total_atoms += n_atoms
            physical_error = to_numpy(output["xdm_atomic"] - batch.xdm_targets.to(device))
            for i in range(num_properties):
                physical_errors[i].append(physical_error[:, i])
    metrics = {"loss": total_loss / max(total_atoms, 1)}
    for i in range(num_properties):
        errs = np.concatenate(physical_errors[i])
        metrics[f"mae_{i}"] = compute_mae(errs)
        metrics[f"rmse_{i}"] = compute_rmse(errs)
    return metrics


def main():
    args = build_parser().parse_args()

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    setup_logger(level=logging.INFO, tag=args.name, directory=args.log_dir)

    set_seeds(args.seed)
    set_default_dtype(args.default_dtype)
    device = init_device(args.device)

    target_keys = list(args.target_keys)
    train_files = expand_file_patterns(args.train_files)
    valid_files = (
        expand_file_patterns(args.valid_files) if args.valid_files is not None else None
    )
    test_files = (
        expand_file_patterns(args.test_files) if args.test_files is not None else None
    )
    logging.info(f"Training files: {len(train_files)}")
    if valid_files is not None:
        logging.info(f"Validation files: {len(valid_files)}")
    if test_files is not None:
        logging.info(f"Test files: {len(test_files)}")

    z_table = resolve_z_table(args, train_files)
    logging.info(f"Atomic number table: {z_table}")

    names = split_molecule_names(args, train_files, valid_files, test_files)
    logging.info(
        f"Training molecules: {len(names['train'])}, "
        f"validation molecules: {len(names['valid'])}, "
        f"test molecules: {len(names['test'])}"
    )
    with open(Path(args.results_dir) / f"{args.name}_split.json", "w", encoding="utf-8") as f:
        json.dump(names, f, indent=2)

    if args.element_stats == "mlxdm_2x":
        element_stats = mlxdm_2x_reference_stats(z_table)
        logging.info(
            "Using fixed MLXDM ANI-2x reference mean/std (H, C, N, O, S, F, Cl), "
            "not computed from the training data."
        )
    else:
        element_stats = compute_xdm_element_statistics(
            train_files,
            z_table=z_table,
            target_keys=target_keys,
            species_key=args.species_key,
            coordinates_key=args.coordinates_key,
            molecule_names=names["train"],
        )
    logging.info(f"Per-element mean:\n{element_stats['mean']}")
    logging.info(f"Per-element std:\n{element_stats['std']}")

    train_dataset = XDMHDF5Dataset(
        train_files,
        z_table=z_table,
        r_max=args.r_max,
        target_keys=target_keys,
        species_key=args.species_key,
        coordinates_key=args.coordinates_key,
        molecule_names=names["train"],
    )
    valid_dataset = XDMHDF5Dataset(
        valid_files if valid_files is not None else train_files,
        z_table=z_table,
        r_max=args.r_max,
        target_keys=target_keys,
        species_key=args.species_key,
        coordinates_key=args.coordinates_key,
        molecule_names=names["valid"],
    )
    test_dataset = XDMHDF5Dataset(
        test_files if test_files is not None else train_files,
        z_table=z_table,
        r_max=args.r_max,
        target_keys=target_keys,
        species_key=args.species_key,
        coordinates_key=args.coordinates_key,
        molecule_names=names["test"],
    )
    logging.info(
        f"Training conformers: {len(train_dataset)}, "
        f"validation conformers: {len(valid_dataset)}, "
        f"test conformers: {len(test_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.valid_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.valid_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    avg_num_neighbors = args.avg_num_neighbors
    if avg_num_neighbors is None:
        avg_num_neighbors = compute_avg_num_neighbors(train_loader)
    logging.info(f"Average number of neighbors: {avg_num_neighbors:.3f}")

    model = build_atomic_xdm_mace_from_args(
        args=vars(args),
        z_table=z_table,
        avg_num_neighbors=avg_num_neighbors,
        element_means=element_stats["mean"],
        element_stds=element_stats["std"],
    ).to(device)
    logging.info(f"Number of parameters: {count_parameters(model)}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=args.lr_factor, patience=args.lr_scheduler_patience
    )
    loss_weights = torch.tensor(
        args.loss_weights, dtype=torch.get_default_dtype(), device=device
    )

    start_epoch = 0
    best_valid_loss = float("inf")
    epochs_without_improvement = 0

    latest_path = Path(args.checkpoints_dir) / f"{args.name}_latest.pt"
    best_path = Path(args.checkpoints_dir) / f"{args.name}_best.pt"

    if args.restart_latest and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_valid_loss = checkpoint["best_valid_loss"]
        logging.info(f"Restarted from {latest_path} at epoch {start_epoch}")

    def save_checkpoint(path: Path, epoch: int):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_valid_loss": best_valid_loss,
                "args": vars(args),
                "z_table": z_table.zs,
                "element_mean": element_stats["mean"].tolist(),
                "element_std": element_stats["std"].tolist(),
                "avg_num_neighbors": avg_num_neighbors,
            },
            path,
        )

    for epoch in range(start_epoch, args.max_num_epochs):
        model.train()
        epoch_start = time.time()
        train_loss = 0.0
        train_atoms = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss, _, _ = batch_loss_and_metrics(model, batch, device, loss_weights)
            loss.backward()
            optimizer.step()
            n_atoms = batch.node_attrs.shape[0]
            train_loss += loss.item() * n_atoms
            train_atoms += n_atoms
        train_loss /= max(train_atoms, 1)

        if epoch % args.eval_interval == 0 or epoch == args.max_num_epochs - 1:
            metrics = evaluate(
                model, valid_loader, device, loss_weights, len(target_keys)
            )
            lr_before = optimizer.param_groups[0]["lr"]
            scheduler.step(metrics["loss"])
            lr_after = optimizer.param_groups[0]["lr"]
            if lr_after < lr_before:
                logging.info(f"Reducing learning rate: {lr_before:.3e} -> {lr_after:.3e}")
            elapsed = time.time() - epoch_start
            mae_str = ", ".join(
                f"{key}_mae={metrics[f'mae_{i}']:.4f}"
                for i, key in enumerate(target_keys)
            )
            logging.info(
                f"Epoch {epoch}: train_loss={train_loss:.6f}, "
                f"valid_loss={metrics['loss']:.6f}, {mae_str}, lr={lr_after:.3e}, "
                f"time={elapsed:.1f}s"
            )

            save_checkpoint(latest_path, epoch)
            if metrics["loss"] < best_valid_loss:
                best_valid_loss = metrics["loss"]
                epochs_without_improvement = 0
                save_checkpoint(best_path, epoch)
                torch.save(model, Path(args.model_dir) / f"{args.name}.model")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    logging.info(
                        f"Early stopping at epoch {epoch} "
                        f"(no improvement for {args.patience} evaluations)."
                    )
                    break

    logging.info(f"Training complete. Best validation loss: {best_valid_loss:.6f}")
    logging.info(f"Best model checkpoint: {best_path}")
    logging.info(f"Best full model: {Path(args.model_dir) / f'{args.name}.model'}")

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, loss_weights, len(target_keys))
    test_mae_str = ", ".join(
        f"{key}_mae={test_metrics[f'mae_{i}']:.4f}" for i, key in enumerate(target_keys)
    )
    logging.info(f"Test loss={test_metrics['loss']:.6f}, {test_mae_str}")
    with open(Path(args.results_dir) / f"{args.name}_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "target_keys": target_keys,
                **{
                    f"{key}_{metric}": test_metrics[f"{metric}_{i}"]
                    for i, key in enumerate(target_keys)
                    for metric in ("mae", "rmse")
                },
                "loss": test_metrics["loss"],
            },
            f,
            indent=2,
        )

    # Written only on normal completion (max_num_epochs reached or early
    # stopping), never on a mid-loop kill (e.g. SLURM walltime) -- a chained
    # job script can check for this file to decide whether to resubmit.
    (Path(args.results_dir) / f"{args.name}_DONE").touch()


if __name__ == "__main__":
    main()
