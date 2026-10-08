###########################################################################################
# Training script for a plain (non-XDM) MACE energy/forces model, reading
# directly from the same ANI-style HDF5 schema mace_run_train_xdm reads --
# no separate xyz-conversion/mace_prepare_data pass needed. Works whether or
# not a given leaf group also carries XDM labels (M1/M2/M3/Veff): this only
# needs atomic_numbers/coordinates/energies/forces, so the same file --
# XDM-labeled subset and all -- can feed both mace_run_train_xdm and this
# script directly.
###########################################################################################

import argparse
import glob
import json
import logging
import math
import time
from functools import partial
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from e3nn import o3

from mace.data import (
    ANIHDF5EnergyForcesDataset,
    discover_atomic_number_table,
    estimate_atomic_energies_linear_regression,
    split_molecule_names,
)
from mace.modules import ScaleShiftMACE, WeightedEnergyForcesLoss, gate_dict, interaction_classes
from mace.modules.utils import compute_avg_num_neighbors, compute_mean_rms_energy_forces
from mace.modules.wrapper_ops import CuEquivarianceConfig, OEQConfig
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
from mace.tools.arg_parser import str2bool
from mace.tools.torch_geometric.dataloader import DataLoader
from mace.tools.torch_tools import to_numpy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a plain MACE energy/forces model directly from an "
        "ANI-style HDF5 dataset (the same file mace_run_train_xdm reads)."
    )

    # Data
    parser.add_argument(
        "--train_files",
        required=True,
        nargs="+",
        help="One or more ANI-style HDF5 files and/or glob patterns, covering "
        "the training pool. The same molecule name may appear in multiple "
        "files (e.g. successive active-learning batches); their conformers "
        "are pooled together.",
    )
    parser.add_argument(
        "--valid_files",
        default=None,
        nargs="+",
        help="Optional separate HDF5 file(s)/glob(s) for validation. If "
        "omitted, a fraction of the molecule names found across --train_files "
        "is held out (by molecule identity, not by individual conformer).",
    )
    parser.add_argument("--valid_fraction", type=float, default=0.1)
    parser.add_argument(
        "--test_files",
        default=None,
        nargs="+",
        help="Optional separate HDF5 file(s)/glob(s) for a final held-out "
        "test set. If omitted, a further fraction of the molecule names is "
        "held out for testing.",
    )
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument(
        "--energy_key", default="energies", help="HDF5 key for total energy."
    )
    parser.add_argument(
        "--forces_key", default="forces", help="HDF5 key for per-atom forces."
    )
    parser.add_argument(
        "--energy_forces_unit",
        default="hartree",
        choices=["hartree", "ev"],
        help="Unit energy_key/forces_key are stored in. 'hartree' (default) "
        "converts to eV / eV-per-Angstrom, matching these ANI-PBE0 datasets' "
        "convention (see scripts/convert_xdm_hdf5_to_extxyz.py's docstring "
        "for how that was established). Pass 'ev' if your file already "
        "stores eV.",
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
        help="Comma-separated list of atomic numbers, e.g. '1,6,7,8'. If "
        "omitted, the set of elements is discovered from the training files. "
        "Pass this explicitly (matching the same order mace_run_train_xdm "
        "uses) if you plan to --foundation_model warm-start an XDM fine-tune "
        "from this model later, so per-element weights land in the same "
        "element slots.",
    )

    # Model hyperparameters (mirrors mace_run_train_xdm's flags, so the two
    # can be pointed at the same data with matching architecture)
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
        default="RealAgnosticResidualInteractionBlock",
        choices=list(interaction_classes.keys()),
        help="Matches mace_run_train_xdm's own default -- keep this the same "
        "in both if you plan to --foundation_model warm-start an XDM "
        "fine-tune from this model, since a mismatched first interaction "
        "block silently weakens the backbone-tensor transplant.",
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
    parser.add_argument(
        "--enable_cueq",
        type=str2bool,
        default=False,
        help="Build with cuequivariance-accelerated tensor-product kernels "
        "instead of plain e3nn. Needs `cuequivariance`, `cuequivariance-torch`, "
        "and a CUDA-version-matched `cuequivariance-ops-torch-cuXX` package "
        "installed.",
    )
    parser.add_argument(
        "--enable_oeq",
        type=str2bool,
        default=False,
        help="Build with openequivariance-accelerated tensor-product kernels "
        "instead of plain e3nn. Mutually exclusive with --enable_cueq; if "
        "both are set, cueq wins.",
    )

    # Optimization
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--valid_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=5e-7)
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        default=None,
        help="If set, clip gradient norm to this value before each optimizer "
        "step.",
    )
    parser.add_argument("--max_num_epochs", type=int, default=200)
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Evaluations (one every --eval_interval epochs, not every "
        "epoch) without validation improvement before early stopping.",
    )
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_scheduler_patience", type=int, default=10)
    parser.add_argument("--energy_weight", type=float, default=1.0)
    parser.add_argument("--forces_weight", type=float, default=100.0)

    # Misc / bookkeeping
    parser.add_argument("--name", default="pbe0_model")
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
    return discover_atomic_number_table(train_files, species_key=args.species_key)


def dataloader_worker_init_fn(_worker_id: int, default_dtype: str) -> None:
    # torch.set_default_dtype() is per-process global state set once in
    # main(); DataLoader worker subprocesses don't reliably inherit it (e.g.
    # under the "spawn" start method), so AtomicData built inside
    # ANIHDF5EnergyForcesDataset.__getitem__ can silently come out float32
    # even when --default_dtype float64 was requested. Re-apply it in each
    # worker.
    set_default_dtype(default_dtype)


def evaluate(
    model: ScaleShiftMACE,
    data_loader: DataLoader,
    device: torch.device,
    loss_fn: torch.nn.Module,
) -> Dict[str, float]:
    # No torch.no_grad() here: forces are computed via autograd w.r.t.
    # positions inside model.forward() even when training=False, so gradient
    # tracking must stay enabled through the forward pass (matching
    # mace.tools.train.evaluate's own pattern) -- .backward()/optimizer.step()
    # are simply never called here.
    model.eval()
    total_loss = 0.0
    total_atoms = 0
    energy_errors_per_atom = []
    force_errors = []
    for batch in data_loader:
        batch = batch.to(device)
        output = model(batch.to_dict(), training=False, compute_force=True)
        loss = loss_fn(batch, output)
        n_atoms = batch.node_attrs.shape[0]
        total_loss += loss.item() * n_atoms
        total_atoms += n_atoms
        graph_sizes = to_numpy(batch.ptr[1:] - batch.ptr[:-1])
        e_err = to_numpy(output["energy"] - batch.energy) / graph_sizes
        energy_errors_per_atom.append(e_err)
        f_err = to_numpy(output["forces"] - batch.forces).reshape(-1)
        force_errors.append(f_err)
    energy_errors_per_atom = np.concatenate(energy_errors_per_atom)
    force_errors = np.concatenate(force_errors)
    return {
        "loss": total_loss / max(total_atoms, 1),
        "mae_e_per_atom": compute_mae(energy_errors_per_atom),
        "rmse_e_per_atom": compute_rmse(energy_errors_per_atom),
        "mae_f": compute_mae(force_errors),
        "rmse_f": compute_rmse(force_errors),
    }


def main():
    args = build_parser().parse_args()

    # DataLoader workers hand tensors back to the main process via mmap'd
    # shared-memory files under the default "file_descriptor" strategy. On
    # Alliance Canada compute nodes /dev/shm is small and not scaled to
    # --mem-per-cpu, so a full pass with num_workers>0 (e.g. inside
    # compute_mean_rms_energy_forces) can die with
    # "unable to mmap ...: Cannot allocate memory". "file_system" instead
    # uses named temp files and isn't subject to that cap.
    torch.multiprocessing.set_sharing_strategy("file_system")

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    setup_logger(level=logging.INFO, tag=args.name, directory=args.log_dir)

    set_seeds(args.seed)
    set_default_dtype(args.default_dtype)
    device = init_device(args.device)

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

    target_keys = [args.energy_key, args.forces_key]
    names = split_molecule_names(
        train_files,
        valid_files,
        test_files,
        target_keys,
        valid_fraction=args.valid_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    logging.info(
        f"Training molecules: {len(names['train'])}, "
        f"validation molecules: {len(names['valid'])}, "
        f"test molecules: {len(names['test'])}"
    )
    with open(Path(args.results_dir) / f"{args.name}_split.json", "w", encoding="utf-8") as f:
        json.dump(names, f, indent=2)

    atomic_energies = estimate_atomic_energies_linear_regression(
        train_files,
        z_table=z_table,
        energy_key=args.energy_key,
        species_key=args.species_key,
        molecule_names=names["train"],
        energy_forces_unit=args.energy_forces_unit,
    )
    logging.info(f"Atomic energies (E0s, eV): {atomic_energies}")

    train_dataset = ANIHDF5EnergyForcesDataset(
        train_files,
        z_table=z_table,
        r_max=args.r_max,
        energy_key=args.energy_key,
        forces_key=args.forces_key,
        species_key=args.species_key,
        coordinates_key=args.coordinates_key,
        energy_forces_unit=args.energy_forces_unit,
        molecule_names=names["train"],
    )
    valid_dataset = ANIHDF5EnergyForcesDataset(
        valid_files if valid_files is not None else train_files,
        z_table=z_table,
        r_max=args.r_max,
        energy_key=args.energy_key,
        forces_key=args.forces_key,
        species_key=args.species_key,
        coordinates_key=args.coordinates_key,
        energy_forces_unit=args.energy_forces_unit,
        molecule_names=names["valid"],
    )
    test_dataset = ANIHDF5EnergyForcesDataset(
        test_files if test_files is not None else train_files,
        z_table=z_table,
        r_max=args.r_max,
        energy_key=args.energy_key,
        forces_key=args.forces_key,
        species_key=args.species_key,
        coordinates_key=args.coordinates_key,
        energy_forces_unit=args.energy_forces_unit,
        molecule_names=names["test"],
    )
    logging.info(
        f"Training conformers: {len(train_dataset)}, "
        f"validation conformers: {len(valid_dataset)}, "
        f"test conformers: {len(test_dataset)}"
    )

    worker_init_fn = partial(dataloader_worker_init_fn, default_dtype=args.default_dtype)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.valid_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.valid_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
    )

    avg_num_neighbors = args.avg_num_neighbors
    if avg_num_neighbors is None:
        avg_num_neighbors = compute_avg_num_neighbors(train_loader)
    logging.info(f"Average number of neighbors: {avg_num_neighbors:.3f}")

    atomic_inter_shift, atomic_inter_scale = compute_mean_rms_energy_forces(
        train_loader, atomic_energies
    )
    # compute_mean_rms_energy_forces returns (mean, std) over heads; single
    # head here, so take the scalar.
    atomic_inter_shift = float(np.asarray(atomic_inter_shift).reshape(-1)[0])
    atomic_inter_scale = float(np.asarray(atomic_inter_scale).reshape(-1)[0])
    logging.info(
        f"Atomic inter shift/scale: {atomic_inter_shift:.6f} / {atomic_inter_scale:.6f}"
    )

    if args.enable_cueq and args.enable_oeq:
        logging.warning(
            "Both --enable_cueq and --enable_oeq are set; using cueq. Pass only "
            "one of the two."
        )
        args.enable_oeq = False

    cueq_config = None
    oeq_config = None
    if args.enable_cueq:
        cueq_config = CuEquivarianceConfig(
            enabled=True, layout="ir_mul", group="O3_e3nn", optimize_all=True,
            conv_fusion=True,
        )
    elif args.enable_oeq:
        oeq_config = OEQConfig(enabled=True, optimize_all=True)

    model = ScaleShiftMACE(
        r_max=args.r_max,
        num_bessel=args.num_bessel,
        num_polynomial_cutoff=args.num_polynomial_cutoff,
        max_ell=args.max_ell,
        interaction_cls=interaction_classes[args.interaction],
        interaction_cls_first=interaction_classes[args.interaction_first],
        num_interactions=args.num_interactions,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps(args.hidden_irreps),
        MLP_irreps=o3.Irreps(args.MLP_irreps),
        atomic_energies=atomic_energies,
        avg_num_neighbors=avg_num_neighbors,
        atomic_numbers=z_table.zs,
        correlation=args.correlation,
        gate=gate_dict[args.gate],
        atomic_inter_scale=atomic_inter_scale,
        atomic_inter_shift=atomic_inter_shift,
        # Deliberately not exposed via CLI and always False: AtomicXDMMACE has
        # no --use_reduced_cg flag and always builds as if it were False, so
        # leaving this at MACE's own True default would desync the
        # product-basis tensor shapes from an XDM model built on the same
        # data, silently breaking a --foundation_model warm start later.
        use_reduced_cg=False,
        cueq_config=cueq_config,
        oeq_config=oeq_config,
    ).to(device)
    logging.info(f"Number of parameters: {count_parameters(model)}")
    if args.clip_grad_norm is not None:
        logging.info(f"Gradient norm clipping enabled at {args.clip_grad_norm}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=args.lr_factor, patience=args.lr_scheduler_patience
    )
    loss_fn = WeightedEnergyForcesLoss(
        energy_weight=args.energy_weight, forces_weight=args.forces_weight
    ).to(device)

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
        epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            logging.warning(
                "Checkpoint has no scheduler_state_dict (saved before this was "
                "tracked) -- the LR scheduler's plateau-detection state starts "
                "fresh from this restart rather than continuing where it left off."
            )
        logging.info(f"Restarted from {latest_path} at epoch {start_epoch}")

    def save_checkpoint(path: Path, epoch: int):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "best_valid_loss": best_valid_loss,
                "epochs_without_improvement": epochs_without_improvement,
                "args": vars(args),
                "z_table": z_table.zs,
                "atomic_energies": atomic_energies.tolist(),
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
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch.to_dict(), training=True, compute_force=True)
            loss = loss_fn(batch, output)
            loss.backward()
            if args.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            n_atoms = batch.node_attrs.shape[0]
            train_loss += loss.item() * n_atoms
            train_atoms += n_atoms
        train_loss /= max(train_atoms, 1)

        if epoch % args.eval_interval == 0 or epoch == args.max_num_epochs - 1:
            metrics = evaluate(model, valid_loader, device, loss_fn)
            if math.isnan(metrics["loss"]):
                # Same failure mode as mace_run_train_xdm: best_path is only
                # written below when loss improves, so NaN loss means it's
                # never created and the unconditional torch.load(best_path)
                # after the loop crashes with a confusing FileNotFoundError.
                raise RuntimeError(
                    f"Validation loss is NaN at epoch {epoch} -- training has "
                    "diverged. Try a lower --lr, or enable --clip_grad_norm."
                )
            lr_before = optimizer.param_groups[0]["lr"]
            scheduler.step(metrics["loss"])
            lr_after = optimizer.param_groups[0]["lr"]
            if lr_after < lr_before:
                logging.info(f"Reducing learning rate: {lr_before:.3e} -> {lr_after:.3e}")
            elapsed = time.time() - epoch_start
            logging.info(
                f"Epoch {epoch}: train_loss={train_loss:.6f}, "
                f"valid_loss={metrics['loss']:.6f}, "
                f"rmse_e_per_atom={metrics['rmse_e_per_atom']:.4f}, "
                f"rmse_f={metrics['rmse_f']:.4f}, lr={lr_after:.3e}, "
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
    test_metrics = evaluate(model, test_loader, device, loss_fn)
    logging.info(
        f"Test loss={test_metrics['loss']:.6f}, "
        f"rmse_e_per_atom={test_metrics['rmse_e_per_atom']:.4f}, "
        f"rmse_f={test_metrics['rmse_f']:.4f}"
    )
    with open(Path(args.results_dir) / f"{args.name}_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    # Written only on normal completion (max_num_epochs reached or early
    # stopping), never on a mid-loop kill (e.g. SLURM walltime) -- a chained
    # job script can check for this file to decide whether to resubmit.
    (Path(args.results_dir) / f"{args.name}_DONE").touch()


if __name__ == "__main__":
    main()
