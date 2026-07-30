###########################################################################################
# Run a trained AtomicXDMMACE model on new structures to predict per-atom XDM
# dispersion coefficients (M1, M2, M3, Veff).
###########################################################################################

import argparse

import numpy as np
import torch
from ase.io import read, write

from mace.data.xdm import build_xdm_atomic_data
from mace.modules import load_xdm_model
from mace.tools import AtomicNumberTable, init_device, set_default_dtype
from mace.tools.torch_geometric.batch import Batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a trained AtomicXDMMACE model on new structures to "
        "predict per-atom XDM dispersion coefficients (M1, M2, M3, Veff)."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to either a .model file (torch.save of the full "
        "nn.Module, saved to --model_dir) or a training checkpoint "
        "(<name>_{latest,best}.pt, saved to --checkpoints_dir) from "
        "run_train_xdm.py.",
    )
    parser.add_argument(
        "--configs",
        required=True,
        help="Structure file readable by ASE (xyz, extxyz, etc.), containing "
        "one or more configurations.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output extxyz file with predicted per-atom XDM coefficients "
        "attached as atom arrays.",
    )
    parser.add_argument(
        "--target_keys",
        nargs=4,
        default=["M1", "M2", "M3", "Veff"],
        metavar=("M1_KEY", "M2_KEY", "M3_KEY", "VEFF_KEY"),
        help="Names to use for the predicted per-atom arrays written to --output.",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "xpu"])
    parser.add_argument(
        "--default_dtype", default="float64", choices=["float32", "float64"]
    )
    parser.add_argument("--batch_size", type=int, default=32)
    return parser


def main():
    args = build_parser().parse_args()
    set_default_dtype(args.default_dtype)
    device = init_device(args.device)

    model = load_xdm_model(args.model, device=device)

    r_max = float(model.r_max.item())
    z_table = AtomicNumberTable(model.atomic_numbers.tolist())
    num_xdm_targets = model.num_xdm_targets

    atoms_list = read(args.configs, index=":")
    graphs = [
        build_xdm_atomic_data(
            atomic_numbers=atoms.get_atomic_numbers(),
            positions=atoms.get_positions(),
            xdm_targets=np.zeros((len(atoms), num_xdm_targets)),
            z_table=z_table,
            cutoff=r_max,
        )
        for atoms in atoms_list
    ]

    predictions = []
    for start in range(0, len(graphs), args.batch_size):
        batch = Batch.from_data_list(graphs[start : start + args.batch_size]).to(device)
        with torch.no_grad():
            output = model(batch.to_dict())
        preds = output["xdm_atomic"].cpu().numpy()
        ptr = batch.ptr.cpu().numpy()
        for i in range(len(ptr) - 1):
            predictions.append(preds[ptr[i] : ptr[i + 1]])

    for atoms, pred in zip(atoms_list, predictions):
        for i, key in enumerate(args.target_keys):
            atoms.new_array(key, pred[:, i])

    write(args.output, atoms_list)
    print(
        f"Wrote {len(atoms_list)} configuration(s) with predicted "
        f"{args.target_keys} to {args.output}"
    )


if __name__ == "__main__":
    main()
