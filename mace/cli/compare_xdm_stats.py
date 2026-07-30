###########################################################################################
# Compare the fixed MLXDM ANI-2x reference mean/std (the default standardization
# used by mace_run_train_xdm --element_stats mlxdm_2x) against a dataset's actual
# empirical per-element statistics. A mismatch here isn't a code bug, but it makes
# the standardized-space loss harder to interpret and can slow/destabilize
# training -- the network first has to compensate for a miscalibrated prior
# before it's really fitting structure.
###########################################################################################

import argparse
import json
import logging
from typing import List, Optional

import numpy as np
from ase.data import chemical_symbols

from mace.cli.run_train_xdm import expand_file_patterns
from mace.data import (
    compute_xdm_element_statistics,
    discover_atomic_number_table,
    discover_molecule_names,
    mlxdm_2x_reference_stats,
)
from mace.tools import AtomicNumberTable, setup_logger

DEFAULT_TARGET_KEYS = ("M1", "M2", "M3", "Veff")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_files", required=True, nargs="+")
    parser.add_argument(
        "--split_file",
        default=None,
        help="Restrict to this file's 'train' molecule list (the "
        "<name>_split.json written by mace_run_train_xdm), matching what "
        "training actually standardizes against. Recommended.",
    )
    parser.add_argument(
        "--max_molecules",
        type=int,
        default=2000,
        help="Randomly subsample to at most this many molecules for speed "
        "(the underlying statistics computation is not vectorized, so "
        "scanning millions of conformers would take a very long time; a "
        "few thousand molecules is plenty to estimate mean/std reliably). "
        "Set to 0 to use every molecule.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target_keys", nargs=4, default=list(DEFAULT_TARGET_KEYS))
    parser.add_argument("--species_key", default="atomic_numbers")
    parser.add_argument("--coordinates_key", default="coordinates")
    parser.add_argument(
        "--atomic_numbers",
        default=None,
        help="Comma-separated list, e.g. '1,6,7,8'. If omitted, discovered "
        "from the data (can be slow on a huge dataset -- prefer passing this "
        "explicitly, e.g. '1,6,7,8,9,16,17' for the usual H,C,N,O,F,S,Cl set).",
    )
    return parser


def main():
    args = build_parser().parse_args()
    setup_logger(level=logging.INFO)

    target_keys = list(args.target_keys)
    train_files = expand_file_patterns(args.train_files)

    if args.atomic_numbers is not None:
        zs = sorted(int(z) for z in args.atomic_numbers.split(","))
        z_table = AtomicNumberTable(zs)
    else:
        logging.info("Discovering elements present (pass --atomic_numbers to skip this)...")
        z_table = discover_atomic_number_table(train_files, species_key=args.species_key)
    logging.info(f"Elements: {z_table.zs}")

    molecule_names: Optional[List[str]] = None
    if args.split_file is not None:
        with open(args.split_file, "r", encoding="utf-8") as f:
            molecule_names = json.load(f)["train"]
        logging.info(f"Restricting to {len(molecule_names)} molecules from --split_file's 'train' set")
    else:
        molecule_names = discover_molecule_names(train_files)
        logging.info(f"Found {len(molecule_names)} molecules across --train_files")

    if args.max_molecules and len(molecule_names) > args.max_molecules:
        rng = np.random.RandomState(args.seed)
        molecule_names = list(
            rng.choice(molecule_names, size=args.max_molecules, replace=False)
        )
        logging.info(f"Subsampled to {len(molecule_names)} molecules for speed")

    logging.info("Computing empirical per-element statistics (this reads every "
                 "conformer of each selected molecule once)...")
    empirical = compute_xdm_element_statistics(
        train_files,
        z_table=z_table,
        target_keys=target_keys,
        species_key=args.species_key,
        coordinates_key=args.coordinates_key,
        molecule_names=molecule_names,
    )

    try:
        fixed = mlxdm_2x_reference_stats(z_table)
    except ValueError as e:
        print(f"\nCannot compare against the fixed MLXDM reference: {e}\n")
        fixed = None

    print(
        f"\n{'elem':<5}{'prop':<6}{'emp_mean':>12}{'fix_mean':>12}{'mean_z':>9}"
        f"{'emp_std':>12}{'fix_std':>12}{'std_ratio':>10}  flag"
    )
    for idx, z in enumerate(z_table.zs):
        symbol = chemical_symbols[z]
        for j, key in enumerate(target_keys):
            emp_mean = empirical["mean"][idx, j]
            emp_std = empirical["std"][idx, j]
            flag = ""
            if fixed is not None:
                fix_mean = fixed["mean"][idx, j]
                fix_std = fixed["std"][idx, j]
                mean_z = (fix_mean - emp_mean) / emp_std if emp_std > 0 else float("nan")
                std_ratio = fix_std / emp_std if emp_std > 0 else float("nan")
                if abs(mean_z) > 1.0:
                    flag += "mean off by >1 empirical std; "
                if std_ratio == std_ratio and (std_ratio > 2.0 or std_ratio < 0.5):
                    flag += "std off by >2x"
            else:
                fix_mean = fix_std = mean_z = std_ratio = float("nan")
            print(
                f"{symbol:<5}{key:<6}{emp_mean:>12.4f}{fix_mean:>12.4f}{mean_z:>9.2f}"
                f"{emp_std:>12.4f}{fix_std:>12.4f}{std_ratio:>10.3f}  {flag}"
            )

    counts_str = ", ".join(
        f"{chemical_symbols[z]}={int(c)}" for z, c in zip(z_table.zs, empirical["counts"])
    )
    print(f"\nAtom instances used per element: {counts_str}")
    print(
        "\nmean_z: how many empirical std's away the fixed MLXDM mean is from "
        "this dataset's actual mean (near 0 = well matched; e.g. 2.0 means the "
        "fixed prior's center is 2 empirical std's off)."
    )
    print(
        "std_ratio: fixed_std / empirical_std (near 1 = well matched; >1 means "
        "the fixed prior assumes more spread than this data has, so "
        "standardized targets are smaller than intended -- less gradient "
        "signal; <1 means the opposite -- larger standardized targets, larger "
        "gradients, more risk of instability)."
    )


if __name__ == "__main__":
    main()
