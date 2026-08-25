#!/usr/bin/env python3
"""Convert an XDMHDF5Dataset-schema file (molecule-name-keyed groups with
atomic_numbers/coordinates/energies/forces/M1/M2/M3/Veff datasets, as read by
mace_run_train_xdm) into an ASE extended-xyz file that plain mace_prepare_data
/ mace_run_train can read.

XDMHDF5Dataset itself never reads "energies"/"forces" -- it only pulls
atomic_numbers/coordinates and the XDM moment targets -- so this is a
separate, from-scratch read of the same file for the total-energy/forces
labels sitting alongside them.

Units: per mace/cli/eval_intermolecular_xdm.py's --energy_key help text (same
"energies" dataset name), the reference energy is in Hartree; coordinates are
documented (XDMHDF5Dataset's docstring) as Angstrom. Forces are assumed
Hartree/Angstrom (the ANI-1x/ANI-2x convention, and the only unit consistent
with the energies+coordinates units for a physically meaningful gradient) --
this was NOT independently confirmed against the data itself, since nothing
else in the repo touches these "forces" datasets to compare against.
CHECK THIS before trusting stage-1 training: e.g. pick one small molecule,
compute a numerical gradient of "energies" wrt "coordinates" for a couple of
displaced conformers, and see if it's ~consistent with the stored "forces"
under the Hartree/Angstrom assumption.

Output units are eV / Angstrom / eV-per-Angstrom, matching MACE's convention.
"""

import argparse
import sys

import h5py
import numpy as np
from ase.data import chemical_symbols

HARTREE_TO_EV = 27.211386245988


def find_leaf_groups(node, path=""):
    items = list(node.items())
    has_subgroup = any(isinstance(v, h5py.Group) for _, v in items)
    has_dataset = any(isinstance(v, h5py.Dataset) for _, v in items)
    leaves = []
    if has_dataset and not has_subgroup:
        leaves.append(path)
    if has_subgroup:
        for name, child in items:
            if isinstance(child, h5py.Group):
                child_path = f"{path}/{name}" if path else name
                leaves.extend(find_leaf_groups(child, child_path))
    return leaves


def write_frame(fh, symbols, positions, energy_ev, forces_ev_ang, config_type):
    n_atoms = len(symbols)
    fh.write(f"{n_atoms}\n")
    fh.write(
        'Lattice="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0" '
        "Properties=species:S:1:pos:R:3:REF_forces:R:3 "
        f'REF_energy={float(energy_ev)!r} pbc="F F F" config_type={config_type}\n'
    )
    for sym, pos, force in zip(symbols, positions, forces_ev_ang):
        p = [float(x) for x in pos]
        fo = [float(x) for x in force]
        fh.write(
            f"{sym} {p[0]!r} {p[1]!r} {p[2]!r} "
            f"{fo[0]!r} {fo[1]!r} {fo[2]!r}\n"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("hdf5_path", help="Path to anipbe0-2x_mlxdm2.hdf5 (or similar)")
    ap.add_argument("out_extxyz", help="Output extended-xyz path")
    ap.add_argument(
        "--limit-molecules",
        type=int,
        default=None,
        help="Only convert the first N molecule groups (for a quick smoke test).",
    )
    ap.add_argument(
        "--config-type",
        default="Default",
        help="config_type tag written into every frame "
        "(matches --config_type_weights in mace_prepare_data).",
    )
    args = ap.parse_args()

    f = h5py.File(args.hdf5_path, "r")
    leaves = find_leaf_groups(f)
    if args.limit_molecules is not None:
        leaves = leaves[: args.limit_molecules]
    print(f"Found {len(leaves)} molecule groups", file=sys.stderr)

    n_conf_written = 0
    with open(args.out_extxyz, "w", encoding="utf-8") as out:
        for i, leaf in enumerate(leaves):
            grp = f[leaf]
            atomic_numbers = np.asarray(grp["atomic_numbers"][()])
            symbols = [chemical_symbols[z] for z in atomic_numbers]
            coords = np.asarray(grp["coordinates"], dtype=np.float64)  # [n_conf, n_atoms, 3], Angstrom
            energies_hartree = np.asarray(grp["energies"], dtype=np.float64)  # [n_conf]
            forces_hartree_ang = np.asarray(grp["forces"], dtype=np.float64)  # [n_conf, n_atoms, 3]

            n_conf = coords.shape[0]
            for c in range(n_conf):
                write_frame(
                    out,
                    symbols,
                    coords[c],
                    energies_hartree[c] * HARTREE_TO_EV,
                    forces_hartree_ang[c] * HARTREE_TO_EV,
                    args.config_type,
                )
            n_conf_written += n_conf

            if (i + 1) % 500 == 0:
                print(
                    f"  {i + 1}/{len(leaves)} molecules, "
                    f"{n_conf_written} conformers written",
                    file=sys.stderr,
                )

    print(f"Done: {n_conf_written} conformers -> {args.out_extxyz}", file=sys.stderr)


if __name__ == "__main__":
    main()
