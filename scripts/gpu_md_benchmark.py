###########################################################################################
# GPU-resident MD timing benchmark for MACE-PBE0 and MACE-PBE0+MACE-XDM,
# built on the ASE-free System/langevin engine in scripts/gpu_md.py. Mirrors
# the structure of RowleyGroup/torchanipbe0's own GPU Langevin benchmark
# (same water-ball sweep, same per-structure timing CSV / time-series CSV /
# trajectory-frame outputs), but against MACE models instead of ANI-PBE0.
#
# Usage:
#   python scripts/gpu_md_benchmark.py pbe0 --pbe0-model mace-pbe0_0_s2.model
#   python scripts/gpu_md_benchmark.py pbe0_xdm \
#       --pbe0-model mace-pbe0_0_s2.model --xdm-model xdm_element.model
###########################################################################################

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_md  # noqa: E402  (needs the sys.path.insert above)

DEFAULT_XYZ_DIR = "/lustre06/project/6060902/crowley/timing/xyz"

T = 298.15
MD_STEPS = 100000
TIMESTEP_FS = 1.0
FRICTION = 0.002


def build_model(model_name: str, pbe0_model: str, xdm_model: Optional[str], device: str):
    if model_name == "pbe0":
        model = gpu_md.load_pbe0_model(pbe0_model, device=device)
    else:
        if xdm_model is None:
            raise SystemExit("pbe0_xdm requires --xdm-model")
        model = gpu_md.load_combined_model(pbe0_model, xdm_model, device=device)

    param_types = {p.device.type for p in model.parameters()}
    if param_types != {torch.device(device).type}:
        raise RuntimeError(
            f"Expected all model parameters on device type '{device}', "
            f"found {param_types}"
        )
    print(f"Confirmed model parameters on device: {device}")
    return model


def xyz_atom_count(path: str) -> int:
    with open(path) as fh:
        return int(fh.readline())


def find_xyz_files(xyz_dir: str, pattern: str) -> List[str]:
    paths = glob.glob(os.path.join(xyz_dir, pattern))
    if not paths:
        print(f"Error: no files matching '{pattern}' found in '{xyz_dir}'.")
        sys.exit(1)
    return sorted(paths, key=xyz_atom_count)


def n_waters_from_filename(path: str) -> Optional[int]:
    import re

    match = re.search(r"(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else None


class ExtendedXYZWriter:
    """Minimal append-mode extended-XYZ trajectory writer (no ASE)."""

    def __init__(self, path: str, symbols: List[str]):
        self._fh = open(path, "w", encoding="utf-8")
        self._symbols = symbols

    def write(self, positions: np.ndarray) -> None:
        n = len(self._symbols)
        self._fh.write(f"{n}\n")
        self._fh.write("Properties=species:S:1:pos:R:3\n")
        for sym, pos in zip(self._symbols, positions):
            self._fh.write(f"{sym} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}\n")

    def close(self) -> None:
        self._fh.close()


def make_callback(system, symbols, csv_path, traj_path, start_time, ts_interval, traj_interval):
    """Build a single langevin() callback that logs a time-series row every
    ts_interval steps and a trajectory frame every traj_interval steps. Both
    operations call .item()/.cpu(), so they are the only points in the run
    that synchronize the GPU -- everything else stays device-resident."""
    ts_file = open(csv_path, mode="w", newline="")
    writer = csv.writer(ts_file)
    writer.writerow(["step", "elapsed_seconds", "epot_eV", "temperature_K"])
    traj = ExtendedXYZWriter(traj_path, symbols)

    def callback(step, energy, e_kin, temperature):
        log_now = (step + 1) % ts_interval == 0
        write_frame = (step + 1) % traj_interval == 0
        if not (log_now or write_frame):
            return

        if log_now:
            elapsed = time.perf_counter() - start_time
            writer.writerow([step + 1, f"{elapsed:.4f}", f"{energy:.6f}", f"{temperature:.2f}"])
            ts_file.flush()

        if write_frame:
            traj.write(system.positions.detach().cpu().numpy())

    def close():
        ts_file.close()
        traj.close()

    return callback, close


def run_benchmark(model, device, xyz_files, csv_filename, outdir, traj_interval, ts_interval):
    os.makedirs(outdir, exist_ok=True)

    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["xyz_file", "n_atoms", "n_waters", "time_seconds"])

        print(f"Starting GPU-resident Langevin benchmark over {len(xyz_files)} "
              f"water-ball structures, {MD_STEPS} steps each...")

        for path in xyz_files:
            atomic_numbers, positions, cell = gpu_md.read_xyz(path)
            symbols = [gpu_md.CHEMICAL_SYMBOLS[z] for z in atomic_numbers]
            n_atoms = len(atomic_numbers)
            n_waters = n_waters_from_filename(path)
            if n_waters is None:
                n_waters = n_atoms // 3

            name = os.path.basename(path)
            stem = os.path.splitext(name)[0]
            print(f"Running MD for {name} ({n_atoms} atoms)...")

            system = gpu_md.System(
                model=model,
                atomic_numbers=atomic_numbers,
                positions=positions,
                cell=cell,
                pbc=(False, False, False),
                device=device,
            )
            gpu_md.init_maxwell_boltzmann(system, temperature_K=T, seed=0)

            time_initial = time.perf_counter()
            callback, close_logs = make_callback(
                system, symbols,
                os.path.join(outdir, f"{stem}.timeseries.csv"),
                os.path.join(outdir, f"{stem}.traj.xyz"),
                time_initial, ts_interval, traj_interval,
            )

            gpu_md.langevin(
                system,
                dt_fs=TIMESTEP_FS,
                n_steps=MD_STEPS,
                temperature_K=T,
                friction=FRICTION,
                seed=0,
                callback=callback,
            )
            close_logs()

            time_final = time.perf_counter()
            elapsed_time = time_final - time_initial

            print(f"Completed {name} in {elapsed_time:.2f} seconds "
                  f"({MD_STEPS / elapsed_time:.1f} steps/s).")

            writer.writerow([name, n_atoms, n_waters, elapsed_time])
            f.flush()

    print(f"All timings successfully saved to {csv_filename}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model", choices=["pbe0", "pbe0_xdm"],
                         help="pbe0: electronic-only MACE-PBE0. "
                              "pbe0_xdm: composite MACE-PBE0+MACE-XDM (electronic+dispersion)")
    parser.add_argument("--pbe0-model", required=True,
                         help="Path to the short-range MACE-PBE0 model file")
    parser.add_argument("--xdm-model", default=None,
                         help="Path to a trained AtomicXDMMACE model/checkpoint "
                              "(required for the pbe0_xdm model choice)")
    parser.add_argument("--xyz-dir", default=DEFAULT_XYZ_DIR,
                         help="Directory of water-ball xyz files (default: %(default)s)")
    parser.add_argument("--pattern", default="water_ball_*.xyz",
                         help="Glob pattern for xyz files within --xyz-dir (default: %(default)s)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only benchmark the N smallest structures (default: all)")
    parser.add_argument("--xyz-file", default=None,
                         help="Benchmark only this single xyz file, instead of scanning --xyz-dir "
                              "(used to run one water-ball size per Slurm job)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                         help="torch device (default: cuda if available, else cpu)")
    parser.add_argument("--csv", default=None,
                         help="Output CSV path (default: simulation_timings_<model>_gpu.csv)")
    parser.add_argument("--outdir", default=".",
                         help="Directory to write per-structure trajectory/time-series "
                              "files to (default: %(default)s)")
    parser.add_argument("--traj-interval", type=int, default=1000,
                         help="Steps between trajectory frames (default: %(default)s)")
    parser.add_argument("--ts-interval", type=int, default=100,
                         help="Steps between time-series log rows (default: %(default)s)")
    args = parser.parse_args()

    model = build_model(args.model, args.pbe0_model, args.xdm_model, args.device)

    if args.xyz_file:
        if not os.path.isfile(args.xyz_file):
            print(f"Error: '{args.xyz_file}' not found.")
            sys.exit(1)
        xyz_files = [args.xyz_file]
        stem = os.path.splitext(os.path.basename(args.xyz_file))[0]
        default_csv = f"simulation_timings_{args.model}_gpu_{stem}.csv"
    else:
        xyz_files = find_xyz_files(args.xyz_dir, args.pattern)
        if args.limit is not None:
            xyz_files = xyz_files[:args.limit]
        default_csv = f"simulation_timings_{args.model}_gpu.csv"

    run_benchmark(
        model,
        args.device,
        xyz_files,
        args.csv or default_csv,
        args.outdir,
        args.traj_interval,
        args.ts_interval,
    )


if __name__ == "__main__":
    main()
