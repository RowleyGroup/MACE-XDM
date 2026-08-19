###########################################################################################
# GPU-resident Langevin MD benchmark over a directory of water-ball xyz files, for the
# MACEPBE0 and MACEXDM models.
#
# MACEPBE0 is a plain MACE checkpoint (trained on PBE0 energies/forces), loaded directly
# with `torch.load` -- no `mace.calculators.mace_off`/ASE calculator wrapper involved.
# MACEXDM is MACEPBE0 plus torchanipbe0's MLXDM_2x dispersion-energy module added on top,
# mirroring how ANIPBE0_2x_MLXDM_2x composes ANIPBE0_2x electronic energies with MLXDM_2x
# dispersion energies. Both the MACE electronic term and the MLXDM dispersion term are
# invoked as plain `torch.nn.Module` calls (never wrapped in `ase.Atoms`/`Calculator`), so
# species/coordinates/velocities/masses/forces stay as torch tensors on --device for the
# whole 100000-step run: the only host<->device syncs are the periodic time-series/
# trajectory logging points (controlled by --ts-interval/--traj-interval), not every step.
#
# Built on top of scripts/gpu_md.py's ASE-free `System`/`langevin` (see that file for the
# neighbor-list/integrator details); this script only adds the optional MLXDM dispersion
# term and the water-ball benchmarking CLI/output layout.
###########################################################################################

import argparse
import csv
import glob
import os
import re
import sys
import time
from typing import List, Optional, Sequence, Tuple

import ase.io
import torch
from ase import Atoms
from ase.io.trajectory import Trajectory

from scripts.gpu_md import System, init_maxwell_boltzmann, langevin

DEFAULT_XYZ_DIR = "/lustre06/project/6060902/crowley/timing/xyz"

# Hartree -> eV (CODATA 2018), used only for the MLXDM dispersion term, which reports
# energies in Hartree following torchani/torchanipbe0 convention.
HARTREE_TO_EV = 27.211386245988

T = 298.15
MD_STEPS = 100000
TIMESTEP_FS = 1.0
FRICTION = 0.002
WARMUP_STEPS = 10


class MLXDMDispersion:
    """Wraps torchanipbe0's MLXDM_2x dispersion-energy module so it can be added on top of
    a MACE electronic-energy model. Invoked as a plain module call (never through an ASE
    Calculator), following the exact autograd pattern torchanipbe0's own ASE calculator
    uses internally (torchanipbe0/ase.py: requires_grad_ -> forward -> -grad(energy, x))
    so forces come from the same computational graph the reference ASE calculator would
    produce, just without the per-step ase.Atoms round trip."""

    def __init__(self, device: torch.device):
        import torchanipbe0
        from torchanipbe0 import utils as ani_utils

        self.device = device
        self.model = torchanipbe0.models.MLXDM_2x(device).eval()
        self._species_to_tensor = ani_utils.ChemicalSymbolsToInts(self.model.species)
        self._species: Optional[torch.Tensor] = None

    def set_structure(self, symbols: Sequence[str]) -> None:
        self._species = self._species_to_tensor(list(symbols)).unsqueeze(0).to(self.device)

    def energy_forces(self, positions: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """positions: [n_atoms, 3] tensor, Angstrom (any dtype/device). Returns
        (energy [eV], forces [n_atoms, 3] float64 eV/Angstrom)."""
        coords = positions.detach().to(torch.float32).unsqueeze(0).to(self.device)
        coords.requires_grad_(True)
        energy_eV = self.model((self._species, coords)).energies.sum() * HARTREE_TO_EV
        (grad,) = torch.autograd.grad(energy_eV, coords)
        forces_eV = (-grad.squeeze(0)).to(torch.float64)
        return float(energy_eV.item()), forces_eV


class DispersionSystem(System):
    """A gpu_md.System that optionally adds an MLXDMDispersion contribution on top of the
    MACE model's energy/forces. Forces are additive by superposition, so the MACE and
    dispersion terms can be evaluated independently and summed."""

    def __init__(self, *args, dispersion: Optional[MLXDMDispersion] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dispersion = dispersion

    def energy_forces(self, compute_stress: bool = False):
        energy, forces, stress = super().energy_forces(compute_stress=compute_stress)
        if self.dispersion is not None:
            e_disp, f_disp = self.dispersion.energy_forces(self.positions)
            energy += e_disp
            forces = forces + f_disp
        return energy, forces, stress


def build_model(model_name: str, model_path: str, device: str):
    device = torch.device(device)
    mace_model = torch.load(model_path, map_location=device, weights_only=False)
    mace_model = mace_model.to(device).eval()

    param_types = {p.device.type for p in mace_model.parameters()}
    if param_types != {device.type}:
        raise RuntimeError(
            f"Expected all model parameters on device type '{device.type}', "
            f"found {param_types}"
        )
    print(f"Confirmed model parameters on device: {device}")

    dispersion = MLXDMDispersion(device) if model_name == "macexdm" else None
    return mace_model, dispersion, device


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
    match = re.search(r"(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else None


def make_callback(system: DispersionSystem, symbols: Sequence[str], csv_path: str,
                   traj_path: str, start_time: float, ts_interval: int, traj_interval: int):
    """Build a single langevin() callback that logs a time-series row every ts_interval
    steps and a trajectory frame every traj_interval steps. Both operations read
    `system.positions`/`.cpu()` it for the trajectory frame, so they are the only points
    in the run that synchronize the GPU -- everything else stays device-resident."""
    ts_file = open(csv_path, mode="w", newline="")
    writer = csv.writer(ts_file)
    writer.writerow(["step", "elapsed_seconds", "epot_eV", "temperature_K"])
    traj = Trajectory(traj_path, "w")

    def callback(step, energy, e_kin, temperature):
        log_now = step % ts_interval == 0
        write_frame = step % traj_interval == 0
        if not (log_now or write_frame):
            return

        if log_now:
            elapsed = time.perf_counter() - start_time
            writer.writerow([step, f"{elapsed:.4f}", f"{energy:.6f}", f"{temperature:.2f}"])
            ts_file.flush()

        if write_frame:
            atoms = Atoms(symbols=symbols, positions=system.positions.detach().cpu().numpy())
            traj.write(atoms)

    def close():
        ts_file.close()
        traj.close()

    return callback, close


def run_benchmark(mace_model, dispersion: Optional[MLXDMDispersion], device: torch.device,
                   xyz_files: List[str], csv_filename: str, outdir: str,
                   traj_interval: int, ts_interval: int):
    os.makedirs(outdir, exist_ok=True)

    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["xyz_file", "n_atoms", "n_waters", "time_seconds"])

        print(f"Starting GPU-resident Langevin benchmark over {len(xyz_files)} "
              f"water-ball structures, {MD_STEPS} steps each...")

        for path in xyz_files:
            atoms = ase.io.read(path)
            symbols = atoms.get_chemical_symbols()
            atomic_numbers = atoms.get_atomic_numbers()
            positions = atoms.get_positions()
            n_atoms = len(atoms)
            n_waters = n_waters_from_filename(path)
            if n_waters is None:
                n_waters = n_atoms // 3

            name = os.path.basename(path)
            stem = os.path.splitext(name)[0]
            print(f"Running MD for {name} ({n_atoms} atoms)...")

            if dispersion is not None:
                dispersion.set_structure(symbols)

            system = DispersionSystem(
                mace_model, atomic_numbers, positions,
                pbc=(False, False, False), device=device, dispersion=dispersion,
            )
            init_maxwell_boltzmann(system, temperature_K=T, seed=0)

            # Untimed warm-up steps: absorb one-time costs (CUDA context/kernel JIT,
            # cuDNN/cuBLAS autotuning for these tensor shapes) so they don't get charged
            # to the timed run below, especially on the first structure of a fresh process.
            langevin(system, dt_fs=TIMESTEP_FS, n_steps=WARMUP_STEPS, temperature_K=T,
                     friction=FRICTION, seed=0)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            time_initial = time.perf_counter()
            callback, close_logs = make_callback(
                system, symbols,
                os.path.join(outdir, f"{stem}.timeseries.csv"),
                os.path.join(outdir, f"{stem}.traj"),
                time_initial, ts_interval, traj_interval,
            )

            langevin(system, dt_fs=TIMESTEP_FS, n_steps=MD_STEPS, temperature_K=T,
                     friction=FRICTION, seed=1, callback=callback)
            close_logs()

            time_final = time.perf_counter()
            elapsed_time = time_final - time_initial

            print(f"Completed {name} in {elapsed_time:.2f} seconds "
                  f"({MD_STEPS / elapsed_time:.1f} steps/s).")

            writer.writerow([name, n_atoms, n_waters, elapsed_time])
            f.flush()

    print(f"All timings successfully saved to {csv_filename}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model", choices=["macepbe0", "macexdm"],
                         help="macepbe0: electronic-only MACE model trained on PBE0. "
                              "macexdm: macepbe0 + MLXDM_2x dispersion (electronic+dispersion)")
    parser.add_argument("--model-path", required=True,
                         help="Path to the MACEPBE0 MACE checkpoint (.model file). Used "
                              "for both 'macepbe0' and 'macexdm' (macexdm additionally "
                              "adds torchanipbe0's bundled MLXDM_2x dispersion module).")
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

    mace_model, dispersion, device = build_model(args.model, args.model_path, args.device)

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
        mace_model,
        dispersion,
        device,
        xyz_files,
        args.csv or default_csv,
        args.outdir,
        args.traj_interval,
        args.ts_interval,
    )


if __name__ == "__main__":
    main()
