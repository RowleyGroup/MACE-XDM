###########################################################################################
# GPU-resident geometry minimization and velocity-Verlet MD for MACE models.
#
# Runs directly against a loaded MACE `torch.nn.Module` -- no ASE `Atoms`,
# `Calculator` or `Dynamics` classes are used anywhere in this file. Atomic
# positions, velocities, forces and the neighbor list all live as `torch`
# tensors on the target device (CPU or CUDA) for the whole trajectory; the
# only place we ever touch the CPU is the periodic neighbor-list rebuild
# (which needs numpy for `matscipy`, the same backend MACE's own ASE
# calculator uses under the hood) and the occasional logging line.
#
# Usage:
#   python scripts/gpu_md.py --model model_swa.model --device cuda
#   python scripts/gpu_md.py --model model_swa.model --xyz start.xyz --device cuda
###########################################################################################

from __future__ import annotations

import argparse
import dataclasses
import re
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from mace.data.neighborhood import get_neighborhood
from mace.tools.torch_tools import to_one_hot
from mace.tools.utils import AtomicNumberTable, atomic_numbers_to_indices

# ---------------------------------------------------------------------------
# Unit system: eV (energy), Angstrom (length), amu (mass), fs (time).
# The conversion factors below are derived from SI/CODATA constants so that
# `F = m * a` and `E_kin = 1/2 m v^2` hold in this unit system, exactly like
# ASE's internal units -- but computed here from scratch so nothing in this
# file depends on ASE.
# ---------------------------------------------------------------------------
_EV_J = 1.602176634e-19  # J per eV
_AMU_KG = 1.66053906660e-27  # kg per amu
_ANGSTROM_M = 1.0e-10  # m per Angstrom
_FS_S = 1.0e-15  # s per fs

# a [Angstrom / fs^2] = ACC_FACTOR * F [eV / Angstrom] / m [amu]
ACC_FACTOR = (_EV_J / _ANGSTROM_M / _AMU_KG) * (_FS_S**2) / _ANGSTROM_M
# E_kin [eV] = 0.5 * KE_FACTOR * sum(m [amu] * v [Angstrom/fs]^2)
KE_FACTOR = (_AMU_KG * _ANGSTROM_M**2 / _FS_S**2) / _EV_J
KB_EV = 8.617333262e-5  # Boltzmann constant, eV / K

# Standard atomic weights, indexed by atomic number Z (index 0 unused).
ATOMIC_MASSES = [0.0] + [
    1.008, 4.002602, 6.94, 9.012183, 10.81, 12.011, 14.007, 15.999,
    18.998403, 20.1797, 22.989769, 24.305, 26.981538, 28.085, 30.973762, 32.06,
    35.45, 39.948, 39.0983, 40.078, 44.955908, 47.867, 50.9415, 51.9961,
    54.938044, 55.845, 58.933194, 58.6934, 63.546, 65.38, 69.723, 72.63,
    74.921595, 78.971, 79.904, 83.798, 85.4678, 87.62, 88.90584, 91.224,
    92.90637, 95.95, 97.90721, 101.07, 102.9055, 106.42, 107.8682, 112.414,
    114.818, 118.71, 121.76, 127.6, 126.90447, 131.293, 132.905452, 137.327,
    138.90547, 140.116, 140.90766, 144.242, 144.91276, 150.36, 151.964, 157.25,
    158.92535, 162.5, 164.93033, 167.259, 168.93422, 173.054, 174.9668, 178.49,
    180.94788, 183.84, 186.207, 190.23, 192.217, 195.084, 196.966569, 200.592,
    204.38, 207.2, 208.9804, 208.98243, 209.98715, 222.01758, 223.01974, 226.02541,
    227.02775, 232.0377, 231.03588, 238.02891, 237.04817, 244.06421, 243.06138, 247.07035,
    247.07031, 251.07959, 252.083, 257.09511, 258.09843, 259.101, 262.11,
]

CHEMICAL_SYMBOLS = [
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr",
]
SYMBOL_TO_Z = {s: z for z, s in enumerate(CHEMICAL_SYMBOLS)}


def symbols_to_atomic_numbers(symbols: Sequence[str]) -> np.ndarray:
    return np.array([SYMBOL_TO_Z[s] for s in symbols], dtype=np.int64)


# ---------------------------------------------------------------------------
# Minimal extended-XYZ reader (no ASE). Supports a `Lattice="..."` key in the
# comment line for periodic cells; falls back to an isolated (non-periodic)
# system otherwise.
# ---------------------------------------------------------------------------
def read_xyz(path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    n_atoms = int(lines[0].split()[0])
    comment = lines[1]
    cell = None
    match = re.search(r'Lattice="([^"]+)"', comment)
    if match:
        cell = np.array([float(x) for x in match.group(1).split()]).reshape(3, 3)
    symbols: List[str] = []
    positions = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        parts = lines[2 + i].split()
        symbols.append(parts[0])
        positions[i] = [float(parts[1]), float(parts[2]), float(parts[3])]
    atomic_numbers = symbols_to_atomic_numbers(symbols)
    return atomic_numbers, positions, cell


def demo_structure() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3x3x3 conventional FCC copper supercell, built without ASE."""
    a = 3.6149  # Cu lattice constant, Angstrom
    basis = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]
    )
    reps = 3
    cells = np.array(
        [(i, j, k) for i in range(reps) for j in range(reps) for k in range(reps)]
    )
    frac = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 3) / reps
    cell = np.eye(3) * a * reps
    positions = frac @ cell
    atomic_numbers = np.full(positions.shape[0], SYMBOL_TO_Z["Cu"], dtype=np.int64)
    return atomic_numbers, positions, cell


# ---------------------------------------------------------------------------
# GPU-resident atomic system wrapping a MACE model.
# ---------------------------------------------------------------------------
class System:
    """Holds state (positions, cell, neighbor list) for one structure and
    evaluates energy/forces with a MACE model, staying on `device` between
    calls. The neighbor list is rebuilt with a Verlet skin so it doesn't need
    to be recomputed every step."""

    def __init__(
        self,
        model: torch.nn.Module,
        atomic_numbers: np.ndarray,
        positions: np.ndarray,
        cell: Optional[np.ndarray] = None,
        pbc: Tuple[bool, bool, bool] = (False, False, False),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        skin: float = 2.0,
    ):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.model_dtype = next(model.parameters()).dtype
        self.r_max = float(model.r_max.item())
        self.skin = skin
        self.pbc = tuple(bool(p) for p in pbc)

        z_table = AtomicNumberTable(model.atomic_numbers.tolist())
        indices = atomic_numbers_to_indices(atomic_numbers, z_table=z_table)
        node_indices = torch.tensor(indices, dtype=torch.long, device=self.device).unsqueeze(-1)
        self.node_attrs = to_one_hot(node_indices, num_classes=len(z_table)).to(
            self.model_dtype
        )

        self.n_atoms = len(atomic_numbers)
        self.atomic_numbers = atomic_numbers
        self.masses = torch.tensor(
            [ATOMIC_MASSES[z] for z in atomic_numbers],
            dtype=torch.float64,
            device=self.device,
        )

        # Physics state kept in float64 regardless of the model's compute dtype.
        self.positions = torch.tensor(positions, dtype=torch.float64, device=self.device)
        self.cell = torch.tensor(
            cell if cell is not None else np.zeros((3, 3)),
            dtype=torch.float64,
            device=self.device,
        )
        self.velocities = torch.zeros_like(self.positions)

        self.batch = torch.zeros(self.n_atoms, dtype=torch.long, device=self.device)
        self.ptr = torch.tensor([0, self.n_atoms], dtype=torch.long, device=self.device)
        self.head = torch.zeros(1, dtype=torch.long, device=self.device)

        self._edge_index: Optional[torch.Tensor] = None
        self._shifts: Optional[torch.Tensor] = None
        self._unit_shifts: Optional[torch.Tensor] = None
        self._ref_positions: Optional[torch.Tensor] = None
        self.rebuild_neighbors()

    def rebuild_neighbors(self) -> None:
        """Recompute the neighbor list (cutoff = r_max + skin) on CPU via
        matscipy and upload it back to `device`. This is the only point in
        the whole minimize/MD loop that syncs GPU -> CPU."""
        pos_np = self.positions.detach().cpu().numpy()
        cell_np = self.cell.detach().cpu().numpy()
        edge_index, shifts, unit_shifts, _ = get_neighborhood(
            positions=pos_np,
            cutoff=self.r_max + self.skin,
            pbc=self.pbc,
            cell=cell_np.copy(),
        )
        self._edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device)
        self._shifts = torch.tensor(shifts, dtype=torch.float64, device=self.device)
        self._unit_shifts = torch.tensor(
            unit_shifts, dtype=torch.float64, device=self.device
        )
        self._ref_positions = self.positions.clone()

    def maybe_rebuild_neighbors(self) -> bool:
        """Verlet-list skin check: rebuild once any atom could plausibly have
        entered/left another atom's cutoff shell, i.e. when twice the largest
        displacement since the last build exceeds the skin distance."""
        disp = (self.positions - self._ref_positions).norm(dim=-1).max()
        if 2.0 * disp.item() > self.skin:
            self.rebuild_neighbors()
            return True
        return False

    def energy_forces(self, compute_stress: bool = False) -> Tuple[float, torch.Tensor, Optional[torch.Tensor]]:
        """Evaluate the model at the current positions. Returns
        (energy [eV], forces [n_atoms, 3] in eV/Angstrom, stress or None),
        all as float64 tensors detached from the autograd graph."""
        positions = self.positions.detach().clone().to(self.model_dtype)
        cell = self.cell.detach().clone().to(self.model_dtype).unsqueeze(0)
        data = {
            "positions": positions,
            "node_attrs": self.node_attrs,
            "edge_index": self._edge_index,
            "shifts": self._shifts.to(self.model_dtype),
            "unit_shifts": self._unit_shifts.to(self.model_dtype),
            "cell": cell,
            "batch": self.batch,
            "ptr": self.ptr,
            "head": self.head,
        }
        out = self.model(
            data,
            training=False,
            compute_force=True,
            compute_stress=compute_stress,
        )
        energy = float(out["energy"].detach().item())
        forces = out["forces"].detach().to(torch.float64)
        stress = out["stress"].detach().to(torch.float64) if compute_stress and out["stress"] is not None else None
        return energy, forces, stress

    def accelerations(self, forces: torch.Tensor) -> torch.Tensor:
        return ACC_FACTOR * forces / self.masses.unsqueeze(-1)

    def kinetic_energy(self) -> float:
        return 0.5 * KE_FACTOR * float((self.masses.unsqueeze(-1) * self.velocities**2).sum().item())

    def temperature(self) -> float:
        ndof = max(3 * self.n_atoms - 3, 1)
        return 2.0 * self.kinetic_energy() / (ndof * KB_EV)


# ---------------------------------------------------------------------------
# FIRE minimizer (Bitzek et al., PRL 97, 170201 (2006)), operating purely on
# `system`'s torch tensors.
# ---------------------------------------------------------------------------
def minimize_fire(
    system: System,
    fmax: float = 0.05,
    steps: int = 500,
    dt_start: float = 0.1,
    dt_max: float = 1.0,
    n_min: int = 5,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    alpha_start: float = 0.1,
    f_alpha: float = 0.99,
    callback: Optional[Callable[[int, float, float], None]] = None,
) -> float:
    dt = dt_start
    alpha = alpha_start
    n_pos = 0
    velocities = torch.zeros_like(system.positions)

    energy, forces, _ = system.energy_forces()
    for step in range(steps):
        fmax_now = forces.abs().max().item()
        if callback is not None:
            callback(step, energy, fmax_now)
        if fmax_now < fmax:
            break

        power = float((forces * velocities).sum().item())
        if power > 0.0:
            speed = velocities.norm(dim=-1, keepdim=True)
            fhat = forces / forces.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            velocities = (1.0 - alpha) * velocities + alpha * speed * fhat
            n_pos += 1
            if n_pos > n_min:
                dt = min(dt * f_inc, dt_max)
                alpha *= f_alpha
        else:
            velocities.zero_()
            n_pos = 0
            dt *= f_dec
            alpha = alpha_start

        velocities = velocities + dt * system.accelerations(forces)
        system.positions = system.positions + dt * velocities

        system.maybe_rebuild_neighbors()
        energy, forces, _ = system.energy_forces()

    return energy


# ---------------------------------------------------------------------------
# Velocity-Verlet NVE integrator.
# ---------------------------------------------------------------------------
def init_maxwell_boltzmann(system: System, temperature_K: float, seed: Optional[int] = None) -> None:
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    std = torch.sqrt(KB_EV * temperature_K / system.masses / KE_FACTOR).unsqueeze(-1)
    noise = torch.randn((system.n_atoms, 3), generator=gen).to(system.device)
    velocities = std * noise
    # Remove center-of-mass momentum.
    com_v = (system.masses.unsqueeze(-1) * velocities).sum(dim=0) / system.masses.sum()
    velocities = velocities - com_v
    # Rescale to the exact target temperature.
    system.velocities = velocities
    current_t = system.temperature()
    if current_t > 0:
        system.velocities *= (temperature_K / current_t) ** 0.5


def velocity_verlet(
    system: System,
    dt_fs: float,
    n_steps: int,
    callback: Optional[Callable[[int, float, float, float], None]] = None,
) -> None:
    energy, forces, _ = system.energy_forces()
    accel = system.accelerations(forces)

    for step in range(n_steps):
        if callback is not None:
            e_kin = system.kinetic_energy()
            callback(step, energy, e_kin, system.temperature())

        system.velocities = system.velocities + 0.5 * dt_fs * accel
        system.positions = system.positions + dt_fs * system.velocities

        system.maybe_rebuild_neighbors()
        energy, forces, _ = system.energy_forces()
        accel = system.accelerations(forces)

        system.velocities = system.velocities + 0.5 * dt_fs * accel


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to a MACE model file")
    parser.add_argument("--xyz", default=None, help="Optional (extended) XYZ file to load")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fmax", type=float, default=0.03, help="Force convergence, eV/Angstrom")
    parser.add_argument("--min-steps", type=int, default=200)
    parser.add_argument("--md-steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=1.0, help="MD timestep, fs")
    parser.add_argument("--temperature", type=float, default=300.0, help="Initial temperature, K")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    model = torch.load(args.model, map_location=args.device, weights_only=False)

    if args.xyz is not None:
        atomic_numbers, positions, cell = read_xyz(args.xyz)
        pbc = (True, True, True) if cell is not None else (False, False, False)
    else:
        atomic_numbers, positions, cell = demo_structure()
        pbc = (True, True, True)

    system = System(
        model=model,
        atomic_numbers=atomic_numbers,
        positions=positions,
        cell=cell,
        pbc=pbc,
        device=args.device,
    )

    print(f"System: {system.n_atoms} atoms on {system.device}, r_max={system.r_max:.2f} A")

    print("\n--- FIRE minimization ---")

    def min_cb(step: int, energy: float, fmax_now: float) -> None:
        if step % 10 == 0:
            print(f"step {step:4d}  E = {energy:12.4f} eV  fmax = {fmax_now:8.4f} eV/A")

    e_final = minimize_fire(system, fmax=args.fmax, steps=args.min_steps, callback=min_cb)
    print(f"Converged energy: {e_final:.4f} eV")

    print("\n--- Velocity-Verlet MD (NVE) ---")
    init_maxwell_boltzmann(system, temperature_K=args.temperature, seed=args.seed)

    def md_cb(step: int, energy: float, e_kin: float, temperature: float) -> None:
        if step % 10 == 0:
            print(
                f"step {step:4d}  E_pot = {energy:12.4f} eV  E_kin = {e_kin:8.4f} eV  "
                f"E_tot = {energy + e_kin:12.4f} eV  T = {temperature:7.1f} K"
            )

    velocity_verlet(system, dt_fs=args.dt, n_steps=args.md_steps, callback=md_cb)


if __name__ == "__main__":
    main()
