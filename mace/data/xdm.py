###########################################################################################
# Data handling for training MACE to predict per-atom XDM dispersion coefficients
# (M1, M2, M3, Veff) from an ANI-style HDF5 dataset.
###########################################################################################

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from ase.data import atomic_numbers as ase_atomic_numbers

from mace.tools import AtomicNumberTable, atomic_numbers_to_indices, to_one_hot

from .atomic_data import AtomicData
from .neighborhood import get_neighborhood

DEFAULT_TARGET_KEYS: Tuple[str, str, str, str] = ("M1", "M2", "M3", "Veff")


def species_to_atomic_numbers(species: np.ndarray) -> np.ndarray:
    """Convert a species array to atomic numbers.

    Accepts either an already-numeric array of atomic numbers, or an array of
    element symbols (as bytes or str, the common ANI/TorchANI HDF5 convention).
    """
    species = np.asarray(species)
    if np.issubdtype(species.dtype, np.integer):
        return species.astype(np.int64)
    if np.issubdtype(species.dtype, np.floating):
        return species.astype(np.int64)
    flat_symbols = [
        s.decode("utf-8") if isinstance(s, bytes) else str(s)
        for s in species.reshape(-1).tolist()
    ]
    flat_numbers = np.array(
        [ase_atomic_numbers[s] for s in flat_symbols], dtype=np.int64
    )
    return flat_numbers.reshape(species.shape)


def build_xdm_atomic_data(
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    xdm_targets: np.ndarray,
    z_table: AtomicNumberTable,
    cutoff: float,
) -> AtomicData:
    """Build an AtomicData graph for a single molecule with per-atom XDM targets.

    xdm_targets: [n_atoms, n_properties] array, e.g. columns (M1, M2, M3, Veff).
    """
    edge_index, shifts, unit_shifts, cell = get_neighborhood(
        positions=positions, cutoff=cutoff
    )
    indices = atomic_numbers_to_indices(atomic_numbers, z_table=z_table)
    node_attrs = to_one_hot(
        torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
        num_classes=len(z_table),
    )
    return AtomicData(
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        node_attrs=node_attrs,
        positions=torch.tensor(positions, dtype=torch.get_default_dtype()),
        shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
        unit_shifts=torch.tensor(unit_shifts, dtype=torch.get_default_dtype()),
        cell=torch.tensor(cell, dtype=torch.get_default_dtype()),
        weight=torch.tensor(1.0, dtype=torch.get_default_dtype()),
        head=None,
        energy_weight=None,
        forces_weight=None,
        stress_weight=None,
        virials_weight=None,
        dipole_weight=None,
        charges_weight=None,
        polarizability_weight=None,
        forces=None,
        energy=None,
        stress=None,
        virials=None,
        dipole=None,
        charges=None,
        polarizability=None,
        elec_temp=None,
        xdm_targets=torch.tensor(xdm_targets, dtype=torch.get_default_dtype()),
    )


class XDMHDF5Dataset(torch.utils.data.Dataset):
    """Reads an ANI-style HDF5 file of molecules labeled with atomic XDM coefficients.

    Expected layout (one group per molecular formula, conformers batched together,
    matching the conventions used by TorchANI-style datasets)::

        /<formula>/<species_key>       [n_atoms] or [n_conf, n_atoms]
        /<formula>/<coordinates_key>   [n_conf, n_atoms, 3]  (Angstrom)
        /<formula>/<M1, M2, M3, Veff>  each [n_conf, n_atoms]

    ``species`` may either be a fixed per-formula array of atomic numbers/symbols
    (shape ``[n_atoms]``, shared by every conformer in the group) or vary per
    conformer (shape ``[n_conf, n_atoms]``); both are handled.
    """

    def __init__(
        self,
        file_path: str,
        z_table: AtomicNumberTable,
        r_max: float,
        target_keys: Sequence[str] = DEFAULT_TARGET_KEYS,
        species_key: str = "species",
        coordinates_key: str = "coordinates",
        groups: Optional[List[str]] = None,
    ):
        super().__init__()
        self.file_path = file_path
        self._file = None
        self.z_table = z_table
        self.r_max = r_max
        self.target_keys = list(target_keys)
        self.species_key = species_key
        self.coordinates_key = coordinates_key

        self.index: List[Tuple[str, int]] = []
        with h5py.File(file_path, "r") as f:
            group_names = list(groups) if groups is not None else list(f.keys())
            for name in group_names:
                n_conf = f[name][coordinates_key].shape[0]
                self.index.extend((name, i) for i in range(n_conf))
        self.group_names = group_names

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.file_path, "r")
        return self._file

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_file"] = None
        return state

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> AtomicData:
        group_name, conf_idx = self.index[idx]
        grp = self.file[group_name]

        species = grp[self.species_key]
        species_arr = species[conf_idx] if species.ndim == 2 else species[()]
        atomic_numbers = species_to_atomic_numbers(species_arr)

        positions = np.asarray(grp[self.coordinates_key][conf_idx], dtype=np.float64)
        targets = np.stack(
            [np.asarray(grp[key][conf_idx], dtype=np.float64) for key in self.target_keys],
            axis=-1,
        )  # [n_atoms, n_properties]

        return build_xdm_atomic_data(
            atomic_numbers=atomic_numbers,
            positions=positions,
            xdm_targets=targets,
            z_table=self.z_table,
            cutoff=self.r_max,
        )


def discover_atomic_number_table(
    file_path: str,
    species_key: str = "species",
    groups: Optional[List[str]] = None,
) -> AtomicNumberTable:
    """Scan an XDM HDF5 dataset for the set of chemical elements present."""
    zs = set()
    with h5py.File(file_path, "r") as f:
        group_names = list(groups) if groups is not None else list(f.keys())
        for name in group_names:
            atomic_numbers = species_to_atomic_numbers(f[name][species_key][()])
            zs.update(int(z) for z in np.asarray(atomic_numbers).reshape(-1))
    return AtomicNumberTable(sorted(zs))


def compute_xdm_element_statistics(
    file_path: str,
    z_table: AtomicNumberTable,
    target_keys: Sequence[str] = DEFAULT_TARGET_KEYS,
    species_key: str = "species",
    coordinates_key: str = "coordinates",
    groups: Optional[List[str]] = None,
    std_floor: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """Compute per-element mean/std of each XDM target property.

    Streams through every conformer of every group once, accumulating sums and
    sums-of-squares per element per property, then returns the resulting
    ``mean``/``std`` arrays of shape ``[len(z_table), len(target_keys)]``, ready
    to feed into ``AtomicElementReferenceBlock``/``AtomicXDMMACE``.
    """
    n_elements = len(z_table)
    n_properties = len(target_keys)
    count = np.zeros(n_elements, dtype=np.float64)
    total = np.zeros((n_elements, n_properties), dtype=np.float64)
    total_sq = np.zeros((n_elements, n_properties), dtype=np.float64)

    with h5py.File(file_path, "r") as f:
        group_names = list(groups) if groups is not None else list(f.keys())
        for name in group_names:
            grp = f[name]
            species = grp[species_key]
            n_conf = grp[coordinates_key].shape[0]
            targets = np.stack(
                [np.asarray(grp[key][()], dtype=np.float64) for key in target_keys],
                axis=-1,
            )  # [n_conf, n_atoms, n_properties]
            for i in range(n_conf):
                species_arr = species[i] if species.ndim == 2 else species[()]
                indices = atomic_numbers_to_indices(
                    species_to_atomic_numbers(species_arr), z_table=z_table
                )
                for atom_idx, element_idx in enumerate(indices):
                    count[element_idx] += 1
                    total[element_idx] += targets[i, atom_idx]
                    total_sq[element_idx] += targets[i, atom_idx] ** 2

    missing = count == 0
    if missing.any():
        missing_zs = [z_table.index_to_z(i) for i in np.nonzero(missing)[0]]
        logging.warning(
            "No atoms found for elements with atomic numbers %s; "
            "their mean/std default to 0/1.",
            missing_zs,
        )
    safe_count = np.where(missing, 1.0, count)[:, None]
    mean = total / safe_count
    variance = total_sq / safe_count - mean**2
    std = np.sqrt(np.clip(variance, a_min=0.0, a_max=None))
    std = np.where(std < std_floor, std_floor, std)
    mean = np.where(missing[:, None], 0.0, mean)
    std = np.where(missing[:, None], 1.0, std)

    return {"mean": mean, "std": std, "counts": count}
