###########################################################################################
# Data handling for training MACE to predict per-atom XDM dispersion coefficients
# (M1, M2, M3, Veff) from an ANI-style HDF5 dataset spread across one or more files.
###########################################################################################

import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from ase.data import atomic_numbers as ase_atomic_numbers

from mace.tools import AtomicNumberTable, atomic_numbers_to_indices, to_one_hot

from .atomic_data import AtomicData
from .neighborhood import get_neighborhood

DEFAULT_TARGET_KEYS: Tuple[str, str, str, str] = ("M1", "M2", "M3", "Veff")

FilePaths = Union[str, Sequence[str]]


def _as_file_list(file_paths: FilePaths) -> List[str]:
    return [file_paths] if isinstance(file_paths, str) else list(file_paths)


def find_leaf_groups(node: h5py.Group, path: str = "") -> List[str]:
    """Recursively find the HDF5 paths of "leaf" groups (whose direct children
    are datasets rather than further sub-groups), at any nesting depth.

    Molecule groups in these ANI-style files are not always at the top level of
    the file (there may be an extra wrapper group per file, e.g. a batch/run
    name); this walks down through any such wrapper groups to find the actual
    per-molecule groups.
    """
    children = list(node.items())
    has_subgroup = any(isinstance(v, h5py.Group) for _, v in children)
    has_dataset = any(isinstance(v, h5py.Dataset) for _, v in children)

    leaves = []
    if has_dataset and not has_subgroup:
        leaves.append(path)
    if has_subgroup:
        for name, child in children:
            if isinstance(child, h5py.Group):
                child_path = f"{path}/{name}" if path else name
                leaves.extend(find_leaf_groups(child, child_path))
    return leaves


def _filter_leaf_groups_with_keys(
    f: h5py.File, leaf_paths: List[str], required_keys: Sequence[str]
) -> List[str]:
    """Keep only leaf groups that have every dataset in ``required_keys``.

    Mixed-provenance files commonly pool XDM-labeled molecules (M1/M2/M3/Veff
    present) together with DFT-only data -- e.g. active-learning batches that
    have energies/forces but no XDM labels yet -- under the same top-level
    file. Without this filter, the first such leaf group encountered crashes
    with a KeyError deep inside whichever function tries to read its
    (nonexistent) XDM target dataset, rather than being skipped.
    """
    kept = [p for p in leaf_paths if all(k in f[p] for k in required_keys)]
    n_skipped = len(leaf_paths) - len(kept)
    if n_skipped:
        logging.warning(
            "Skipping %d/%d leaf groups missing %s (likely DFT-only data "
            "mixed into this file, e.g. active-learning batches with no XDM "
            "labels yet).",
            n_skipped,
            len(leaf_paths),
            list(required_keys),
        )
    return kept


def molecule_name(leaf_path: str) -> str:
    """The trailing path component of a leaf group, used as the molecule's
    identity for train/valid splitting and for merging conformers of the same
    molecule that are stored in different files."""
    return leaf_path.rsplit("/", 1)[-1]


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
    """Reads one or more ANI-style HDF5 files of molecules labeled with atomic
    XDM coefficients.

    Each file may nest its per-molecule groups under an arbitrary number of
    wrapper groups (e.g. a batch/run name); the actual leaf groups (the ones
    holding datasets) are discovered automatically at any depth. Within a leaf
    group::

        .../<molecule>/<species_key>       [n_atoms] or [n_conf, n_atoms]
        .../<molecule>/<coordinates_key>   [n_conf, n_atoms, 3]  (Angstrom)
        .../<molecule>/<M1, M2, M3, Veff>  each [n_conf, n_atoms]

    ``species_key`` defaults to ``atomic_numbers`` (an integer array); a
    ``species`` array of element symbols (bytes or str) is also supported by
    passing ``species_key="species"``. Either may be a fixed per-molecule
    array (shape ``[n_atoms]``) or vary per conformer (shape ``[n_conf, n_atoms]``).

    If the same molecule name appears in more than one file (e.g. successive
    active-learning batches each contributing a new conformer), all of their
    conformers are pooled under that molecule for this dataset. Pass
    ``molecule_names`` to restrict to a specific subset (e.g. for a train/valid
    split by molecule identity).
    """

    def __init__(
        self,
        file_paths: FilePaths,
        z_table: AtomicNumberTable,
        r_max: float,
        target_keys: Sequence[str] = DEFAULT_TARGET_KEYS,
        species_key: str = "atomic_numbers",
        coordinates_key: str = "coordinates",
        molecule_names: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.file_paths = _as_file_list(file_paths)
        self._files: Dict[int, h5py.File] = {}
        self.z_table = z_table
        self.r_max = r_max
        self.target_keys = list(target_keys)
        self.species_key = species_key
        self.coordinates_key = coordinates_key
        molecule_name_filter = (
            set(molecule_names) if molecule_names is not None else None
        )

        # index entries: (file_idx, leaf_path, conformer_idx)
        self.index: List[Tuple[int, str, int]] = []
        for file_idx, path in enumerate(self.file_paths):
            with h5py.File(path, "r") as f:
                leaf_paths = _filter_leaf_groups_with_keys(
                    f, find_leaf_groups(f), self.target_keys
                )
                for leaf_path in leaf_paths:
                    if (
                        molecule_name_filter is not None
                        and molecule_name(leaf_path) not in molecule_name_filter
                    ):
                        continue
                    n_conf = f[leaf_path][coordinates_key].shape[0]
                    self.index.extend(
                        (file_idx, leaf_path, i) for i in range(n_conf)
                    )

    def _file(self, file_idx: int) -> h5py.File:
        if file_idx not in self._files:
            self._files[file_idx] = h5py.File(self.file_paths[file_idx], "r")
        return self._files[file_idx]

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_files"] = {}
        return state

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> AtomicData:
        file_idx, leaf_path, conf_idx = self.index[idx]
        grp = self._file(file_idx)[leaf_path]

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


def discover_molecule_names(
    file_paths: FilePaths,
    target_keys: Sequence[str] = DEFAULT_TARGET_KEYS,
) -> List[str]:
    """Union of molecule (leaf-group) names across one or more HDF5 files,
    restricted to leaf groups that actually carry every key in
    ``target_keys`` (see ``_filter_leaf_groups_with_keys``) -- this is what
    train/valid/test splitting draws from, so a molecule with no XDM labels
    should never be assigned to a split in the first place."""
    names = set()
    for path in _as_file_list(file_paths):
        with h5py.File(path, "r") as f:
            leaf_paths = _filter_leaf_groups_with_keys(
                f, find_leaf_groups(f), target_keys
            )
            names.update(molecule_name(p) for p in leaf_paths)
    return sorted(names)


def discover_atomic_number_table(
    file_paths: FilePaths,
    species_key: str = "atomic_numbers",
    molecule_names: Optional[Sequence[str]] = None,
) -> AtomicNumberTable:
    """Scan one or more XDM HDF5 files for the set of chemical elements present."""
    molecule_name_filter = set(molecule_names) if molecule_names is not None else None
    zs = set()
    for path in _as_file_list(file_paths):
        with h5py.File(path, "r") as f:
            for leaf_path in find_leaf_groups(f):
                if (
                    molecule_name_filter is not None
                    and molecule_name(leaf_path) not in molecule_name_filter
                ):
                    continue
                atomic_numbers = species_to_atomic_numbers(f[leaf_path][species_key][()])
                zs.update(int(z) for z in np.asarray(atomic_numbers).reshape(-1))
    return AtomicNumberTable(sorted(zs))


def compute_xdm_element_statistics(
    file_paths: FilePaths,
    z_table: AtomicNumberTable,
    target_keys: Sequence[str] = DEFAULT_TARGET_KEYS,
    species_key: str = "atomic_numbers",
    coordinates_key: str = "coordinates",
    molecule_names: Optional[Sequence[str]] = None,
    std_floor: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """Compute per-element mean/std of each XDM target property.

    Streams through every conformer of every molecule group once (across all
    given files), accumulating sums and sums-of-squares per element per
    property, then returns the resulting ``mean``/``std`` arrays of shape
    ``[len(z_table), len(target_keys)]``, ready to feed into
    ``AtomicElementReferenceBlock``/``AtomicXDMMACE``.
    """
    molecule_name_filter = set(molecule_names) if molecule_names is not None else None
    n_elements = len(z_table)
    n_properties = len(target_keys)
    count = np.zeros(n_elements, dtype=np.float64)
    total = np.zeros((n_elements, n_properties), dtype=np.float64)
    total_sq = np.zeros((n_elements, n_properties), dtype=np.float64)

    for path in _as_file_list(file_paths):
        with h5py.File(path, "r") as f:
            leaf_paths = _filter_leaf_groups_with_keys(
                f, find_leaf_groups(f), target_keys
            )
            for leaf_path in leaf_paths:
                if (
                    molecule_name_filter is not None
                    and molecule_name(leaf_path) not in molecule_name_filter
                ):
                    continue
                grp = f[leaf_path]
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
