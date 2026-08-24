import argparse
from functools import partial

import h5py
import numpy as np
import torch

from mace.cli.run_train_xdm import dataloader_worker_init_fn, split_molecule_names
from mace.data.xdm import XDMHDF5Dataset
from mace.tools import AtomicNumberTable, set_default_dtype
from mace.tools.torch_geometric.dataloader import DataLoader

SYMBOL_TO_Z = {"H": 1, "C": 6, "O": 8}


def _write_master_file(path, n_molecules=20, seed=0, prefix="mol"):
    rng = np.random.RandomState(seed)
    with h5py.File(path, "w") as f:
        wrapper = f.create_group("ani2x_pbe0xdm_0")
        for i in range(n_molecules):
            species = ["O", "H", "H"] if i % 2 == 0 else ["C", "H", "H", "H", "H"]
            grp = wrapper.create_group(f"{prefix}_{i}")
            n_atoms = len(species)
            grp.create_dataset(
                "atomic_numbers",
                data=np.array([SYMBOL_TO_Z[s] for s in species], dtype=np.uint8),
            )
            grp.create_dataset("coordinates", data=rng.randn(1, n_atoms, 3).astype(np.float32))
            for key in ["M1", "M2", "M3", "Veff"]:
                grp.create_dataset(key, data=rng.rand(1, n_atoms))
    return path


def _namespace(**overrides):
    defaults = dict(
        valid_files=None,
        test_files=None,
        valid_fraction=0.2,
        test_fraction=0.2,
        seed=0,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_split_molecule_names_disjoint_and_covers_all(tmp_path):
    path = _write_master_file(tmp_path / "master.h5", n_molecules=20)
    args = _namespace()
    names = split_molecule_names(args, [str(path)], None, None)

    train, valid, test = set(names["train"]), set(names["valid"]), set(names["test"])
    assert not (train & valid)
    assert not (train & test)
    assert not (valid & test)
    assert train | valid | test == {f"mol_{i}" for i in range(20)}
    assert len(valid) == 4  # round(20 * 0.2)
    assert len(test) == 4


def test_split_molecule_names_deterministic_with_seed(tmp_path):
    path = _write_master_file(tmp_path / "master.h5", n_molecules=20)
    args1 = _namespace(seed=42)
    args2 = _namespace(seed=42)
    names1 = split_molecule_names(args1, [str(path)], None, None)
    names2 = split_molecule_names(args2, [str(path)], None, None)
    assert names1 == names2


def test_split_molecule_names_different_seeds_differ(tmp_path):
    path = _write_master_file(tmp_path / "master.h5", n_molecules=20)
    names1 = split_molecule_names(_namespace(seed=1), [str(path)], None, None)
    names2 = split_molecule_names(_namespace(seed=2), [str(path)], None, None)
    assert names1 != names2


def test_split_molecule_names_explicit_test_files_excluded_from_train(tmp_path):
    train_path = _write_master_file(tmp_path / "train.h5", n_molecules=10, prefix="trainmol")
    test_path = _write_master_file(tmp_path / "test.h5", n_molecules=5, prefix="testmol")
    args = _namespace(test_files=[str(test_path)])
    names = split_molecule_names(args, [str(train_path)], None, [str(test_path)])

    assert set(names["test"]) == {f"testmol_{i}" for i in range(5)}
    assert not (set(names["train"]) & set(names["test"]))
    assert not (set(names["valid"]) & set(names["test"]))
    # train pool (10) split further into valid_fraction/(1-valid_fraction) since
    # test came from a separate file and wasn't part of the train-file pool
    assert set(names["train"]) | set(names["valid"]) == {f"trainmol_{i}" for i in range(10)}


def test_dataloader_worker_init_fn_fixes_node_attrs_dtype_under_spawn(tmp_path):
    # torch.set_default_dtype() is per-process global state; under the
    # "spawn" multiprocessing context, DataLoader worker subprocesses are
    # fresh interpreters that don't inherit it, so AtomicData built inside
    # XDMHDF5Dataset.__getitem__ (node_attrs, positions, etc., all built via
    # torch.get_default_dtype()) silently comes out float32 unless each
    # worker re-applies the requested default dtype itself.
    path = _write_master_file(tmp_path / "master.h5", n_molecules=4)
    z_table = AtomicNumberTable([1, 6, 8])
    dataset = XDMHDF5Dataset(str(path), z_table=z_table, r_max=5.0)

    original_dtype = torch.get_default_dtype()
    try:
        set_default_dtype("float64")

        loader_without_fix = DataLoader(
            dataset,
            batch_size=2,
            num_workers=2,
            multiprocessing_context="spawn",
        )
        batch = next(iter(loader_without_fix))
        assert batch.node_attrs.dtype == torch.float32, (
            "expected the pre-fix bug to reproduce here (worker didn't "
            "inherit float64); if this fails, the underlying multiprocessing "
            "dtype-inheritance behavior this test targets may have changed"
        )

        loader_with_fix = DataLoader(
            dataset,
            batch_size=2,
            num_workers=2,
            multiprocessing_context="spawn",
            worker_init_fn=partial(dataloader_worker_init_fn, default_dtype="float64"),
        )
        batch = next(iter(loader_with_fix))
        assert batch.node_attrs.dtype == torch.float64
        assert batch.positions.dtype == torch.float64
    finally:
        torch.set_default_dtype(original_dtype)
