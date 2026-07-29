import argparse

import h5py
import numpy as np

from mace.cli.run_train_xdm import split_molecule_names

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
