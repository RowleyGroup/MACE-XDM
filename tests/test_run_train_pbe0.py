import numpy as np
import h5py
import torch

from mace.cli.run_train_pbe0 import main
from mace.cli.run_train_xdm import warm_start_from_foundation_model
from mace.data.xdm import (
    ANIHDF5EnergyForcesDataset,
    HARTREE_TO_EV,
    estimate_atomic_energies_linear_regression,
)
from mace.modules import AtomicXDMMACE, gate_dict, interaction_classes
from mace.tools import AtomicNumberTable

SYMBOL_TO_Z = {"H": 1, "C": 6, "O": 8}


def _write_mixed_file(path, n_labeled=20, n_dft_only=8, seed=0):
    """A file mixing XDM-labeled molecules (M1/M2/M3/Veff present) with
    DFT-only molecules (no XDM labels) under one wrapper group, matching the
    real ANI-PBE0 dataset schema this script reads."""
    rng = np.random.RandomState(seed)
    with h5py.File(path, "w") as f:
        wrapper = f.create_group("batch0")
        for i in range(n_labeled):
            species = ["C", "H", "H", "H", "O"]
            grp = wrapper.create_group(f"labeled_{i}")
            n_atoms = len(species)
            n_conf = 3
            grp.create_dataset(
                "atomic_numbers",
                data=np.array([SYMBOL_TO_Z[s] for s in species], dtype=np.uint8),
            )
            grp.create_dataset(
                "coordinates", data=rng.randn(n_conf, n_atoms, 3).astype(np.float32)
            )
            grp.create_dataset(
                "energies", data=(-100.0 + rng.randn(n_conf) * 0.01).astype(np.float64)
            )
            grp.create_dataset(
                "forces", data=(rng.randn(n_conf, n_atoms, 3) * 0.01).astype(np.float32)
            )
            for key in ["M1", "M2", "M3", "Veff"]:
                grp.create_dataset(key, data=rng.rand(n_conf, n_atoms))
        for i in range(n_dft_only):
            species = ["C", "H", "H", "H", "H"]
            grp = wrapper.create_group(f"dftonly_{i}")
            n_atoms = len(species)
            n_conf = 3
            grp.create_dataset(
                "atomic_numbers",
                data=np.array([SYMBOL_TO_Z[s] for s in species], dtype=np.uint8),
            )
            grp.create_dataset(
                "coordinates", data=rng.randn(n_conf, n_atoms, 3).astype(np.float32)
            )
            grp.create_dataset(
                "energies", data=(-80.0 + rng.randn(n_conf) * 0.01).astype(np.float64)
            )
            grp.create_dataset(
                "forces", data=(rng.randn(n_conf, n_atoms, 3) * 0.01).astype(np.float32)
            )
            grp.create_dataset("scf", data=rng.randn(n_conf))
    return path


def test_ani_hdf5_energy_forces_dataset_reads_both_labeled_and_dft_only(tmp_path):
    path = _write_mixed_file(tmp_path / "mixed.h5", n_labeled=5, n_dft_only=3)
    z_table = AtomicNumberTable([1, 6, 8])
    dataset = ANIHDF5EnergyForcesDataset(str(path), z_table=z_table, r_max=5.0)
    # 8 molecules total (labeled + dft-only) x 3 conformers each -- this
    # dataset only needs energies/forces, present in every group, so both
    # kinds of molecule are included.
    assert len(dataset) == 8 * 3
    item = dataset[0]
    assert item.energy is not None
    assert item.forces is not None


def test_ani_hdf5_energy_forces_dataset_skips_groups_missing_energy(tmp_path):
    path = tmp_path / "partial.h5"
    with h5py.File(path, "w") as f:
        g1 = f.create_group("has_energy")
        g1.create_dataset("atomic_numbers", data=np.array([1, 1], dtype=np.uint8))
        g1.create_dataset("coordinates", data=np.random.randn(2, 2, 3).astype(np.float32))
        g1.create_dataset("energies", data=np.random.randn(2))
        g1.create_dataset("forces", data=np.random.randn(2, 2, 3).astype(np.float32))

        g2 = f.create_group("no_energy")
        g2.create_dataset("atomic_numbers", data=np.array([1, 1], dtype=np.uint8))
        g2.create_dataset("coordinates", data=np.random.randn(2, 2, 3).astype(np.float32))
        g2.create_dataset("some_other_property", data=np.random.randn(2))

    z_table = AtomicNumberTable([1])
    dataset = ANIHDF5EnergyForcesDataset(str(path), z_table=z_table, r_max=5.0)
    assert len(dataset) == 2  # only has_energy's 2 conformers


def test_estimate_atomic_energies_linear_regression_recovers_known_e0s(tmp_path):
    # Build a file where every conformer's energy is EXACTLY sum(E0_z * count_z)
    # (in Hartree, since energy_forces_unit defaults to "hartree" and this
    # test checks the eV-converted output), so the least-squares solve should
    # recover the known E0s (up to floating point) with zero residual.
    true_e0_ev = {1: -13.6, 6: -1029.4, 8: -2041.8}
    path = tmp_path / "e0.h5"
    with h5py.File(path, "w") as f:
        wrapper = f.create_group("batch0")
        species_options = [["H", "H", "O"], ["C", "H", "H", "H", "H"], ["C", "O", "H", "H"]]
        for i, species in enumerate(species_options):
            grp = wrapper.create_group(f"mol_{i}")
            n_atoms = len(species)
            zs = [SYMBOL_TO_Z[s] for s in species]
            energy_ev = sum(true_e0_ev[z] for z in zs)
            energy_hartree = energy_ev / HARTREE_TO_EV
            grp.create_dataset("atomic_numbers", data=np.array(zs, dtype=np.uint8))
            grp.create_dataset("coordinates", data=np.random.randn(1, n_atoms, 3).astype(np.float32))
            grp.create_dataset("energies", data=np.array([energy_hartree]))
            grp.create_dataset("forces", data=np.zeros((1, n_atoms, 3), dtype=np.float32))

    z_table = AtomicNumberTable([1, 6, 8])
    e0 = estimate_atomic_energies_linear_regression(str(path), z_table=z_table)
    for i, z in enumerate(z_table.zs):
        assert e0[i] == pytest_approx(true_e0_ev[z])


def pytest_approx(x, tol=1e-6):
    import pytest

    return pytest.approx(x, abs=tol)


def test_run_train_pbe0_end_to_end_and_foundation_model_compatible(tmp_path):
    path = _write_mixed_file(tmp_path / "mixed.h5", n_labeled=30, n_dft_only=10)

    args = [
        "--train_files", str(path),
        "--valid_fraction", "0.2",
        "--test_fraction", "0.2",
        "--r_max", "5.0",
        "--num_bessel", "4",
        "--num_polynomial_cutoff", "5",
        "--max_ell", "2",
        "--num_interactions", "1",
        "--hidden_irreps", "8x0e",
        "--MLP_irreps", "8x0e",
        "--correlation", "2",
        "--batch_size", "4",
        "--valid_batch_size", "4",
        "--max_num_epochs", "2",
        "--patience", "5",
        "--device", "cpu",
        "--default_dtype", "float64",
        "--num_workers", "0",
        "--seed", "0",
        "--name", "test_pbe0",
        "--log_dir", str(tmp_path / "logs"),
        "--checkpoints_dir", str(tmp_path / "checkpoints"),
        "--model_dir", str(tmp_path / "models"),
        "--results_dir", str(tmp_path / "results"),
    ]

    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["run_train_pbe0.py"] + args
        main()
    finally:
        sys.argv = old_argv

    model_path = tmp_path / "models" / "test_pbe0.model"
    assert model_path.exists()
    assert (tmp_path / "results" / "test_pbe0_DONE").exists()

    pbe0_model = torch.load(model_path, weights_only=False)

    from e3nn import o3

    xdm = AtomicXDMMACE(
        r_max=5.0,
        num_bessel=4,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=interaction_classes["RealAgnosticResidualInteractionBlock"],
        num_interactions=1,
        num_elements=3,
        hidden_irreps=o3.Irreps("8x0e"),
        MLP_irreps=o3.Irreps("8x0e"),
        avg_num_neighbors=float(pbe0_model.interactions[0].avg_num_neighbors),
        atomic_numbers=pbe0_model.atomic_numbers.tolist(),
        correlation=2,
        gate=gate_dict["silu"],
        element_means=torch.zeros(3, 4),
        element_stds=torch.ones(3, 4),
        num_xdm_targets=4,
    )

    warm_start_from_foundation_model(xdm, str(model_path), torch.device("cpu"))
    xdm_state = xdm.state_dict()
    pbe0_state = pbe0_model.state_dict()
    # At least the first interaction block's weights should have transplanted
    # exactly (same name, same tensor values).
    name = "interactions.0.linear_up.weight"
    assert torch.equal(xdm_state[name], pbe0_state[name])
