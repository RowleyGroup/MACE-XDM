import h5py
import numpy as np
import torch
from e3nn import o3
from scipy.spatial.transform import Rotation as R

from mace import modules
from mace.data.xdm import (
    XDMHDF5Dataset,
    build_xdm_atomic_data,
    compute_xdm_element_statistics,
    discover_atomic_number_table,
    species_to_atomic_numbers,
)
from mace.tools import AtomicNumberTable
from mace.tools.torch_geometric.batch import Batch
from mace.tools.torch_geometric.dataloader import DataLoader

torch.set_default_dtype(torch.float64)


def _write_synthetic_dataset(path):
    rng = np.random.RandomState(0)
    with h5py.File(path, "w") as f:
        # species fixed per group, shape [n_atoms]
        grp = f.create_group("H2O")
        n_conf = 6
        grp.create_dataset("species", data=np.array([b"O", b"H", b"H"]))
        base = np.array([[0.0, 0.0, 0.0], [0.0, 0.76, 0.59], [0.0, -0.76, 0.59]])
        grp.create_dataset("coordinates", data=base[None] + rng.randn(n_conf, 3, 3) * 0.05)
        for key in ["M1", "M2", "M3", "Veff"]:
            grp.create_dataset(key, data=rng.rand(n_conf, 3) + 1.0)

        # species varies per conformer, shape [n_conf, n_atoms]
        grp2 = f.create_group("CH4")
        n_conf2 = 5
        species2 = np.tile(np.array([b"C", b"H", b"H", b"H", b"H"]), (n_conf2, 1))
        grp2.create_dataset("species", data=species2)
        grp2.create_dataset("coordinates", data=rng.randn(n_conf2, 5, 3))
        for key in ["M1", "M2", "M3", "Veff"]:
            grp2.create_dataset(key, data=rng.rand(n_conf2, 5) + 2.0)
    return path


def test_species_to_atomic_numbers():
    assert list(species_to_atomic_numbers(np.array([b"H", b"C", b"O"]))) == [1, 6, 8]
    assert list(species_to_atomic_numbers(np.array([1, 6, 8]))) == [1, 6, 8]
    nested = species_to_atomic_numbers(np.array([[b"H", b"C"], [b"O", b"H"]]))
    assert nested.tolist() == [[1, 6], [8, 1]]


def test_discover_atomic_number_table(tmp_path):
    path = _write_synthetic_dataset(tmp_path / "xdm.h5")
    z_table = discover_atomic_number_table(str(path))
    assert z_table.zs == [1, 6, 8]


def test_compute_xdm_element_statistics(tmp_path):
    path = _write_synthetic_dataset(tmp_path / "xdm.h5")
    z_table = discover_atomic_number_table(str(path))
    stats = compute_xdm_element_statistics(str(path), z_table)
    assert stats["mean"].shape == (3, 4)
    assert stats["std"].shape == (3, 4)
    # H atoms appear in both H2O (2 per conformer) and CH4 (4 per conformer)
    n_h = 2 * 6 + 4 * 5
    assert stats["counts"][z_table.z_to_index(1)] == n_h
    assert np.all(stats["std"] > 0)


def test_xdm_hdf5_dataset(tmp_path):
    path = _write_synthetic_dataset(tmp_path / "xdm.h5")
    z_table = discover_atomic_number_table(str(path))
    dataset = XDMHDF5Dataset(str(path), z_table=z_table, r_max=5.0)
    assert len(dataset) == 6 + 5
    item = dataset[0]
    assert item.xdm_targets.shape[1] == 4
    assert item.xdm_targets.shape[0] == item.node_attrs.shape[0]

    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    assert batch.xdm_targets.shape[1] == 4


def test_xdm_dataset_group_subset(tmp_path):
    path = _write_synthetic_dataset(tmp_path / "xdm.h5")
    z_table = discover_atomic_number_table(str(path))
    train_ds = XDMHDF5Dataset(str(path), z_table=z_table, r_max=5.0, groups=["H2O"])
    valid_ds = XDMHDF5Dataset(str(path), z_table=z_table, r_max=5.0, groups=["CH4"])
    assert len(train_ds) == 6
    assert len(valid_ds) == 5


def _build_model(z_table, num_xdm_targets=4):
    n_elements = len(z_table)
    rng = np.random.RandomState(0)
    element_means = rng.rand(n_elements, num_xdm_targets) + 1.0
    element_stds = rng.rand(n_elements, num_xdm_targets) * 0.1 + 0.1
    return modules.AtomicXDMMACE(
        r_max=5.0,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes["RealAgnosticInteractionBlock"],
        num_interactions=2,
        num_elements=n_elements,
        hidden_irreps=o3.Irreps("16x0e + 16x1o"),
        MLP_irreps=o3.Irreps("16x0e"),
        avg_num_neighbors=3.0,
        atomic_numbers=z_table.zs,
        correlation=2,
        gate=torch.nn.functional.silu,
        element_means=element_means,
        element_stds=element_stds,
        num_xdm_targets=num_xdm_targets,
    )


def test_atomic_xdm_mace_forward_shapes():
    z_table = AtomicNumberTable([1, 6, 8])
    model = _build_model(z_table)

    atomic_numbers = np.array([8, 1, 1])
    positions = np.array([[0.0, -2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    targets = np.zeros((3, 4))
    graph = build_xdm_atomic_data(atomic_numbers, positions, targets, z_table, cutoff=3.0)
    batch = Batch.from_data_list([graph])

    output = model(batch.to_dict())
    assert output["xdm_atomic"].shape == (3, 4)
    assert output["xdm_standardized"].shape == (3, 4)


def test_atomic_xdm_mace_rotation_invariance():
    z_table = AtomicNumberTable([1, 6, 8])
    model = _build_model(z_table)

    atomic_numbers = np.array([8, 1, 1])
    positions = np.array([[0.0, -2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    targets = np.zeros((3, 4))

    rot = R.from_euler("z", 60, degrees=True).as_matrix()
    positions_rotated = positions @ rot.T

    graph = build_xdm_atomic_data(atomic_numbers, positions, targets, z_table, cutoff=3.0)
    graph_rotated = build_xdm_atomic_data(
        atomic_numbers, positions_rotated, targets, z_table, cutoff=3.0
    )
    batch = Batch.from_data_list([graph, graph_rotated])

    output = model(batch.to_dict())
    xdm_atomic = output["xdm_atomic"]
    assert torch.allclose(xdm_atomic[0:3], xdm_atomic[3:6], atol=1e-8)


def test_atomic_xdm_mace_reference_roundtrip():
    z_table = AtomicNumberTable([1, 6, 8])
    model = _build_model(z_table)

    atomic_numbers = np.array([8, 1, 1])
    positions = np.array([[0.0, -2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    targets = np.zeros((3, 4))
    graph = build_xdm_atomic_data(atomic_numbers, positions, targets, z_table, cutoff=3.0)
    batch = Batch.from_data_list([graph])

    output = model(batch.to_dict())
    standardized_again = model.xdm_reference.standardize(
        output["xdm_atomic"], batch.node_attrs
    )
    assert torch.allclose(standardized_again, output["xdm_standardized"], atol=1e-8)


def test_atomic_xdm_mace_training_step_reduces_loss(tmp_path):
    path = _write_synthetic_dataset(tmp_path / "xdm.h5")
    z_table = discover_atomic_number_table(str(path))
    stats = compute_xdm_element_statistics(str(path), z_table)
    dataset = XDMHDF5Dataset(str(path), z_table=z_table, r_max=5.0)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = modules.AtomicXDMMACE(
        r_max=5.0,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes["RealAgnosticInteractionBlock"],
        num_interactions=2,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps("16x0e + 16x1o"),
        MLP_irreps=o3.Irreps("16x0e"),
        avg_num_neighbors=3.0,
        atomic_numbers=z_table.zs,
        correlation=2,
        gate=torch.nn.functional.silu,
        element_means=stats["mean"],
        element_stds=stats["std"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    def epoch_loss():
        total, count = 0.0, 0
        for batch in loader:
            output = model(batch.to_dict())
            target_standardized = model.xdm_reference.standardize(
                batch.xdm_targets, batch.node_attrs
            )
            loss = torch.mean((output["xdm_standardized"] - target_standardized) ** 2)
            total += loss.item() * batch.node_attrs.shape[0]
            count += batch.node_attrs.shape[0]
        return total / count

    first_loss = epoch_loss()
    for _ in range(20):
        for batch in loader:
            optimizer.zero_grad()
            output = model(batch.to_dict())
            target_standardized = model.xdm_reference.standardize(
                batch.xdm_targets, batch.node_attrs
            )
            loss = torch.mean((output["xdm_standardized"] - target_standardized) ** 2)
            loss.backward()
            optimizer.step()
    last_loss = epoch_loss()

    assert last_loss < first_loss
