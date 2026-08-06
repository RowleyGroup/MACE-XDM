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
    discover_molecule_names,
    find_leaf_groups,
    species_to_atomic_numbers,
)
from mace.tools import AtomicNumberTable
from mace.tools.torch_geometric.batch import Batch
from mace.tools.torch_geometric.dataloader import DataLoader

torch.set_default_dtype(torch.float64)

SYMBOL_TO_Z = {"H": 1, "C": 6, "N": 7, "O": 8}


def _write_wrapped_dataset(path, wrapper_name="ani2x_pbe0xdm_0", n_conf_h2o=6, n_conf_ch4=5):
    """Mirrors the real layout: file -> wrapper group -> per-molecule leaf
    groups, with 'atomic_numbers' given directly and a redundant 'species'
    array of symbols with an odd extra trailing dimension, like the real data.
    """
    rng = np.random.RandomState(0)
    with h5py.File(path, "w") as f:
        wrapper = f.create_group(wrapper_name)

        grp = wrapper.create_group("ani2x_H2O")
        base = np.array([[0.0, 0.0, 0.0], [0.0, 0.76, 0.59], [0.0, -0.76, 0.59]])
        species = ["O", "H", "H"]
        grp.create_dataset(
            "atomic_numbers", data=np.array([SYMBOL_TO_Z[s] for s in species], dtype=np.uint8)
        )
        grp.create_dataset("species", data=np.array([[s.encode()] for s in species]))
        grp.create_dataset(
            "coordinates",
            data=(base[None] + rng.randn(n_conf_h2o, 3, 3) * 0.05).astype(np.float32),
        )
        for key in ["M1", "M2", "M3", "Veff"]:
            grp.create_dataset(key, data=rng.rand(n_conf_h2o, 3) + 1.0)

        grp2 = wrapper.create_group("ani2x_CH4")
        species2 = ["C", "H", "H", "H", "H"]
        grp2.create_dataset(
            "atomic_numbers", data=np.array([SYMBOL_TO_Z[s] for s in species2], dtype=np.uint8)
        )
        grp2.create_dataset("species", data=np.array([[s.encode()] for s in species2]))
        grp2.create_dataset(
            "coordinates", data=rng.randn(n_conf_ch4, 5, 3).astype(np.float32)
        )
        for key in ["M1", "M2", "M3", "Veff"]:
            grp2.create_dataset(key, data=rng.rand(n_conf_ch4, 5) + 2.0)
    return path


def test_find_leaf_groups(tmp_path):
    path = _write_wrapped_dataset(tmp_path / "xdm.h5")
    with h5py.File(path, "r") as f:
        leaves = find_leaf_groups(f)
    assert sorted(leaves) == ["ani2x_pbe0xdm_0/ani2x_CH4", "ani2x_pbe0xdm_0/ani2x_H2O"]


def test_species_to_atomic_numbers():
    assert list(species_to_atomic_numbers(np.array([b"H", b"C", b"O"]))) == [1, 6, 8]
    assert list(species_to_atomic_numbers(np.array([1, 6, 8], dtype=np.uint8))) == [1, 6, 8]
    nested = species_to_atomic_numbers(np.array([[b"H", b"C"], [b"O", b"H"]]))
    assert nested.tolist() == [[1, 6], [8, 1]]


def test_discover_atomic_number_table(tmp_path):
    path = _write_wrapped_dataset(tmp_path / "xdm.h5")
    z_table = discover_atomic_number_table(str(path))
    assert z_table.zs == [1, 6, 8]


def test_discover_atomic_number_table_species_key(tmp_path):
    path = _write_wrapped_dataset(tmp_path / "xdm.h5")
    # Symbol-based "species" key should give the same result as the
    # integer "atomic_numbers" key (the fixture's two arrays agree).
    z_table = discover_atomic_number_table(str(path), species_key="species")
    assert z_table.zs == [1, 6, 8]


def test_discover_molecule_names_multi_file(tmp_path):
    path1 = _write_wrapped_dataset(tmp_path / "a.h5", wrapper_name="ani2x_pbe0xdm_0")
    path2 = _write_wrapped_dataset(tmp_path / "b.h5", wrapper_name="ani2x_pbe0xdm_1")
    names = discover_molecule_names([str(path1), str(path2)])
    assert names == ["ani2x_CH4", "ani2x_H2O"]


def test_compute_xdm_element_statistics(tmp_path):
    path = _write_wrapped_dataset(tmp_path / "xdm.h5")
    z_table = discover_atomic_number_table(str(path))
    stats = compute_xdm_element_statistics(str(path), z_table)
    assert stats["mean"].shape == (3, 4)
    assert stats["std"].shape == (3, 4)
    n_h = 2 * 6 + 4 * 5
    assert stats["counts"][z_table.z_to_index(1)] == n_h
    assert np.all(stats["std"] > 0)


def test_xdm_hdf5_dataset_single_file(tmp_path):
    path = _write_wrapped_dataset(tmp_path / "xdm.h5")
    z_table = discover_atomic_number_table(str(path))
    dataset = XDMHDF5Dataset(str(path), z_table=z_table, r_max=5.0)
    assert len(dataset) == 6 + 5
    item = dataset[0]
    assert item.xdm_targets.shape[1] == 4
    assert item.xdm_targets.shape[0] == item.node_attrs.shape[0]

    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    assert batch.xdm_targets.shape[1] == 4


def test_xdm_hdf5_dataset_molecule_name_subset(tmp_path):
    path = _write_wrapped_dataset(tmp_path / "xdm.h5")
    z_table = discover_atomic_number_table(str(path))
    train_ds = XDMHDF5Dataset(
        str(path), z_table=z_table, r_max=5.0, molecule_names=["ani2x_H2O"]
    )
    valid_ds = XDMHDF5Dataset(
        str(path), z_table=z_table, r_max=5.0, molecule_names=["ani2x_CH4"]
    )
    assert len(train_ds) == 6
    assert len(valid_ds) == 5


def test_xdm_hdf5_dataset_pools_conformers_across_files(tmp_path):
    path1 = _write_wrapped_dataset(
        tmp_path / "a.h5", wrapper_name="ani2x_pbe0xdm_0", n_conf_h2o=1, n_conf_ch4=1
    )
    path2 = _write_wrapped_dataset(
        tmp_path / "b.h5", wrapper_name="ani2x_pbe0xdm_1", n_conf_h2o=1, n_conf_ch4=1
    )
    z_table = discover_atomic_number_table([str(path1), str(path2)])
    dataset = XDMHDF5Dataset([str(path1), str(path2)], z_table=z_table, r_max=5.0)
    # each file contributes 1 conformer per molecule; 2 molecules x 2 files
    assert len(dataset) == 4

    train_ds = XDMHDF5Dataset(
        [str(path1), str(path2)],
        z_table=z_table,
        r_max=5.0,
        molecule_names=["ani2x_H2O"],
    )
    # both files' H2O conformers should be pooled together
    assert len(train_ds) == 2


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
    path = _write_wrapped_dataset(tmp_path / "xdm.h5")
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


def test_atomic_xdm_mace_final_readout_is_per_element():
    z_table = AtomicNumberTable([1, 6, 8])  # H, C, O
    model = _build_model(z_table)

    assert isinstance(
        model.final_readout,
        (modules.PerElementLinearReadoutBlock, modules.PerElementNonLinearReadoutBlock),
    )

    # A water molecule (O, H, H) so all three elements have atoms.
    atomic_numbers = np.array([8, 1, 1])
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.76, 0.59], [0.0, -0.76, 0.59]])
    targets = np.zeros((3, 4))
    graph = build_xdm_atomic_data(atomic_numbers, positions, targets, z_table, cutoff=5.0)
    batch = Batch.from_data_list([graph])
    data = batch.to_dict()

    with torch.no_grad():
        before = model(data)["xdm_standardized"].clone()

    # Perturb only H's (index 0) final-readout weights.
    with torch.no_grad():
        model.final_readout.linear_2.weight[0] += 1.0
        model.final_readout.linear_2.bias[0] += 1.0

    with torch.no_grad():
        after = model(data)["xdm_standardized"]

    is_h = np.array([False, True, True])  # O, H, H
    # Perturbing H's weights must change H atoms' predictions...
    assert not torch.allclose(before[is_h], after[is_h])
    # ...and must leave every other element's predictions untouched, since
    # they no longer share any final-layer weights with H.
    assert torch.allclose(before[~is_h], after[~is_h])


def test_atomic_xdm_mace_final_readout_gradients_are_element_isolated():
    z_table = AtomicNumberTable([1, 6, 8])  # H, C, O
    model = _build_model(z_table)

    atomic_numbers = np.array([8, 1, 1])
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.76, 0.59], [0.0, -0.76, 0.59]])
    targets = np.zeros((3, 4))
    graph = build_xdm_atomic_data(atomic_numbers, positions, targets, z_table, cutoff=5.0)
    batch = Batch.from_data_list([graph])

    output = model(batch.to_dict())
    is_h = torch.tensor([False, True, True])
    # A loss computed only from H atoms should never produce a gradient on
    # O's (index 2) dedicated final-readout weights.
    loss = output["xdm_standardized"][is_h].pow(2).sum()
    loss.backward()

    assert model.final_readout.linear_2.weight.grad[0].abs().sum() > 0  # H: touched
    assert model.final_readout.linear_2.weight.grad[2].abs().sum() == 0  # O: untouched
