import numpy as np
import torch
from e3nn import o3
from scipy.spatial.transform import Rotation as R

from mace.data import (
    build_xdm_atomic_data,
    default_mlxdm_2x_atomic_number_table,
    mlxdm_2x_polarizability_reference,
)
from mace.modules import (
    MACE,
    AtomicXDMMACE,
    MACEXDMDispersion,
    XDMDispersionEnergy,
    gate_dict,
    interaction_classes,
)
from mace.tools import AtomicNumberTable, atomic_numbers_to_indices, to_one_hot
from mace.tools.torch_geometric.batch import Batch

torch.set_default_dtype(torch.float64)

BOHR_TO_ANGSTROM = 0.529177249


def _build_two_atom_graph(z_table, atomic_numbers, positions):
    indices = atomic_numbers_to_indices(np.array(atomic_numbers), z_table=z_table)
    node_attrs = to_one_hot(
        torch.tensor(indices, dtype=torch.long).unsqueeze(-1), num_classes=len(z_table)
    )
    return node_attrs


def test_xdm_dispersion_energy_matches_hand_computation():
    z_table = AtomicNumberTable([1, 6])  # H, C
    ref = mlxdm_2x_polarizability_reference(z_table)
    module = XDMDispersionEnergy(
        alpha_free=ref["alpha_free"], v_free=ref["v_free"], cutoff=14.0, a1=0.4186, a2=2.6791
    )

    positions = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    node_attrs = _build_two_atom_graph(z_table, [1, 6], positions)
    batch = torch.zeros(2, dtype=torch.long)

    M1_A, M2_A, M3_A, Veff_A = 2.0, 10.0, 100.0, 6.0
    M1_B, M2_B, M3_B, Veff_B = 3.0, 20.0, 300.0, 25.0
    xdm_atomic = torch.tensor([[M1_A, M2_A, M3_A, Veff_A], [M1_B, M2_B, M3_B, Veff_B]])

    e_disp = module(
        positions=positions, node_attrs=node_attrs, batch=batch, num_graphs=1, xdm_atomic=xdm_atomic
    )

    alpha_free_H, v_free_H = ref["alpha_free"][0], ref["v_free"][0]
    alpha_free_C, v_free_C = ref["alpha_free"][1], ref["v_free"][1]
    alpha_A = Veff_A * alpha_free_H / v_free_H
    alpha_B = Veff_B * alpha_free_C / v_free_C

    denom = M1_A / alpha_A + M1_B / alpha_B
    C6 = M1_A * M1_B / denom
    C8 = 1.5 * (M1_A * M2_B + M1_B * M2_A) / denom
    C10 = 2 * (M1_A * M3_B + M3_A * M1_B + 2.1 * M2_A * M2_B) / denom

    r_crit = ((C8 / C6) ** 0.5 + (C10 / C6) ** 0.25 + (C10 / C8) ** 0.5) / 3.0
    a1, a2 = 0.4186, 2.6791
    r_vdw = a2 + a1 * r_crit * BOHR_TO_ANGSTROM

    r = 3.0
    e6_manual = -C6 / (r**6 + r_vdw**6) * BOHR_TO_ANGSTROM**6
    e8_manual = -C8 / (r**8 + r_vdw**8) * BOHR_TO_ANGSTROM**8
    e10_manual = -C10 / (r**10 + r_vdw**10) * BOHR_TO_ANGSTROM**10
    e_manual = e6_manual + e8_manual + e10_manual

    assert torch.allclose(e_disp, torch.tensor([e_manual]), rtol=1e-10)

    components = module(
        positions=positions,
        node_attrs=node_attrs,
        batch=batch,
        num_graphs=1,
        xdm_atomic=xdm_atomic,
        return_components=True,
    )
    assert torch.allclose(components["total"], e_disp)
    assert torch.allclose(components["e6"], torch.tensor([e6_manual]), rtol=1e-10)
    assert torch.allclose(components["e8"], torch.tensor([e8_manual]), rtol=1e-10)
    assert torch.allclose(components["e10"], torch.tensor([e10_manual]), rtol=1e-10)
    assert torch.allclose(
        components["e6"] + components["e8"] + components["e10"], components["total"]
    )


def test_xdm_dispersion_energy_invariances_and_forces():
    z_table = default_mlxdm_2x_atomic_number_table()
    ref = mlxdm_2x_polarizability_reference(z_table)
    module = XDMDispersionEnergy(alpha_free=ref["alpha_free"], v_free=ref["v_free"])

    atomic_numbers = [8, 1, 1]
    positions = torch.tensor(
        [[0.0, -2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], requires_grad=True
    )
    node_attrs = _build_two_atom_graph(z_table, atomic_numbers, positions)
    batch = torch.zeros(3, dtype=torch.long)
    torch.manual_seed(0)
    xdm_atomic = torch.rand(3, 4) * 5 + 1.0

    e = module(positions=positions, node_attrs=node_attrs, batch=batch, num_graphs=1, xdm_atomic=xdm_atomic)

    shift = torch.tensor([5.0, -3.0, 2.0])
    e_shifted = module(
        positions=positions.detach() + shift, node_attrs=node_attrs, batch=batch, num_graphs=1,
        xdm_atomic=xdm_atomic,
    )
    assert torch.allclose(e, e_shifted, atol=1e-10)

    rot = torch.tensor(R.from_euler("xyz", [30, 40, 50], degrees=True).as_matrix())
    positions_rot = positions.detach() @ rot.T
    e_rot = module(
        positions=positions_rot, node_attrs=node_attrs, batch=batch, num_graphs=1, xdm_atomic=xdm_atomic
    )
    assert torch.allclose(e, e_rot, atol=1e-10)

    e.backward()
    analytic_force = -positions.grad.clone()

    eps = 1e-6
    pos0 = positions.detach().clone()
    numeric_force = torch.zeros(3, 3, dtype=torch.float64)
    for i in range(3):
        for k in range(3):
            p_plus = pos0.clone()
            p_plus[i, k] += eps
            p_minus = pos0.clone()
            p_minus[i, k] -= eps
            e_plus = module(
                positions=p_plus, node_attrs=node_attrs, batch=batch, num_graphs=1, xdm_atomic=xdm_atomic
            )
            e_minus = module(
                positions=p_minus, node_attrs=node_attrs, batch=batch, num_graphs=1, xdm_atomic=xdm_atomic
            )
            numeric_force[i, k] = -(e_plus - e_minus) / (2 * eps)

    assert torch.allclose(analytic_force, numeric_force, atol=1e-6)


def _build_xdm_model(z_table, num_xdm_targets=4):
    n_elements = len(z_table)
    rng = np.random.RandomState(0)
    return AtomicXDMMACE(
        r_max=5.0,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=interaction_classes["RealAgnosticInteractionBlock"],
        num_interactions=2,
        num_elements=n_elements,
        hidden_irreps=o3.Irreps("16x0e + 16x1o"),
        MLP_irreps=o3.Irreps("16x0e"),
        avg_num_neighbors=5.0,
        atomic_numbers=z_table.zs,
        correlation=2,
        gate=gate_dict["silu"],
        element_means=rng.rand(n_elements, num_xdm_targets) + 1.0,
        element_stds=rng.rand(n_elements, num_xdm_targets) * 0.1 + 0.1,
        num_xdm_targets=num_xdm_targets,
    )


def _build_short_range_model(z_table, atomic_energies):
    return MACE(
        r_max=5.0,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=interaction_classes["RealAgnosticInteractionBlock"],
        num_interactions=2,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps("16x0e + 16x1o"),
        MLP_irreps=o3.Irreps("16x0e"),
        atomic_energies=np.array(atomic_energies),
        avg_num_neighbors=5.0,
        atomic_numbers=z_table.zs,
        correlation=2,
        gate=torch.nn.functional.silu,
    )


def test_mace_xdm_dispersion_combined_forward_and_forces():
    xdm_z_table = default_mlxdm_2x_atomic_number_table()
    xdm_model = _build_xdm_model(xdm_z_table)

    # Deliberately different element ordering/subset than the XDM model.
    sr_z_table = AtomicNumberTable([8, 1, 6])
    sr_model = _build_short_range_model(sr_z_table, atomic_energies=[-75.0, -0.5, -37.8])

    disp_ref = mlxdm_2x_polarizability_reference(xdm_z_table)
    disp_module = XDMDispersionEnergy(alpha_free=disp_ref["alpha_free"], v_free=disp_ref["v_free"])

    combined = MACEXDMDispersion(
        short_range_model=sr_model, xdm_model=xdm_model, dispersion_energy=disp_module
    )

    atomic_numbers = np.array([8, 1, 1])
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.76, 0.59], [0.0, -0.76, 0.59]])
    targets = np.zeros((3, 4))
    graph = build_xdm_atomic_data(atomic_numbers, positions, targets, xdm_z_table, cutoff=5.0)
    batch = Batch.from_data_list([graph])
    data = batch.to_dict()

    out = combined(data, training=False, compute_force=True)
    assert torch.allclose(
        out["energy"], out["short_range_energy"] + out["dispersion_energy"], atol=1e-12
    )
    assert out["forces"].shape == (3, 3)

    eps = 1e-6
    pos0 = data["positions"].detach().clone()
    numeric_force = torch.zeros(3, 3, dtype=torch.float64)
    for i in range(3):
        for k in range(3):
            p_plus = pos0.clone()
            p_plus[i, k] += eps
            p_minus = pos0.clone()
            p_minus[i, k] -= eps
            data_plus = dict(data)
            data_plus["positions"] = p_plus.clone().requires_grad_(True)
            data_minus = dict(data)
            data_minus["positions"] = p_minus.clone().requires_grad_(True)
            e_plus = combined(data_plus, training=False, compute_force=False)["energy"]
            e_minus = combined(data_minus, training=False, compute_force=False)["energy"]
            numeric_force[i, k] = -(e_plus - e_minus).item() / (2 * eps)

    assert torch.allclose(out["forces"], numeric_force, atol=1e-6)


def test_mace_xdm_dispersion_unsupported_element_raises():
    xdm_z_table = default_mlxdm_2x_atomic_number_table()
    xdm_model = _build_xdm_model(xdm_z_table)

    sr_z_table = AtomicNumberTable([6])
    sr_model = _build_short_range_model(sr_z_table, atomic_energies=[-37.8])

    disp_ref = mlxdm_2x_polarizability_reference(xdm_z_table)
    disp_module = XDMDispersionEnergy(alpha_free=disp_ref["alpha_free"], v_free=disp_ref["v_free"])

    combined = MACEXDMDispersion(
        short_range_model=sr_model, xdm_model=xdm_model, dispersion_energy=disp_module
    )

    atomic_numbers = np.array([8, 1, 1])
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.76, 0.59], [0.0, -0.76, 0.59]])
    targets = np.zeros((3, 4))
    graph = build_xdm_atomic_data(atomic_numbers, positions, targets, xdm_z_table, cutoff=5.0)
    data = Batch.from_data_list([graph]).to_dict()

    try:
        combined(data, training=False, compute_force=False)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "1" in str(exc) or "8" in str(exc)
