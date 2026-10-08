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


def test_xdm_dispersion_sparse_path_matches_dense_path():
    """Above _DENSE_MAX_NODES, XDMDispersionEnergy switches from a dense
    [n,n] distance matrix to a cell-list neighbor search (mace.data.neighborhood)
    to avoid O(n_nodes^2) memory on large single-structure MD systems. The two
    pair-finding strategies must agree on both energy and forces."""
    z_table = default_mlxdm_2x_atomic_number_table()
    ref = mlxdm_2x_polarizability_reference(z_table)
    module = XDMDispersionEnergy(alpha_free=ref["alpha_free"], v_free=ref["v_free"], cutoff=14.0)

    torch.manual_seed(0)
    n_atoms = 50  # well above a lowered threshold, small enough to run fast
    atomic_numbers = np.random.RandomState(0).choice(z_table.zs, size=n_atoms)
    positions = torch.randn(n_atoms, 3, dtype=torch.float64) * 6.0
    positions.requires_grad_(True)
    node_attrs = _build_two_atom_graph(z_table, atomic_numbers, positions)
    batch = torch.zeros(n_atoms, dtype=torch.long)
    xdm_atomic = torch.rand(n_atoms, 4, dtype=torch.float64) * 5 + 1.0

    idx_i_dense, idx_j_dense, r_dense = module._dense_pairs(positions, batch)
    idx_i_sparse, idx_j_sparse, r_sparse = module._sparse_pairs(positions, batch, num_graphs=1)

    def sort_pairs(idx_i, idx_j, r):
        key = idx_i * (n_atoms + 1) + idx_j
        order = torch.argsort(key)
        return idx_i[order], idx_j[order], r[order]

    di, dj, dr = sort_pairs(idx_i_dense, idx_j_dense, r_dense)
    si, sj, sr = sort_pairs(idx_i_sparse, idx_j_sparse, r_sparse)
    assert torch.equal(di, si)
    assert torch.equal(dj, sj)
    assert torch.allclose(dr, sr, atol=1e-10)

    e_dense = module(positions=positions, node_attrs=node_attrs, batch=batch, num_graphs=1, xdm_atomic=xdm_atomic)
    e_dense.backward()
    f_dense = -positions.grad.clone()
    positions.grad = None

    module.dense_max_nodes = 1  # force the forward() dispatch onto the sparse path
    try:
        e_sparse = module(positions=positions, node_attrs=node_attrs, batch=batch, num_graphs=1, xdm_atomic=xdm_atomic)
        e_sparse.backward()
        f_sparse = -positions.grad.clone()
    finally:
        module.dense_max_nodes = XDMDispersionEnergy._DENSE_MAX_NODES

    assert torch.allclose(e_dense, e_sparse, rtol=1e-10)
    assert torch.allclose(f_dense, f_sparse, atol=1e-8)


def _random_structure(z_table, n_atoms, seed=0):
    rng = np.random.RandomState(seed)
    atomic_numbers = rng.choice(z_table.zs, size=n_atoms)
    positions = torch.tensor(rng.randn(n_atoms, 3) * 6.0)
    node_attrs = _build_two_atom_graph(z_table, atomic_numbers, positions)
    batch = torch.zeros(n_atoms, dtype=torch.long)
    xdm_atomic = torch.tensor(rng.rand(n_atoms, 4) * 5 + 1.0)
    return positions, node_attrs, batch, xdm_atomic


def test_xdm_dispersion_position_cache_matches_uncached():
    """use_position_cache=True must give numerically identical energies to the
    uncached path both on the rebuild call and after a small move that stays
    inside the cache skin (reused candidate set, distances still recomputed
    fresh) -- the cache changes which pairs are considered, never the maths."""
    z_table = default_mlxdm_2x_atomic_number_table()
    ref = mlxdm_2x_polarizability_reference(z_table)
    n_atoms = 40
    positions, node_attrs, batch, xdm_atomic = _random_structure(z_table, n_atoms)

    uncached = XDMDispersionEnergy(
        alpha_free=ref["alpha_free"], v_free=ref["v_free"], cutoff=14.0, dense_max_nodes=1
    )
    cached = XDMDispersionEnergy(
        alpha_free=ref["alpha_free"],
        v_free=ref["v_free"],
        cutoff=14.0,
        dense_max_nodes=1,
        use_position_cache=True,
        cache_skin=2.0,
    )

    kwargs = dict(node_attrs=node_attrs, batch=batch, num_graphs=1, xdm_atomic=xdm_atomic)
    e_uncached_0 = uncached(positions=positions, **kwargs)
    e_cached_0 = cached(positions=positions, **kwargs)
    assert torch.allclose(e_uncached_0, e_cached_0, rtol=1e-12)
    assert cached._cache_ref_positions is not None  # rebuilt on first call

    # Small move, well inside the skin: cache should reuse its candidate set
    # (no rebuild) but must still return the exact same energy as a fresh search.
    rng = np.random.RandomState(1)
    small_move = torch.tensor(rng.randn(n_atoms, 3) * 0.05)
    moved = positions + small_move
    ref_before = cached._cache_ref_positions.clone()
    e_uncached_1 = uncached(positions=moved, **kwargs)
    e_cached_1 = cached(positions=moved, **kwargs)
    assert torch.equal(cached._cache_ref_positions, ref_before)  # confirms no rebuild happened
    assert torch.allclose(e_uncached_1, e_cached_1, rtol=1e-10)

    # Large move, beyond the skin: cache must rebuild and still agree.
    large_move = torch.tensor(rng.randn(n_atoms, 3) * 5.0)
    moved2 = positions + large_move
    e_uncached_2 = uncached(positions=moved2, **kwargs)
    e_cached_2 = cached(positions=moved2, **kwargs)
    assert not torch.equal(cached._cache_ref_positions, ref_before)  # confirms it did rebuild
    assert torch.allclose(e_uncached_2, e_cached_2, rtol=1e-10)


def test_xdm_dispersion_position_cache_ignored_for_batched_graphs():
    """The cache is only safe for a single structure evaluated repeatedly
    (an MD/relaxation loop); for a batch of independent molecules (num_graphs
    > 1, as in training/eval) it must be silently bypassed rather than
    caching stale pairs against unrelated structures."""
    z_table = default_mlxdm_2x_atomic_number_table()
    ref = mlxdm_2x_polarizability_reference(z_table)
    n_per_graph = 20
    p1, na1, _, x1 = _random_structure(z_table, n_per_graph, seed=0)
    p2, na2, _, x2 = _random_structure(z_table, n_per_graph, seed=1)
    positions = torch.cat([p1, p2])
    node_attrs = torch.cat([na1, na2])
    batch = torch.cat([torch.zeros(n_per_graph, dtype=torch.long), torch.ones(n_per_graph, dtype=torch.long)])
    xdm_atomic = torch.cat([x1, x2])

    cached = XDMDispersionEnergy(
        alpha_free=ref["alpha_free"],
        v_free=ref["v_free"],
        cutoff=14.0,
        dense_max_nodes=1,
        use_position_cache=True,
    )
    uncached = XDMDispersionEnergy(
        alpha_free=ref["alpha_free"], v_free=ref["v_free"], cutoff=14.0, dense_max_nodes=1
    )
    kwargs = dict(node_attrs=node_attrs, batch=batch, num_graphs=2, xdm_atomic=xdm_atomic)
    e_cached = cached(positions=positions, **kwargs)
    e_uncached = uncached(positions=positions, **kwargs)
    assert torch.allclose(e_cached, e_uncached, rtol=1e-10)
    assert cached._cache_ref_positions is None  # never touched -- fell through to the fresh path


def test_xdm_dispersion_reset_cache_forces_rebuild():
    """reset_cache() must drop the Verlet-skin cache so the next call does a
    fresh search, regardless of whether the displacement heuristic alone
    would have triggered one -- the escape hatch a caller reusing one
    instance across distinct structures needs (see MACEXDMDispersionCalculator)."""
    z_table = default_mlxdm_2x_atomic_number_table()
    ref = mlxdm_2x_polarizability_reference(z_table)
    n_atoms = 30
    positions, node_attrs, batch, xdm_atomic = _random_structure(z_table, n_atoms)
    kwargs = dict(node_attrs=node_attrs, batch=batch, num_graphs=1, xdm_atomic=xdm_atomic)

    module = XDMDispersionEnergy(
        alpha_free=ref["alpha_free"], v_free=ref["v_free"], cutoff=14.0, dense_max_nodes=1,
        use_position_cache=True,
    )
    module(positions=positions, **kwargs)
    assert module._cache_ref_positions is not None

    module.reset_cache()
    assert module._cache_ref_positions is None
    assert module._cache_idx_i is None
    assert module._cache_idx_j is None

    # Next call must rebuild (not crash on the cleared cache) and still agree
    # with a fully uncached evaluation.
    uncached = XDMDispersionEnergy(
        alpha_free=ref["alpha_free"], v_free=ref["v_free"], cutoff=14.0, dense_max_nodes=1,
    )
    e_cached = module(positions=positions, **kwargs)
    e_uncached = uncached(positions=positions, **kwargs)
    assert module._cache_ref_positions is not None
    assert torch.allclose(e_cached, e_uncached, rtol=1e-10)


def test_xdm_dispersion_apply_resets_cache():
    """The cache tensors are plain attributes, not buffers/parameters, so
    nn.Module._apply (which .to()/.double()/.float()/.cuda() all go through)
    won't move them itself. Reusing a stale, now-wrong-device/dtype cache
    afterwards would crash; _apply is overridden to drop the cache instead
    so the next call transparently rebuilds."""
    z_table = default_mlxdm_2x_atomic_number_table()
    ref = mlxdm_2x_polarizability_reference(z_table)
    n_atoms = 30
    positions, node_attrs, batch, xdm_atomic = _random_structure(z_table, n_atoms)
    kwargs = dict(node_attrs=node_attrs, batch=batch, num_graphs=1, xdm_atomic=xdm_atomic)

    module = XDMDispersionEnergy(
        alpha_free=ref["alpha_free"], v_free=ref["v_free"], cutoff=14.0, dense_max_nodes=1,
        use_position_cache=True,
    )
    module(positions=positions, **kwargs)
    assert module._cache_ref_positions is not None

    module.float()  # exercises _apply; no-op numerically for this test's purposes
    assert module._cache_ref_positions is None
    assert module._cache_idx_i is None
    assert module._cache_idx_j is None

    module.double()
    # Must not crash reusing a cache built under a different dtype, and must
    # produce a fresh, correct cache again.
    module(positions=positions, **kwargs)
    assert module._cache_ref_positions is not None


def test_xdm_dispersion_negative_moments_do_not_produce_nan():
    """c6/c8/c10 are physically positive (even moments of a positive exchange
    hole), but nothing upstream constrains an AtomicXDMMACE readout to
    predict positive M1/M2/M3 -- an undertrained/OOD prediction can go
    negative, and sqrt()/a fractional power of that is NaN, which would
    otherwise poison the whole graph's energy (and, via autograd, every
    parameter touched by that batch)."""
    z_table = AtomicNumberTable([1, 6])
    ref = mlxdm_2x_polarizability_reference(z_table)
    module = XDMDispersionEnergy(alpha_free=ref["alpha_free"], v_free=ref["v_free"], cutoff=14.0)

    positions = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], requires_grad=True)
    node_attrs = _build_two_atom_graph(z_table, [1, 6], positions)
    batch = torch.zeros(2, dtype=torch.long)
    # A negative M1 on atom A alone drives c6 = M1_A*M1_B/denom negative
    # (denom = M1_A/alpha_A + M1_B/alpha_B, also now negative-dominated).
    xdm_atomic = torch.tensor([[-1.0, 10.0, 100.0, 6.0], [3.0, 20.0, 300.0, 25.0]])

    e = module(positions=positions, node_attrs=node_attrs, batch=batch, num_graphs=1, xdm_atomic=xdm_atomic)
    assert torch.isfinite(e).all()
    e.sum().backward()
    assert torch.isfinite(positions.grad).all()


def test_xdm_dispersion_dense_and_sparse_agree_at_exact_cutoff():
    """matscipy's neighbour_list (the sparse path's backend) includes a pair
    exactly at the cutoff distance; the dense path's `dist < cutoff` does
    not. Both _fresh_sparse_pairs and _cached_sparse_pairs re-filter to a
    strict `r < cutoff` so all three pair-finding strategies agree on this
    boundary, as the module docstring claims they do."""
    z_table = AtomicNumberTable([1, 6])
    ref = mlxdm_2x_polarizability_reference(z_table)
    cutoff = 5.0
    module = XDMDispersionEnergy(
        alpha_free=ref["alpha_free"], v_free=ref["v_free"], cutoff=cutoff, dense_max_nodes=1,
    )
    positions = torch.tensor([[0.0, 0.0, 0.0], [cutoff, 0.0, 0.0]])  # exactly at cutoff
    node_attrs = _build_two_atom_graph(z_table, [1, 6], positions)
    batch = torch.zeros(2, dtype=torch.long)

    idx_i_dense, idx_j_dense, _ = module._dense_pairs(positions, batch)
    idx_i_sparse, idx_j_sparse, _ = module._fresh_sparse_pairs(positions, batch, num_graphs=1)
    assert idx_i_dense.numel() == 0
    assert idx_i_sparse.numel() == 0, (
        "matscipy's neighbour_list includes the exact-cutoff pair (its own "
        "convention), so this only passes if _fresh_sparse_pairs re-filters"
    )


def test_mace_xdm_dispersion_r_max_mismatch_warns(caplog):
    """If short_range_model needs a wider cutoff than xdm_model, the ONE
    shared graph built at xdm_model's r_max (the usual convention) silently
    truncates the short-range model's receptive field. MACEXDMDispersion
    can't see what cutoff the caller actually built the graph with, so this
    is a best-effort warning, not a guarantee -- but it must fire."""
    xdm_z_table = default_mlxdm_2x_atomic_number_table()
    xdm_model = _build_xdm_model(xdm_z_table)  # r_max=5.0, see _build_xdm_model
    assert float(xdm_model.r_max.item()) == 5.0

    sr_z_table = AtomicNumberTable([8, 1, 6])
    sr_model = _build_short_range_model(sr_z_table, atomic_energies=[-75.0, -0.5, -37.8])
    sr_model.r_max = torch.tensor(6.0, dtype=torch.get_default_dtype())  # wider than xdm's

    disp_ref = mlxdm_2x_polarizability_reference(xdm_z_table)
    disp_module = XDMDispersionEnergy(alpha_free=disp_ref["alpha_free"], v_free=disp_ref["v_free"])

    with caplog.at_level("WARNING"):
        MACEXDMDispersion(short_range_model=sr_model, xdm_model=xdm_model, dispersion_energy=disp_module)
    assert any("r_max" in rec.message for rec in caplog.records)


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
