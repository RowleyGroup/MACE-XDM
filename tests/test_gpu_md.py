"""Tests for the ASE-free GPU minimization / velocity-Verlet / Langevin MD
helpers in scripts/gpu_md.py. Uses a small randomly-initialized MACE model
(no foundation-model checkpoint needed) so the suite stays fast."""

import numpy as np
import pytest
import torch
import torch.nn.functional
from e3nn import o3

from mace import modules, tools
from scripts.gpu_md import (
    System,
    init_maxwell_boltzmann,
    langevin,
    minimize_fire,
    read_xyz,
    velocity_verlet,
)


@pytest.fixture(name="tiny_model")
def tiny_model_fixture():
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        z_table = tools.AtomicNumberTable([1, 8])
        model = modules.MACE(
            r_max=3.0,
            num_bessel=6,
            num_polynomial_cutoff=5,
            max_ell=2,
            interaction_cls=modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            num_interactions=2,
            num_elements=2,
            hidden_irreps=o3.Irreps("8x0e + 8x1o"),
            MLP_irreps=o3.Irreps("8x0e"),
            gate=torch.nn.functional.silu,
            atomic_energies=np.array([1.0, 3.0]),
            avg_num_neighbors=4.0,
            atomic_numbers=z_table.zs,
            correlation=2,
            radial_type="bessel",
        )
        yield model.eval()
    finally:
        torch.set_default_dtype(old_dtype)


def water_geometry():
    atomic_numbers = np.array([8, 1, 1])
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.757, 0.586, 0.0],
            [-0.757, 0.586, 0.0],
        ]
    )
    return atomic_numbers, positions


def oxygen_box():
    """Small periodic simple-cubic lattice of oxygen atoms."""
    a = 3.0
    reps = 2
    frac = np.array(
        [(i, j, k) for i in range(reps) for j in range(reps) for k in range(reps)],
        dtype=float,
    )
    cell = np.eye(3) * a * reps
    positions = frac * a
    atomic_numbers = np.full(positions.shape[0], 8, dtype=np.int64)
    return atomic_numbers, positions, cell


def test_energy_forces_finite(tiny_model):
    atomic_numbers, positions = water_geometry()
    system = System(
        tiny_model, atomic_numbers, positions, pbc=(False, False, False), device="cpu"
    )
    energy, forces, _ = system.energy_forces()
    assert np.isfinite(energy)
    assert forces.shape == (3, 3)
    assert torch.isfinite(forces).all()


def test_forces_match_finite_difference(tiny_model):
    atomic_numbers, positions = water_geometry()
    system = System(
        tiny_model, atomic_numbers, positions, pbc=(False, False, False), device="cpu"
    )
    _, analytic_forces, _ = system.energy_forces()

    eps = 1e-4
    numeric_forces = np.zeros((3, 3))
    for i in range(3):
        for k in range(3):
            base = system.positions.clone()
            system.positions = base.clone()
            system.positions[i, k] += eps
            e_plus, _, _ = system.energy_forces()
            system.positions = base.clone()
            system.positions[i, k] -= eps
            e_minus, _, _ = system.energy_forces()
            system.positions = base
            numeric_forces[i, k] = -(e_plus - e_minus) / (2 * eps)

    np.testing.assert_allclose(
        analytic_forces.numpy(), numeric_forces, atol=1e-4, rtol=1e-3
    )


def test_minimize_fire_reduces_forces(tiny_model):
    atomic_numbers, positions = water_geometry()
    rng = np.random.default_rng(0)
    perturbed = positions + rng.normal(scale=0.3, size=positions.shape)

    system = System(
        tiny_model, atomic_numbers, perturbed, pbc=(False, False, False), device="cpu"
    )
    _, forces0, _ = system.energy_forces()
    fmax0 = forces0.abs().max().item()

    fmax_history = []
    minimize_fire(
        system,
        fmax=1e-6,
        steps=100,
        callback=lambda step, e, fmax: fmax_history.append(fmax),
    )

    assert fmax_history[-1] < fmax0
    assert fmax_history[-1] < 0.1


def test_velocity_verlet_conserves_energy(tiny_model):
    atomic_numbers, positions, cell = oxygen_box()
    system = System(
        tiny_model,
        atomic_numbers,
        positions,
        cell=cell,
        pbc=(True, True, True),
        device="cpu",
    )
    torch.manual_seed(0)
    init_maxwell_boltzmann(system, temperature_K=300.0, seed=0)

    total_energies = []

    def cb(step, e_pot, e_kin, temperature):
        total_energies.append(e_pot + e_kin)
        assert temperature >= 0.0

    velocity_verlet(system, dt_fs=0.5, n_steps=50, callback=cb)

    total_energies = np.array(total_energies)
    # Total energy should be conserved to well within 1% of its scale.
    assert total_energies.std() < 1e-2 * abs(total_energies.mean())


def test_langevin_zero_friction_matches_velocity_verlet(tiny_model):
    """At friction=0 the Langevin random kicks vanish and its coefficients
    reduce exactly to the velocity-Verlet half-step update, so the two
    integrators should produce identical trajectories."""
    atomic_numbers, positions, cell = oxygen_box()
    system_vv = System(
        tiny_model, atomic_numbers, positions, cell=cell, pbc=(True, True, True), device="cpu"
    )
    torch.manual_seed(0)
    init_maxwell_boltzmann(system_vv, temperature_K=300.0, seed=0)

    system_lgv = System(
        tiny_model, atomic_numbers, positions, cell=cell, pbc=(True, True, True), device="cpu"
    )
    system_lgv.velocities = system_vv.velocities.clone()

    velocity_verlet(system_vv, dt_fs=0.5, n_steps=20)
    langevin(
        system_lgv,
        dt_fs=0.5,
        n_steps=20,
        temperature_K=300.0,
        friction=0.0,
        fixcm=False,
        seed=1,
    )

    np.testing.assert_allclose(
        system_lgv.positions.numpy(), system_vv.positions.numpy(), atol=1e-10
    )
    np.testing.assert_allclose(
        system_lgv.velocities.numpy(), system_vv.velocities.numpy(), atol=1e-10
    )


def test_langevin_thermostats_towards_target_temperature(tiny_model):
    atomic_numbers, positions, cell = oxygen_box()
    system = System(
        tiny_model, atomic_numbers, positions, cell=cell, pbc=(True, True, True), device="cpu"
    )

    target_t = 300.0
    temperatures = []

    def cb(step, e_pot, e_kin, temperature):
        if step >= 200:
            temperatures.append(temperature)

    langevin(
        system,
        dt_fs=0.5,
        n_steps=1000,
        temperature_K=target_t,
        friction=0.05,
        seed=0,
        callback=cb,
    )

    assert all(np.isfinite(system.positions.numpy()).ravel())
    mean_t = np.mean(temperatures)
    assert 0.3 * target_t < mean_t < 3.0 * target_t


def test_neighbor_list_skin_rebuild(tiny_model):
    atomic_numbers, positions, cell = oxygen_box()
    system = System(
        tiny_model,
        atomic_numbers,
        positions,
        cell=cell,
        pbc=(True, True, True),
        device="cpu",
        skin=1.0,
    )
    system.positions = system.positions + 1e-4
    assert system.maybe_rebuild_neighbors() is False

    system.positions = system.positions + 10.0
    assert system.maybe_rebuild_neighbors() is True


def test_read_xyz_periodic(tmp_path):
    xyz_path = tmp_path / "test.xyz"
    xyz_path.write_text(
        "2\n"
        'Lattice="5.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0 5.0" pbc="T T T"\n'
        "O 0.000000 0.000000 0.000000\n"
        "H 1.000000 0.000000 0.000000\n"
    )
    atomic_numbers, positions, cell = read_xyz(str(xyz_path))
    np.testing.assert_array_equal(atomic_numbers, [8, 1])
    np.testing.assert_allclose(positions, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(cell, np.eye(3) * 5.0)


def test_read_xyz_non_periodic(tmp_path):
    xyz_path = tmp_path / "test.xyz"
    xyz_path.write_text("1\ncomment\nH 0.0 0.0 0.0\n")
    atomic_numbers, positions, cell = read_xyz(str(xyz_path))
    np.testing.assert_array_equal(atomic_numbers, [1])
    np.testing.assert_allclose(positions, [[0.0, 0.0, 0.0]])
    assert cell is None
