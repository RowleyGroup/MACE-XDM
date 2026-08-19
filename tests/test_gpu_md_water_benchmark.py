"""Tests for the MACEXDM dispersion integration in scripts/gpu_md_water_benchmark.py.
Uses a small randomly-initialized MACE model (no foundation-model checkpoint needed) and
the real torchanipbe0 MLXDM_2x dispersion module (fast to build, pure-Python resources)."""

import numpy as np
import pytest
import torch
import torch.nn.functional
from e3nn import o3

pytest.importorskip("torchanipbe0")

from mace import modules, tools
from scripts.gpu_md import langevin
from scripts.gpu_md_water_benchmark import DispersionSystem, MLXDMDispersion


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


def two_water_geometry():
    atomic_numbers = np.array([8, 1, 1, 8, 1, 1])
    symbols = ["O", "H", "H", "O", "H", "H"]
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.757, 0.586, 0.0],
            [-0.757, 0.586, 0.0],
            [3.0, 0.0, 0.0],
            [3.757, 0.586, 0.0],
            [3.0, 1.0, 0.7],
        ]
    )
    return atomic_numbers, symbols, positions


def test_mlxdm_forces_match_finite_difference():
    _, symbols, positions = two_water_geometry()
    dispersion = MLXDMDispersion(torch.device("cpu"))
    dispersion.set_structure(symbols)

    positions_t = torch.tensor(positions, dtype=torch.float64)
    _, analytic_forces = dispersion.energy_forces(positions_t)

    eps = 1e-4
    numeric_forces = np.zeros((6, 3))
    for i in range(6):
        for k in range(3):
            p_plus = positions_t.clone()
            p_plus[i, k] += eps
            e_plus, _ = dispersion.energy_forces(p_plus)
            p_minus = positions_t.clone()
            p_minus[i, k] -= eps
            e_minus, _ = dispersion.energy_forces(p_minus)
            numeric_forces[i, k] = -(e_plus - e_minus) / (2 * eps)

    np.testing.assert_allclose(
        analytic_forces.numpy(), numeric_forces, atol=1e-4, rtol=1e-3
    )


def test_dispersion_system_adds_mace_and_mlxdm_contributions(tiny_model):
    atomic_numbers, symbols, positions = two_water_geometry()
    device = torch.device("cpu")

    plain_system = DispersionSystem(
        tiny_model, atomic_numbers, positions, pbc=(False, False, False),
        device=device, dispersion=None,
    )
    e_mace, f_mace, _ = plain_system.energy_forces()

    dispersion = MLXDMDispersion(device)
    dispersion.set_structure(symbols)
    e_disp, f_disp = dispersion.energy_forces(plain_system.positions)

    combined_system = DispersionSystem(
        tiny_model, atomic_numbers, positions, pbc=(False, False, False),
        device=device, dispersion=dispersion,
    )
    e_combined, f_combined, _ = combined_system.energy_forces()

    assert e_combined == pytest.approx(e_mace + e_disp, rel=1e-10)
    np.testing.assert_allclose(f_combined.numpy(), (f_mace + f_disp).numpy(), atol=1e-10)


def test_langevin_runs_with_dispersion_system(tiny_model):
    atomic_numbers, symbols, positions = two_water_geometry()
    device = torch.device("cpu")

    dispersion = MLXDMDispersion(device)
    dispersion.set_structure(symbols)

    system = DispersionSystem(
        tiny_model, atomic_numbers, positions, pbc=(False, False, False),
        device=device, dispersion=dispersion,
    )

    energies = []

    def cb(step, e_pot, e_kin, temperature):
        energies.append(e_pot)
        assert temperature >= 0.0

    langevin(system, dt_fs=0.5, n_steps=20, temperature_K=300.0, friction=0.01,
              seed=0, callback=cb)

    assert np.isfinite(system.positions.numpy()).all()
    assert all(np.isfinite(energies))
