import numpy as np
import torch
from ase import Atoms

from mace.calculators.xdm_dispersion import MACEXDMDispersionCalculator
from mace.data import default_mlxdm_2x_atomic_number_table
from tests.test_xdm_dispersion import _build_short_range_model, _build_xdm_model

torch.set_default_dtype(torch.float64)


def _save_models(tmp_path, sr_r_max=5.0, xdm_r_max=5.0):
    xdm_z_table = default_mlxdm_2x_atomic_number_table()
    xdm_model = _build_xdm_model(xdm_z_table)
    xdm_model.r_max = torch.tensor(xdm_r_max, dtype=torch.get_default_dtype())

    sr_model = _build_short_range_model(xdm_z_table, atomic_energies=np.zeros(len(xdm_z_table)))
    sr_model.r_max = torch.tensor(sr_r_max, dtype=torch.get_default_dtype())

    sr_path = tmp_path / "pbe0.model"
    xdm_path = tmp_path / "xdm.model"
    torch.save(sr_model, sr_path)
    torch.save(xdm_model, xdm_path)
    return str(sr_path), str(xdm_path)


def _water():
    return Atoms(
        "OHH",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.76, 0.59], [0.0, -0.76, 0.59]],
    )


def test_calculator_r_max_defaults_to_max_of_both_models(tmp_path):
    """The one AtomicData graph built in calculate() is shared by both
    sub-models; defaulting to xdm_model.r_max alone would silently truncate
    the short-range model's receptive field whenever it needs a wider
    cutoff (see the matching guard in MACEXDMDispersion.__init__)."""
    sr_path, xdm_path = _save_models(tmp_path, sr_r_max=6.0, xdm_r_max=5.0)
    calc = MACEXDMDispersionCalculator(sr_path, xdm_path, device="cpu")
    assert calc.r_max == 6.0

    sr_path2, xdm_path2 = _save_models(tmp_path, sr_r_max=5.0, xdm_r_max=7.0)
    calc2 = MACEXDMDispersionCalculator(sr_path2, xdm_path2, device="cpu")
    assert calc2.r_max == 7.0

    # An explicit r_max is still honored as an override.
    calc3 = MACEXDMDispersionCalculator(sr_path, xdm_path, device="cpu", r_max=9.0)
    assert calc3.r_max == 9.0


def test_calculator_energy_runs_and_is_finite(tmp_path):
    sr_path, xdm_path = _save_models(tmp_path)
    calc = MACEXDMDispersionCalculator(sr_path, xdm_path, device="cpu")
    atoms = _water()
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    assert np.isfinite(energy)
    assert np.isfinite(forces).all()


def test_calculator_resets_cache_between_distinct_atoms_objects(tmp_path):
    """The dispersion position cache is only safe to reuse within one Atoms
    object's own trajectory; calculate() must drop it whenever the Atoms
    identity changes, rather than relying solely on the cache's own
    shape/displacement heuristic."""
    sr_path, xdm_path = _save_models(tmp_path)
    calc = MACEXDMDispersionCalculator(sr_path, xdm_path, device="cpu")
    disp = calc.model.dispersion_energy
    disp.dense_max_nodes = 1  # force the cached sparse path for this 3-atom test structure

    atoms1 = _water()
    atoms1.calc = calc
    atoms1.get_potential_energy()
    assert disp._cache_ref_positions is not None
    ref_after_1 = disp._cache_ref_positions.clone()

    # Same Atoms object, positions nudged in place (as an ASE Dynamics loop
    # would do) -- must NOT force a reset via the id(atoms) check; whether it
    # rebuilds at all is governed by the existing displacement heuristic.
    atoms1.positions = atoms1.positions + 1e-4
    atoms1.get_potential_energy()

    # A distinct Atoms object, even with identical coordinates, must reset
    # the cache rather than silently reusing atoms1's stale pair list.
    atoms2 = _water()
    atoms2.calc = calc
    atoms2.get_potential_energy()
    assert disp._cache_ref_positions is not None
    assert torch.equal(
        disp._cache_ref_positions, torch.tensor(atoms2.get_positions(), dtype=torch.get_default_dtype())
    )
