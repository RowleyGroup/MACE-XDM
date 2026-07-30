###########################################################################################
# ASE calculator for a MACEXDMDispersion combined potential: a short-range MACE
# energy model plus an XDM dispersion correction from a trained AtomicXDMMACE.
###########################################################################################

from typing import Optional

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from mace.data import build_xdm_atomic_data, mlxdm_2x_polarizability_reference
from mace.modules import MACEXDMDispersion, XDMDispersionEnergy, load_xdm_model
from mace.tools import AtomicNumberTable, load_full_model
from mace.tools.torch_geometric.batch import Batch


class MACEXDMDispersionCalculator(Calculator):
    """ASE calculator for E_total = E_short_range(MACE) + E_dispersion(XDM).

    Loads a short-range MACE energy model (a full ``torch.save``'d module,
    e.g. ``<model_dir>/<name>.model`` from ``mace_run_train``) and a trained
    AtomicXDMMACE model -- which may be either a full saved module or a
    ``mace_run_train_xdm`` training checkpoint (``<name>_{latest,best}.pt``);
    both are handled automatically. Builds the dispersion-energy module from
    MLXDM's fixed reference constants by default (override with
    ``alpha_free``/``v_free`` for a different element set), and exposes the
    combination as a normal ASE calculator. Finite molecules only (no
    PBC/stress support, since the dispersion sum currently uses a dense
    non-periodic pairwise distance matrix -- see ``XDMDispersionEnergy``).
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(
        self,
        short_range_model_path: str,
        xdm_model_path: str,
        device: str = "cpu",
        r_max: Optional[float] = None,
        dispersion_cutoff: float = 14.0,
        a1: float = 0.4186,
        a2: float = 2.6791,
        alpha_free: Optional[np.ndarray] = None,
        v_free: Optional[np.ndarray] = None,
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        self.device = torch.device(device)

        short_range_model = load_full_model(short_range_model_path, device=self.device)
        xdm_model = load_xdm_model(xdm_model_path, device=self.device)

        xdm_z_table = AtomicNumberTable(xdm_model.atomic_numbers.tolist())
        if alpha_free is None or v_free is None:
            ref = mlxdm_2x_polarizability_reference(xdm_z_table)
            alpha_free = ref["alpha_free"]
            v_free = ref["v_free"]
        dispersion_energy = XDMDispersionEnergy(
            alpha_free=alpha_free, v_free=v_free, cutoff=dispersion_cutoff, a1=a1, a2=a2
        )

        self.model = MACEXDMDispersion(
            short_range_model=short_range_model,
            xdm_model=xdm_model,
            dispersion_energy=dispersion_energy,
        ).to(self.device)
        self.model.eval()

        self.z_table = xdm_z_table
        self.r_max = r_max if r_max is not None else float(xdm_model.r_max.item())

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms)

        targets = np.zeros((len(atoms), 4))
        graph = build_xdm_atomic_data(
            atomic_numbers=atoms.get_atomic_numbers(),
            positions=atoms.get_positions(),
            xdm_targets=targets,
            z_table=self.z_table,
            cutoff=self.r_max,
        )
        batch = Batch.from_data_list([graph]).to(self.device)
        data = batch.to_dict()

        out = self.model(data, training=False, compute_force=True)

        energy = float(out["energy"].detach().cpu().numpy()[0])
        forces = out["forces"].detach().cpu().numpy()

        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": forces,
        }
