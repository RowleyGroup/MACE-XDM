###########################################################################################
# Combines a short-range MACE energy/forces model with a trained AtomicXDMMACE
# dispersion-coefficient model into a single potential:
#
#     E_total = E_short_range(structure) + E_XDM_dispersion(structure)
#
# This mirrors RowleyGroup/MLXDM's own ANIDispersion module, which is a plain
# sum of an ANI backbone's energy and a dispersion model's energy (verified
# from its source: `energy = species_energy[1] + self.disp_model(x)[1]`) --
# the standard DFT+D-style recipe, where the short-range model is trained on
# dispersion-deficient reference energies (e.g. bare PBE0) and XDM supplies
# the missing long-range correlation on top, rather than any joint/residual
# training scheme.
###########################################################################################

from typing import Dict, Optional

import torch

from .utils import compute_forces
from .xdm_dispersion import XDMDispersionEnergy

HARTREE_TO_EV = 27.211386245988


class MACEXDMDispersion(torch.nn.Module):
    """Additive combination of a short-range MACE energy model and an XDM
    dispersion correction built from a trained AtomicXDMMACE.

    The two sub-models are independently trained and may each use their own
    AtomicNumberTable/element ordering. The caller only needs to build one
    AtomicData graph, keyed to ``xdm_model``'s z_table (e.g. via
    ``build_xdm_atomic_data``/``XDMHDF5Dataset``); this class re-derives the
    short-range model's own one-hot ``node_attrs`` from the atomic numbers
    recovered out of that graph, so the two z_tables never need to match.

    Assumes both sub-models are single-head (or that head index 0 is the
    correct one for the short-range model); a multi-head foundation-model
    backbone would need an explicit ``head`` field threaded through, which is
    not yet supported here.
    """

    z_to_index_sr: torch.Tensor
    xdm_atomic_numbers_table: torch.Tensor
    hartree_to_ev: torch.Tensor

    def __init__(
        self,
        short_range_model: torch.nn.Module,
        xdm_model: torch.nn.Module,
        dispersion_energy: XDMDispersionEnergy,
        hartree_to_ev: float = HARTREE_TO_EV,
    ):
        super().__init__()
        self.short_range_model = short_range_model
        self.xdm_model = xdm_model
        self.dispersion_energy = dispersion_energy
        self.register_buffer(
            "hartree_to_ev", torch.tensor(hartree_to_ev, dtype=torch.get_default_dtype())
        )

        xdm_zs = xdm_model.atomic_numbers.tolist()
        sr_zs = short_range_model.atomic_numbers.tolist()
        max_z = max(max(xdm_zs), max(sr_zs))
        z_to_index_sr = torch.full((max_z + 1,), -1, dtype=torch.long)
        for idx, z in enumerate(sr_zs):
            z_to_index_sr[z] = idx
        self.register_buffer("z_to_index_sr", z_to_index_sr)
        self.register_buffer(
            "xdm_atomic_numbers_table", torch.tensor(xdm_zs, dtype=torch.long)
        )
        self.num_elements_sr = len(sr_zs)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
    ) -> Dict[str, Optional[torch.Tensor]]:
        num_graphs = int(data["ptr"].numel() - 1)
        data["positions"].requires_grad_(True)

        xdm_out = self.xdm_model(data)

        xdm_element_idx = torch.argmax(data["node_attrs"], dim=-1)
        atomic_numbers_per_atom = self.xdm_atomic_numbers_table[xdm_element_idx]
        sr_element_idx = self.z_to_index_sr[atomic_numbers_per_atom]
        if bool((sr_element_idx < 0).any()):
            missing = torch.unique(atomic_numbers_per_atom[sr_element_idx < 0]).tolist()
            raise ValueError(
                f"Short-range model does not support atomic numbers {missing} "
                f"present in this structure."
            )
        node_attrs_sr = torch.nn.functional.one_hot(
            sr_element_idx, num_classes=self.num_elements_sr
        ).to(dtype=data["node_attrs"].dtype)

        sr_data = dict(data)
        sr_data["node_attrs"] = node_attrs_sr
        short_range_out = self.short_range_model(
            sr_data, training=training, compute_force=False
        )

        e_disp_hartree = self.dispersion_energy(
            positions=data["positions"],
            node_attrs=data["node_attrs"],
            batch=data["batch"],
            num_graphs=num_graphs,
            xdm_atomic=xdm_out["xdm_atomic"],
        )
        e_disp_ev = e_disp_hartree * self.hartree_to_ev
        total_energy = short_range_out["energy"] + e_disp_ev

        forces = None
        if compute_force:
            forces = compute_forces(total_energy, data["positions"], training=training)

        return {
            "energy": total_energy,
            "forces": forces,
            "short_range_energy": short_range_out["energy"],
            "dispersion_energy": e_disp_ev,
            "xdm_atomic": xdm_out["xdm_atomic"],
        }
