###########################################################################################
# Differentiable Becke-Johnson-damped XDM dispersion energy, computed from
# per-atom M1, M2, M3 (exchange-hole moments) and Veff (effective volume) --
# the quantities AtomicXDMMACE predicts.
#
# Formulas and reference constants transcribed from RowleyGroup/MLXDM
# (torchanipbe0/dispersion/nn.py: C6/C8/C10CombineLayer, PolarizabilityLayer,
# vanderWaalsLayer, EnergyLayer). Operates on whole (finite, non-periodic)
# molecules via a dense pairwise distance matrix masked by graph membership
# and cutoff, rather than a fixed-radius neighbor list -- appropriate since
# the dispersion cutoff (14 A by default) is much larger than a typical
# short-range MACE cutoff, and molecules in this dataset are small enough
# (tens to a couple hundred atoms) that an O(n_atoms^2) distance matrix is
# cheap relative to either neural network's forward pass.
###########################################################################################

from typing import Dict, Union

import numpy as np
import torch

from mace.tools.scatter import scatter_sum

BOHR_TO_ANGSTROM = 0.529177249


@torch.jit.unused
def _validate_shapes(alpha_free: torch.Tensor, v_free: torch.Tensor) -> None:
    assert alpha_free.shape == v_free.shape
    assert alpha_free.dim() == 1


class XDMDispersionEnergy(torch.nn.Module):
    """Becke-Johnson-damped XDM dispersion energy for finite molecules.

    Given per-atom (M1, M2, M3, Veff) -- e.g. from AtomicXDMMACE -- and
    positions, computes the total (summed C6+C8+C10) dispersion energy per
    molecule in a batch. Output is in Hartree (the natural unit of the XDM
    combining-rule formulas, since M1/M2/M3/Veff/alpha_free/v_free are all in
    atomic units); convert to eV (x 27.211386245988) before adding to a MACE
    energy model's output, which is conventionally in eV.
    """

    alpha_free: torch.Tensor
    v_free: torch.Tensor

    def __init__(
        self,
        alpha_free: Union[np.ndarray, torch.Tensor],
        v_free: Union[np.ndarray, torch.Tensor],
        cutoff: float = 14.0,
        a1: float = 0.4186,
        a2: float = 2.6791,
    ):
        super().__init__()
        alpha_free_t = torch.as_tensor(alpha_free, dtype=torch.get_default_dtype())
        v_free_t = torch.as_tensor(v_free, dtype=torch.get_default_dtype())
        _validate_shapes(alpha_free_t, v_free_t)
        self.register_buffer("alpha_free", alpha_free_t)  # [n_elements]
        self.register_buffer("v_free", v_free_t)  # [n_elements]
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.get_default_dtype()))
        self.register_buffer("a1", torch.tensor(a1, dtype=torch.get_default_dtype()))
        self.register_buffer("a2", torch.tensor(a2, dtype=torch.get_default_dtype()))
        self.register_buffer(
            "bohr_to_angstrom", torch.tensor(BOHR_TO_ANGSTROM, dtype=torch.get_default_dtype())
        )

    def forward(
        self,
        positions: torch.Tensor,  # [n_nodes, 3], Angstrom
        node_attrs: torch.Tensor,  # [n_nodes, n_elements], one-hot
        batch: torch.Tensor,  # [n_nodes], graph index per atom
        num_graphs: int,
        xdm_atomic: torch.Tensor,  # [n_nodes, 4] = (M1, M2, M3, Veff), atomic units
        return_components: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # [num_graphs] Hartree, or (if return_components) a dict with the
        # separate C6/C8/C10 terms alongside "total" -- e.g. to judge how much
        # of a dispersion-energy error traces back to each term.
        m1 = xdm_atomic[:, 0]
        m2 = xdm_atomic[:, 1]
        m3 = xdm_atomic[:, 2]
        veff = xdm_atomic[:, 3]

        alpha_free_atom = torch.matmul(node_attrs, self.alpha_free.to(dtype=node_attrs.dtype))
        v_free_atom = torch.matmul(node_attrs, self.v_free.to(dtype=node_attrs.dtype))
        alpha = veff * alpha_free_atom / v_free_atom  # [n_nodes]

        n_nodes = positions.shape[0]
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # [n,n,3]
        dist = torch.linalg.norm(diff, dim=-1)  # [n,n]

        same_graph = batch.unsqueeze(1) == batch.unsqueeze(0)
        upper = torch.triu(
            torch.ones(n_nodes, n_nodes, dtype=torch.bool, device=positions.device),
            diagonal=1,
        )
        within_cutoff = dist < self.cutoff
        mask = same_graph & upper & within_cutoff

        idx_i, idx_j = torch.nonzero(mask, as_tuple=True)
        if idx_i.numel() == 0:
            zeros = torch.zeros(num_graphs, dtype=positions.dtype, device=positions.device)
            if return_components:
                return {"total": zeros, "e6": zeros, "e8": zeros, "e10": zeros}
            return zeros

        r = dist[idx_i, idx_j]
        m1_i, m1_j = m1[idx_i], m1[idx_j]
        m2_i, m2_j = m2[idx_i], m2[idx_j]
        m3_i, m3_j = m3[idx_i], m3[idx_j]
        alpha_i, alpha_j = alpha[idx_i], alpha[idx_j]

        denom = m1_i / alpha_i + m1_j / alpha_j
        c6 = m1_i * m1_j / denom
        c8 = 1.5 * (m1_i * m2_j + m1_j * m2_i) / denom
        c10 = 2.0 * (m1_i * m3_j + m3_i * m1_j + 2.1 * m2_i * m2_j) / denom

        r_crit = (
            torch.sqrt(c8 / c6) + torch.pow(c10 / c6, 0.25) + torch.sqrt(c10 / c8)
        ) / 3.0  # bohr
        r_vdw = self.a2 + self.a1 * r_crit * self.bohr_to_angstrom  # Angstrom

        e6_pair = -c6 / (r.pow(6) + r_vdw.pow(6)) * self.bohr_to_angstrom.pow(6)
        e8_pair = -c8 / (r.pow(8) + r_vdw.pow(8)) * self.bohr_to_angstrom.pow(8)
        e10_pair = -c10 / (r.pow(10) + r_vdw.pow(10)) * self.bohr_to_angstrom.pow(10)

        graph_idx = batch[idx_i]
        total = scatter_sum(
            e6_pair + e8_pair + e10_pair, graph_idx, dim=0, dim_size=num_graphs
        )
        if return_components:
            return {
                "total": total,
                "e6": scatter_sum(e6_pair, graph_idx, dim=0, dim_size=num_graphs),
                "e8": scatter_sum(e8_pair, graph_idx, dim=0, dim_size=num_graphs),
                "e10": scatter_sum(e10_pair, graph_idx, dim=0, dim_size=num_graphs),
            }
        return total
