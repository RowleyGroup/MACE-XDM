###########################################################################################
# Differentiable Becke-Johnson-damped XDM dispersion energy, computed from
# per-atom M1, M2, M3 (exchange-hole moments) and Veff (effective volume) --
# the quantities AtomicXDMMACE predicts.
#
# Formulas and reference constants transcribed from RowleyGroup/MLXDM
# (torchanipbe0/dispersion/nn.py: C6/C8/C10CombineLayer, PolarizabilityLayer,
# vanderWaalsLayer, EnergyLayer). Operates on whole (finite, non-periodic)
# molecules, matched into pairs by graph membership and cutoff.
#
# For small structures (the typical training case: batches of independent
# molecules of up to a few hundred atoms each) pairs are found via a dense
# [n_nodes, n_nodes] distance matrix on the GPU -- cheap relative to either
# neural network's forward pass, and simplest to keep exactly-differentiable.
#
# Above dense_max_nodes, that dense matrix (and the same-sized boolean masks
# built alongside it) becomes the dominant memory cost -- O(n_nodes^2) -- which
# matters once this module is driven by an MD loop over a single large (e.g.
# multi-thousand atom) structure rather than a batch of small ones. Above the
# threshold, pairs are instead found with a cell-list neighbor search
# (matscipy, the same backend `mace.data.neighborhood.get_neighborhood` uses
# for the short-range MACE cutoff), run once per graph on CPU -- giving memory
# that scales with the true number of within-cutoff pairs instead of with
# n_nodes^2, at the cost of a GPU->CPU->GPU round trip every call. The
# per-pair energy formulas afterwards are identical in all cases; only how
# (idx_i, idx_j, r) are produced differs.
#
# That CPU round trip dominates real MD wall time once the dense path's
# O(n_nodes^2) memory gets too big to use: measured on an H100 (real
# AtomicXDMMACE + this module, MACE-PBE0+XDM benchmark), the neighbor search
# alone was 80-90% of the total step cost at 2700-6300 atoms, growing
# supralinearly (~N^1.8) while the two neural networks stayed ~linear. Two
# independent, stackable mitigations:
#   1. dense_max_nodes raised from the old fixed 512 to 2048 (~150MB for the
#      dense tensors at 2048 nodes -- negligible on any GPU this runs on) so
#      more real MD sizes skip the CPU path entirely.
#   2. use_position_cache=True, for whatever's still above that: an MD
#      trajectory evaluates the *same* structure over and over with small
#      per-step displacements, so instead of a fresh CPU search every call,
#      search once at cutoff+cache_skin and reuse that candidate pair set
#      until some atom has moved far enough that a pair could plausibly have
#      entered/left the true cutoff shell (standard Verlet-list skin logic).
#      Distances are still recomputed fresh from current positions every
#      call, so results are numerically exact between rebuilds -- only the
#      pair *set* is cached, not the energies. Measured locally: ~1 rebuild
#      per 18 MD steps under Langevin dynamics. ONLY SAFE when forward() is
#      called repeatedly on the same physical structure (an MD/relaxation
#      loop) -- default is off, since it would silently corrupt batched
#      training/eval over many unrelated molecules of the same atom count.
###########################################################################################

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from mace.data.neighborhood import get_neighborhood
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

    # Default above which pair-finding switches from a dense [n,n] GPU distance
    # matrix to a CPU cell-list search (see module docstring). Override per
    # instance via dense_max_nodes= if you know your GPU has room for more.
    _DENSE_MAX_NODES = 2048

    def __init__(
        self,
        alpha_free: Union[np.ndarray, torch.Tensor],
        v_free: Union[np.ndarray, torch.Tensor],
        cutoff: float = 14.0,
        a1: float = 0.4186,
        a2: float = 2.6791,
        dense_max_nodes: int = _DENSE_MAX_NODES,
        use_position_cache: bool = False,
        cache_skin: float = 2.0,
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
        self.dense_max_nodes = dense_max_nodes
        self.use_position_cache = use_position_cache
        self.cache_skin = cache_skin
        self._cache_ref_positions: Optional[torch.Tensor] = None
        self._cache_idx_i: Optional[torch.Tensor] = None
        self._cache_idx_j: Optional[torch.Tensor] = None

    def _dense_pairs(
        self, positions: torch.Tensor, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pair-finding via a dense [n,n] distance matrix. O(n_nodes^2) memory
        -- only used below dense_max_nodes nodes (see module docstring)."""
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
        r = dist[idx_i, idx_j]
        return idx_i, idx_j, r

    def _sparse_pairs(
        self, positions: torch.Tensor, batch: torch.Tensor, num_graphs: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pair-finding via a per-graph cell-list neighbor search (CPU,
        matscipy). Memory scales with the number of within-cutoff pairs
        rather than n_nodes^2 -- used above dense_max_nodes nodes. Routes to
        the Verlet-skin cache when that's enabled and safe (single graph)."""
        if self.use_position_cache and num_graphs == 1:
            return self._cached_sparse_pairs(positions)
        return self._fresh_sparse_pairs(positions, batch, num_graphs)

    def _cached_sparse_pairs(
        self, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Verlet-skin cache for the single-graph case: search once at
        cutoff+cache_skin, then reuse that candidate pair set across calls
        until some atom has moved more than cache_skin/2 since the last
        search (so a pair could plausibly have crossed the true cutoff).
        Distances are recomputed from the CURRENT positions every call, so
        results are numerically identical to a fresh search between
        rebuilds -- only which pairs to check is cached, not the energies.

        Only call this when the same physical structure is evaluated
        repeatedly with small per-step displacements (an MD or relaxation
        loop) -- see the "use_position_cache" note in the module docstring
        for why this is unsafe for batched/i.i.d. training or eval data."""
        cutoff = float(self.cutoff.item())
        ref = self._cache_ref_positions
        rebuild = ref is None or ref.shape != positions.shape
        if not rebuild:
            disp = (positions.detach() - ref).norm(dim=-1).max()
            rebuild = bool(2.0 * disp.item() > self.cache_skin)
        if rebuild:
            pos_np = positions.detach().cpu().numpy()
            edge_index, _, _, _ = get_neighborhood(
                positions=pos_np, cutoff=cutoff + self.cache_skin
            )
            sender, receiver = edge_index[0], edge_index[1]
            keep = sender < receiver
            self._cache_idx_i = torch.as_tensor(
                sender[keep], dtype=torch.long, device=positions.device
            )
            self._cache_idx_j = torch.as_tensor(
                receiver[keep], dtype=torch.long, device=positions.device
            )
            self._cache_ref_positions = positions.detach().clone()
        idx_i, idx_j = self._cache_idx_i, self._cache_idx_j
        assert idx_i is not None and idx_j is not None  # set on the rebuild branch above
        r = torch.linalg.norm(positions[idx_i] - positions[idx_j], dim=-1)
        mask = r < cutoff
        return idx_i[mask], idx_j[mask], r[mask]

    def _fresh_sparse_pairs(
        self, positions: torch.Tensor, batch: torch.Tensor, num_graphs: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The uncached search: exact, safe for any batch of graphs, used
        directly when use_position_cache is off and as the cache's own
        rebuild step (at cutoff+cache_skin instead of cutoff) when it's on."""
        cutoff = float(self.cutoff.item())
        positions_np = positions.detach().cpu().numpy()
        batch_np = batch.detach().cpu().numpy()

        all_i: List[torch.Tensor] = []
        all_j: List[torch.Tensor] = []
        for g in range(num_graphs):
            node_idx_np = np.nonzero(batch_np == g)[0]
            if node_idx_np.size < 2:
                continue
            edge_index, _, _, _ = get_neighborhood(
                positions=positions_np[node_idx_np], cutoff=cutoff
            )
            sender, receiver = edge_index[0], edge_index[1]
            keep = sender < receiver  # one direction per pair, like triu(diagonal=1)
            if not keep.any():
                continue
            node_idx = torch.as_tensor(node_idx_np, dtype=torch.long)
            sender = torch.as_tensor(sender[keep], dtype=torch.long)
            receiver = torch.as_tensor(receiver[keep], dtype=torch.long)
            all_i.append(node_idx[sender])
            all_j.append(node_idx[receiver])

        if not all_i:
            empty = torch.empty(0, dtype=torch.long, device=positions.device)
            return empty, empty, positions.new_zeros(0)

        idx_i = torch.cat(all_i).to(positions.device)
        idx_j = torch.cat(all_j).to(positions.device)
        diff = positions[idx_i] - positions[idx_j]  # [n_pairs, 3]
        r = torch.linalg.norm(diff, dim=-1)  # [n_pairs]
        return idx_i, idx_j, r

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
        if n_nodes > self.dense_max_nodes:
            idx_i, idx_j, r = self._sparse_pairs(positions, batch, num_graphs)
        else:
            idx_i, idx_j, r = self._dense_pairs(positions, batch)

        if idx_i.numel() == 0:
            zeros = torch.zeros(num_graphs, dtype=positions.dtype, device=positions.device)
            if return_components:
                return {"total": zeros, "e6": zeros, "e8": zeros, "e10": zeros}
            return zeros

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
