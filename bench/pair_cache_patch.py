"""PROTOTYPE (not applied to the library):  Verlet-skin cache for XDMDispersionEnergy._sparse_pairs (single graph).

Pairs are searched (CPU matscipy) at cutoff+skin only when some atom has moved
more than skin/2 since the last search; every call recomputes r on the GPU for the
cached superset and masks r < cutoff, so the pair set is identical to a fresh
search at `cutoff`.
"""
import numpy as np
import torch

import mace.modules.xdm_dispersion as xd

SKIN = 2.0
STATS = {"rebuilds": 0, "calls": 0, "rebuild_ms": 0.0}


def install():
    import time
    orig = xd.XDMDispersionEnergy._sparse_pairs

    def cached(self, positions, batch, num_graphs):
        assert num_graphs == 1, "prototype handles a single graph"
        cutoff = float(self.cutoff.item())
        st = getattr(self, "_pair_cache", None)
        STATS["calls"] += 1
        rebuild = st is None or st["ref"].shape != positions.shape
        if not rebuild:
            disp = (positions.detach() - st["ref"]).norm(dim=-1).max().item()
            rebuild = 2.0 * disp > SKIN
        if rebuild:
            t = time.perf_counter()
            pos_np = positions.detach().cpu().numpy()
            edge_index, _, _, _ = xd.get_neighborhood(positions=pos_np, cutoff=cutoff + SKIN)
            s, r_ = edge_index[0], edge_index[1]
            keep = s < r_
            st = {
                "ref": positions.detach().clone(),
                "i": torch.as_tensor(s[keep], dtype=torch.long, device=positions.device),
                "j": torch.as_tensor(r_[keep], dtype=torch.long, device=positions.device),
            }
            self._pair_cache = st
            STATS["rebuilds"] += 1
            STATS["rebuild_ms"] += (time.perf_counter() - t) * 1e3
        idx_i, idx_j = st["i"], st["j"]
        r = torch.linalg.norm(positions[idx_i] - positions[idx_j], dim=-1)
        m = r < cutoff
        return idx_i[m], idx_j[m], r[m]

    xd.XDMDispersionEnergy._sparse_pairs = cached
    return orig
