"""Usage (repo root, mace env active, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1):
    python bench/profile_xdm_components.py <xdm.model> 100,200,300,400,700,1000 5 out.json
    PAIR_CACHE=1 python bench/profile_xdm_components.py ...   # same, with the Verlet-skin pair-cache prototype
Run it on the H100 node to split the real per-step cost of MACE+XDM into: XDM network, dispersion pair search
(CPU matscipy + copies), dispersion maths, and backward.

Break the per-step cost of the MACE+XDM combined potential into components,
vs. water-ball size. Short-range PBE0 is replaced by a tiny stand-in so the
remaining cost is the real AtomicXDMMACE (cueq) + the real XDMDispersionEnergy."""
import sys, os, time, json
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import numpy as np
import torch
import torch.nn.functional
from e3nn import o3

import mace.modules.xdm_dispersion as xd
from mace import modules, tools
from mace.data import mlxdm_2x_polarizability_reference
from mace.modules import MACEXDMDispersion, XDMDispersionEnergy, load_xdm_model
from mace.tools import AtomicNumberTable
import gpu_md

XDM_PATH = sys.argv[1]
SIZES = [int(x) for x in sys.argv[2].split(",")]
NREP = int(sys.argv[3]) if len(sys.argv) > 3 else 5
dev = "cuda"


def sync():
    torch.cuda.synchronize()


def water_ball(n_waters, seed=0):
    rng = np.random.default_rng(seed)
    a = (30.0) ** (1 / 3)  # ~1 g/cm^3
    k = int(np.ceil((n_waters * 1.6) ** (1 / 3)))
    g = np.arange(-k, k + 1)
    pts = np.array(np.meshgrid(g, g, g)).reshape(3, -1).T * a
    pts = pts[np.argsort(np.linalg.norm(pts, axis=1))][:n_waters]
    ref = np.array([[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]])
    Z, P = [], []
    for c in pts:
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        P.append(ref @ q.T + c + rng.normal(scale=0.05, size=(3, 3)))
        Z += [8, 1, 1]
    return np.array(Z), np.concatenate(P)


# tiny short-range stand-in (float64), elements H,O only
old = torch.get_default_dtype()
torch.set_default_dtype(torch.float64)
zt = tools.AtomicNumberTable([1, 8])
sr = modules.MACE(
    r_max=3.0, num_bessel=6, num_polynomial_cutoff=5, max_ell=2,
    interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
    interaction_cls_first=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
    num_interactions=1, num_elements=2, hidden_irreps=o3.Irreps("8x0e + 8x1o"),
    MLP_irreps=o3.Irreps("8x0e"), gate=torch.nn.functional.silu,
    atomic_energies=np.array([1.0, 3.0]), avg_num_neighbors=4.0,
    atomic_numbers=zt.zs, correlation=2, radial_type="bessel").eval()
xdm = load_xdm_model(XDM_PATH, device=dev)
ref = mlxdm_2x_polarizability_reference(AtomicNumberTable(xdm.atomic_numbers.tolist()))
disp = XDMDispersionEnergy(alpha_free=ref["alpha_free"], v_free=ref["v_free"], cutoff=14.0)
model = MACEXDMDispersion(short_range_model=sr, xdm_model=xdm, dispersion_energy=disp).to(dev).eval()
torch.set_default_dtype(old)

# ---- instrumentation
T = {}
def timed(name, fn):
    def w(*a, **k):
        sync(); t = time.perf_counter()
        r = fn(*a, **k)
        sync(); T[name] = T.get(name, 0.0) + time.perf_counter() - t
        return r
    return w
def hook_pair(mod, name):
    st = {}
    def pre(m, i):
        sync(); st["t"] = time.perf_counter()
    def post(m, i, o):
        sync(); T[name] = T.get(name, 0.0) + time.perf_counter() - st["t"]
    mod.register_forward_pre_hook(pre)
    mod.register_forward_hook(post)
hook_pair(model.xdm_model, "fwd_xdm_network")
hook_pair(model.dispersion_energy, "fwd_dispersion_total")
hook_pair(model.short_range_model, "fwd_short_range_stub")
xd.get_neighborhood = timed("cpu_matscipy_search", xd.get_neighborhood)
n_pairs = {}
def wrap_pairs(attr, label):
    orig = getattr(XDMDispersionEnergy, attr)
    def w(self, *a, **k):
        sync(); t = time.perf_counter()
        r = orig(self, *a, **k)
        sync(); T["pair_finding_total"] = T.get("pair_finding_total", 0.0) + time.perf_counter() - t
        n_pairs["n"] = r[0].numel(); n_pairs["path"] = label
        return r
    setattr(XDMDispersionEnergy, attr, w)
if os.environ.get("PAIR_CACHE") == "1":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pair_cache_patch
    pair_cache_patch.install()
wrap_pairs("_sparse_pairs", "sparse(CPU matscipy)")
wrap_pairs("_dense_pairs", "dense(GPU nxn)")

res = []
for nw in SIZES:
    Z, P = water_ball(nw)
    try:
        sysm = gpu_md.System(model, Z, P, pbc=(False, False, False), device=dev, skin=2.0)
        for _ in range(2):
            sysm.energy_forces()
        T.clear(); sync(); t0 = time.perf_counter()
        for _ in range(NREP):
            sysm.energy_forces()
        sync(); tot = (time.perf_counter() - t0) / NREP
        row = {"n_atoms": len(Z), "total_ms": tot * 1e3, "n_pairs_14A": n_pairs["n"], "pair_path": n_pairs["path"]}
        for k, v in T.items():
            row[k + "_ms"] = v / NREP * 1e3
        row["backward_and_other_ms"] = row["total_ms"] - row["fwd_xdm_network_ms"] - row["fwd_dispersion_total_ms"] - row["fwd_short_range_stub_ms"]
        if os.environ.get("PAIR_CACHE") == "1":
            row["cache_rebuilds"] = pair_cache_patch.STATS["rebuilds"]; row["cache_rebuild_ms_each"] = pair_cache_patch.STATS["rebuild_ms"] / max(pair_cache_patch.STATS["rebuilds"], 1)
            pair_cache_patch.STATS.update(rebuilds=0, calls=0, rebuild_ms=0.0)
        res.append(row)
        print(json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        del sysm
        torch.cuda.empty_cache()
    except torch.OutOfMemoryError as e:
        print(f"n_waters={nw}: OOM ({str(e)[:80]})", flush=True)
        torch.cuda.empty_cache()
json.dump(res, open(sys.argv[4] if len(sys.argv) > 4 else "profile.json", "w"), indent=1)
