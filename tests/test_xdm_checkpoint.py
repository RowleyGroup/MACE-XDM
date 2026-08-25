import numpy as np
import torch
from e3nn import o3

from mace.modules import (
    AtomicXDMMACE,
    build_atomic_xdm_mace_from_args,
    gate_dict,
    interaction_classes,
    load_atomic_xdm_mace_checkpoint,
    load_xdm_model,
)
from mace.tools import AtomicNumberTable, safe_jit_load_map_location
from mace.tools.torch_geometric.batch import Batch
from mace.data.xdm import build_xdm_atomic_data

torch.set_default_dtype(torch.float64)


def _make_args_dict():
    return {
        "r_max": 5.0,
        "num_bessel": 8,
        "num_polynomial_cutoff": 5,
        "max_ell": 2,
        "interaction": "RealAgnosticResidualInteractionBlock",
        "interaction_first": "RealAgnosticInteractionBlock",
        "num_interactions": 2,
        "hidden_irreps": "16x0e + 16x1o",
        "MLP_irreps": "16x0e",
        "correlation": 2,
        "gate": "silu",
        "target_keys": ["M1", "M2", "M3", "Veff"],
    }


def _make_checkpoint_dict(model, args, z_table, avg_num_neighbors, element_means, element_stds):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "epoch": 3,
        "best_valid_loss": 0.1,
        "args": args,
        "z_table": z_table.zs,
        "element_mean": np.asarray(element_means).tolist(),
        "element_std": np.asarray(element_stds).tolist(),
        "avg_num_neighbors": avg_num_neighbors,
    }


def _sample_output(model, z_table):
    atomic_numbers = np.array([8, 1, 1])
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.76, 0.59], [0.0, -0.76, 0.59]])
    targets = np.zeros((3, 4))
    graph = build_xdm_atomic_data(atomic_numbers, positions, targets, z_table, cutoff=5.0)
    batch = Batch.from_data_list([graph])
    with torch.no_grad():
        return model(batch.to_dict())["xdm_atomic"]


def test_build_atomic_xdm_mace_from_args_matches_direct_construction():
    z_table = AtomicNumberTable([1, 6, 8])
    args = _make_args_dict()
    rng = np.random.RandomState(0)
    element_means = rng.rand(len(z_table), 4) + 1.0
    element_stds = rng.rand(len(z_table), 4) * 0.1 + 0.1

    via_helper = build_atomic_xdm_mace_from_args(
        args=args,
        z_table=z_table,
        avg_num_neighbors=5.0,
        element_means=element_means,
        element_stds=element_stds,
    )
    direct = AtomicXDMMACE(
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
        avg_num_neighbors=5.0,
        atomic_numbers=z_table.zs,
        correlation=2,
        gate=gate_dict["silu"],
        element_means=element_means,
        element_stds=element_stds,
        num_xdm_targets=4,
    )
    assert via_helper.num_xdm_targets == direct.num_xdm_targets
    assert via_helper.atomic_numbers.tolist() == direct.atomic_numbers.tolist()


def test_load_atomic_xdm_mace_checkpoint_roundtrip(tmp_path):
    z_table = AtomicNumberTable([1, 6, 8])
    args = _make_args_dict()
    rng = np.random.RandomState(1)
    element_means = rng.rand(len(z_table), 4) + 1.0
    element_stds = rng.rand(len(z_table), 4) * 0.1 + 0.1

    model = build_atomic_xdm_mace_from_args(
        args=args, z_table=z_table, avg_num_neighbors=4.0,
        element_means=element_means, element_stds=element_stds,
    )
    model.eval()
    original_output = _sample_output(model, z_table)

    checkpoint = _make_checkpoint_dict(model, args, z_table, 4.0, element_means, element_stds)
    path = tmp_path / "xdm_test_best.pt"
    torch.save(checkpoint, path)

    reloaded = load_atomic_xdm_mace_checkpoint(str(path), device="cpu")
    reloaded_output = _sample_output(reloaded, z_table)

    assert torch.allclose(original_output, reloaded_output, atol=1e-12)


def test_load_xdm_model_detects_checkpoint_vs_full_model(tmp_path):
    z_table = AtomicNumberTable([1, 6, 8])
    args = _make_args_dict()
    rng = np.random.RandomState(2)
    element_means = rng.rand(len(z_table), 4) + 1.0
    element_stds = rng.rand(len(z_table), 4) * 0.1 + 0.1

    model = build_atomic_xdm_mace_from_args(
        args=args, z_table=z_table, avg_num_neighbors=4.0,
        element_means=element_means, element_stds=element_stds,
    )
    model.eval()
    reference_output = _sample_output(model, z_table)

    checkpoint = _make_checkpoint_dict(model, args, z_table, 4.0, element_means, element_stds)
    checkpoint_path = tmp_path / "xdm_best.pt"
    torch.save(checkpoint, checkpoint_path)

    model_path = tmp_path / "xdm.model"
    torch.save(model, model_path)

    from_checkpoint = load_xdm_model(str(checkpoint_path), device="cpu")
    from_model = load_xdm_model(str(model_path), device="cpu")

    assert torch.allclose(_sample_output(from_checkpoint, z_table), reference_output, atol=1e-12)
    assert torch.allclose(_sample_output(from_model, z_table), reference_output, atol=1e-12)


def test_safe_jit_load_map_location_restores_original():
    original = torch.jit.load
    with safe_jit_load_map_location("cpu"):
        assert torch.jit.load is not original
    assert torch.jit.load is original
