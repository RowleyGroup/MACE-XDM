###########################################################################################
# Build/load an AtomicXDMMACE from a run_train_xdm.py checkpoint (the
# <checkpoints_dir>/<name>_{latest,best}.pt files, saved as a plain dict of
# tensors/lists/metadata -- not a full pickled nn.Module), or from either
# that or a full saved model (<model_dir>/<name>.model) interchangeably.
###########################################################################################

from typing import Dict, Union

import torch
from e3nn import o3

from mace.modules.wrapper_ops import CuEquivarianceConfig, OEQConfig
from mace.tools import AtomicNumberTable, safe_jit_load_map_location

from . import gate_dict, interaction_classes
from .models import AtomicXDMMACE


def build_atomic_xdm_mace_from_args(
    args: Dict,
    z_table: AtomicNumberTable,
    avg_num_neighbors: float,
    element_means,
    element_stds,
) -> AtomicXDMMACE:
    """Construct an AtomicXDMMACE from the same hyperparameter dict
    run_train_xdm.py's argparse produces (``vars(args)``/checkpoint's
    ``"args"`` entry), so training and checkpoint-loading always build an
    identical architecture from a single source of truth.

    ``args["enable_cueq"]``/``args["enable_oeq"]`` (both default False, for
    checkpoints saved before these options existed) build the model with
    cuequivariance- or openequivariance-accelerated tensor-product kernels
    instead of plain e3nn -- see --enable_cueq/--enable_oeq's help in
    run_train_xdm.py for what that implies for loading the result later.
    Mutually exclusive; run_train_xdm.py's main() resolves both-set to cueq
    before this is called, so here it's cueq-takes-priority as a fallback.
    """
    cueq_config = None
    oeq_config = None
    if args.get("enable_cueq"):
        cueq_config = CuEquivarianceConfig(
            enabled=True, layout="ir_mul", group="O3_e3nn", optimize_all=True,
            conv_fusion=True,
        )
    elif args.get("enable_oeq"):
        oeq_config = OEQConfig(enabled=True, optimize_all=True)
    return AtomicXDMMACE(
        r_max=args["r_max"],
        num_bessel=args["num_bessel"],
        num_polynomial_cutoff=args["num_polynomial_cutoff"],
        max_ell=args["max_ell"],
        interaction_cls=interaction_classes[args["interaction"]],
        interaction_cls_first=interaction_classes[args["interaction_first"]],
        num_interactions=args["num_interactions"],
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps(args["hidden_irreps"]),
        MLP_irreps=o3.Irreps(args["MLP_irreps"]),
        avg_num_neighbors=avg_num_neighbors,
        atomic_numbers=z_table.zs,
        correlation=args["correlation"],
        gate=gate_dict[args["gate"]],
        element_means=element_means,
        element_stds=element_stds,
        num_xdm_targets=len(args["target_keys"]),
        cueq_config=cueq_config,
        oeq_config=oeq_config,
    )


def load_atomic_xdm_mace_checkpoint(
    checkpoint: Union[str, Dict], device="cpu"
) -> AtomicXDMMACE:
    """Rebuild and load an AtomicXDMMACE from a run_train_xdm.py checkpoint
    dict (or a path to one). Reconstructs the architecture via
    ``build_atomic_xdm_mace_from_args`` then loads ``model_state_dict`` --
    this only ever loads plain tensors, so (unlike a full pickled model) it
    never touches e3nn's CodeGenMixin unpickling path and needs no CPU/GPU
    workaround.
    """
    if isinstance(checkpoint, str):
        checkpoint = torch.load(checkpoint, map_location=device, weights_only=False)

    z_table = AtomicNumberTable(checkpoint["z_table"])
    model = build_atomic_xdm_mace_from_args(
        args=checkpoint["args"],
        z_table=z_table,
        avg_num_neighbors=checkpoint["avg_num_neighbors"],
        element_means=checkpoint["element_mean"],
        element_stds=checkpoint["element_std"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_xdm_model(path: str, device="cpu") -> AtomicXDMMACE:
    """Load a trained AtomicXDMMACE from either a full saved model
    (``torch.save(model, ...)``, e.g. ``<model_dir>/<name>.model``) or a
    training checkpoint (``<checkpoints_dir>/<name>_{latest,best}.pt``) --
    whichever the given path actually contains.
    """
    with safe_jit_load_map_location(device):
        obj = torch.load(path, map_location=device, weights_only=False)

    if isinstance(obj, dict) and "model_state_dict" in obj:
        return load_atomic_xdm_mace_checkpoint(obj, device=device)

    obj.eval()
    return obj.to(device)
