###########################################################################################
# Tools for torch
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
from contextlib import contextmanager
from typing import Dict, Union

import numpy as np
import torch
from e3nn.io import CartesianTensor

TensorDict = Dict[str, torch.Tensor]


def to_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Generates one-hot encoding with <num_classes> classes from <indices>
    :param indices: (N x 1) tensor
    :param num_classes: number of classes
    :param device: torch device
    :return: (N x num_classes) tensor
    """
    shape = indices.shape[:-1] + (num_classes,)
    oh = torch.zeros(shape, device=indices.device).view(shape)

    # scatter_ is the in-place version of scatter
    oh.scatter_(dim=-1, index=indices, value=1)

    return oh.view(*shape)


def count_parameters(module: torch.nn.Module) -> int:
    return int(sum(np.prod(p.shape) for p in module.parameters()))


def tensor_dict_to_device(td: TensorDict, device: torch.device) -> TensorDict:
    return {k: v.to(device) if v is not None else None for k, v in td.items()}


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.cpu().detach().numpy()


def init_device(device_str: str) -> torch.device:
    if "cuda" in device_str:
        assert torch.cuda.is_available(), "No CUDA device available!"
        if ":" in device_str:
            # Check if the desired device is available
            assert int(device_str.split(":")[-1]) < torch.cuda.device_count()
        logging.info(
            f"CUDA version: {torch.version.cuda}, CUDA device: {torch.cuda.current_device()}"
        )
        torch.cuda.init()
        return torch.device(device_str)
    if device_str == "mps":
        assert torch.backends.mps.is_available(), "No MPS backend is available!"
        logging.info("Using MPS GPU acceleration")
        return torch.device("mps")
    if device_str == "xpu":
        torch.xpu.is_available()
        devices = torch.xpu.device_count()
        is_available = devices > 0
        assert is_available, logging.info("No XPU backend is available")
        torch.xpu.memory_stats()
        logging.info("Using XPU GPU acceleration")
        return torch.device("xpu")

    logging.info("Using CPU")
    return torch.device("cpu")


dtype_dict = {"float32": torch.float32, "float64": torch.float64}


def set_default_dtype(dtype: str) -> None:
    torch.set_default_dtype(dtype_dict[dtype])


@contextmanager
def safe_jit_load_map_location(map_location):
    """Work around e3nn<=0.4.x: ``CodeGenMixin.__setstate__`` (used to
    restore the compiled tensor-product/Linear code-gen submodules of any
    e3nn-based model, e.g. MACE) calls ``torch.jit.load(buffer)`` internally
    without forwarding the outer ``torch.load``'s ``map_location``. Without
    this, unpickling a model that was even partly built/saved on a GPU
    machine raises a "Found no NVIDIA driver" RuntimeError when loading it on
    a CPU-only machine, even though ``torch.load(..., map_location="cpu")``
    was used. This context manager temporarily patches ``torch.jit.load`` to
    force the given ``map_location`` regardless of what (if anything) is
    passed to it internally; harmless to apply even when loading something
    that never calls ``torch.jit.load`` at all (e.g. a plain checkpoint dict).
    """
    forced_map_location = map_location
    original_jit_load = torch.jit.load

    def _patched_jit_load(f, map_location=None, _extra_files=None, _restore_shapes=False):  # noqa: ARG001
        return original_jit_load(
            f,
            map_location=forced_map_location,
            _extra_files=_extra_files or {},
            _restore_shapes=_restore_shapes,
        )

    torch.jit.load = _patched_jit_load
    try:
        yield
    finally:
        torch.jit.load = original_jit_load


def load_full_model(path: str, device="cpu") -> torch.nn.Module:
    """Load a full ``torch.save(model, ...)``'d module (e.g. a MACE
    ``<name>.model`` file), safe against the CPU-load issue above."""
    with safe_jit_load_map_location(device):
        model = torch.load(path, map_location=device, weights_only=False)
    model.eval()
    return model.to(device)


def get_change_of_basis() -> torch.Tensor:
    return CartesianTensor("ij=ji").reduced_tensor_products().change_of_basis


def spherical_to_cartesian(t: torch.Tensor, change_of_basis: torch.Tensor):
    # Optionally handle device mismatch
    if change_of_basis.device != t.device:
        change_of_basis = change_of_basis.to(t.device)
    return torch.einsum("ijk,...i->...jk", change_of_basis, t)


def cartesian_to_spherical(t: torch.Tensor):
    """
    Convert cartesian notation to spherical notation
    """
    stress_cart_tensor = CartesianTensor("ij=ji")
    stress_rtp = stress_cart_tensor.reduced_tensor_products()
    return stress_cart_tensor.to_cartesian(t, rtp=stress_rtp)


def voigt_to_matrix(t: torch.Tensor):
    """
    Convert voigt notation to matrix notation
    :param t: (6,) tensor or (3, 3) tensor or (9,) tensor
    :return: (3, 3) tensor
    """
    if t.shape == (3, 3):
        return t
    if t.shape == (6,):
        return torch.tensor(
            [
                [t[0], t[5], t[4]],
                [t[5], t[1], t[3]],
                [t[4], t[3], t[2]],
            ],
            dtype=t.dtype,
        )
    if t.shape == (9,):
        return t.view(3, 3)

    raise ValueError(
        f"Stress tensor must be of shape (6,) or (3, 3), or (9,) but has shape {t.shape}"
    )


def init_wandb(project: str, entity: str, name: str, config: dict, directory: str):
    import wandb

    wandb.init(
        project=project,
        entity=entity,
        name=name,
        config=config,
        dir=directory,
        resume="allow",
    )


@contextmanager
def default_dtype(dtype: Union[torch.dtype, str]):
    """Context manager for configuring the default_dtype used by torch

    Args:
        dtype (torch.dtype|str): the default dtype to use within this context manager
    """
    init = torch.get_default_dtype()
    if isinstance(dtype, str):
        set_default_dtype(dtype)
    else:
        torch.set_default_dtype(dtype)

    yield

    torch.set_default_dtype(init)
