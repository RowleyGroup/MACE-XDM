from typing import Callable, Dict, Optional, Type

import torch

from .blocks import (
    AtomicElementReferenceBlock,
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    GeneralNonLinearBiasReadoutBlock,
    InteractionBlock,
    LinearDipolePolarReadoutBlock,
    LinearDipoleReadoutBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearBiasReadoutBlock,
    NonLinearDipolePolarReadoutBlock,
    NonLinearDipoleReadoutBlock,
    NonLinearReadoutBlock,
    PerElementLinearReadoutBlock,
    PerElementNonLinearReadoutBlock,
    RadialEmbeddingBlock,
    RealAgnosticAttResidualInteractionBlock,
    RealAgnosticDensityInteractionBlock,
    RealAgnosticDensityResidualInteractionBlock,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
    RealAgnosticResidualNonLinearInteractionBlock,
    ScaleShiftBlock,
)
from .extensions import PolarMACE
from .gate import GatedEquivariantBlock
from .loss import (
    DipolePolarLoss,
    DipoleSingleLoss,
    UniversalLoss,
    WeightedEnergyForcesDipoleLoss,
    WeightedEnergyForcesL1L2Loss,
    WeightedEnergyForcesLoss,
    WeightedEnergyForcesStressLoss,
    WeightedEnergyForcesVirialsLoss,
    WeightedForcesLoss,
    WeightedHuberEnergyForcesStressLoss,
)
from .models import (
    MACE,
    AtomicDielectricMACE,
    AtomicDipolesMACE,
    AtomicXDMMACE,
    EnergyDipolesMACE,
    ScaleShiftMACE,
)
from .radial import BesselBasis, GaussianBasis, PolynomialCutoff, ZBLBasis
from .symmetric_contraction import SymmetricContraction
from .utils import (
    compute_avg_num_neighbors,
    compute_dielectric_gradients,
    compute_fixed_charge_dipole,
    compute_fixed_charge_dipole_polar,
    compute_mean_rms_energy_forces,
    compute_mean_std_atomic_inter_energy,
    compute_rms_dipoles,
    compute_statistics,
)
from .xdm_combined import MACEXDMDispersion
from .xdm_dispersion import XDMDispersionEnergy

interaction_classes: Dict[str, Type[InteractionBlock]] = {
    "RealAgnosticResidualInteractionBlock": RealAgnosticResidualInteractionBlock,
    "RealAgnosticAttResidualInteractionBlock": RealAgnosticAttResidualInteractionBlock,
    "RealAgnosticInteractionBlock": RealAgnosticInteractionBlock,
    "RealAgnosticDensityInteractionBlock": RealAgnosticDensityInteractionBlock,
    "RealAgnosticDensityResidualInteractionBlock": RealAgnosticDensityResidualInteractionBlock,
    "RealAgnosticResidualNonLinearInteractionBlock": RealAgnosticResidualNonLinearInteractionBlock,
}

readout_classes: Dict[str, Type[LinearReadoutBlock]] = {
    "LinearReadoutBlock": LinearReadoutBlock,
    "LinearDipoleReadoutBlock": LinearDipoleReadoutBlock,
    "NonLinearDipoleReadoutBlock": NonLinearDipoleReadoutBlock,
    "NonLinearReadoutBlock": NonLinearReadoutBlock,
    "NonLinearBiasReadoutBlock": NonLinearBiasReadoutBlock,
    "GeneralNonLinearBiasReadoutBlock": GeneralNonLinearBiasReadoutBlock,
}

scaling_classes: Dict[str, Callable] = {
    "std_scaling": compute_mean_std_atomic_inter_energy,
    "rms_forces_scaling": compute_mean_rms_energy_forces,
    "rms_dipoles_scaling": compute_rms_dipoles,
}

gate_dict: Dict[str, Optional[Callable]] = {
    "abs": torch.abs,
    "tanh": torch.tanh,
    "silu": torch.nn.functional.silu,
    "None": None,
}

# Imported here (after gate_dict/interaction_classes above) rather than in the
# main import block, since xdm_checkpoint needs those two names from this
# same __init__ module -- importing it earlier would be a circular import.
from .xdm_checkpoint import (  # noqa: E402
    build_atomic_xdm_mace_from_args,
    load_atomic_xdm_mace_checkpoint,
    load_xdm_model,
)

__all__ = [
    "AtomicEnergiesBlock",
    "AtomicElementReferenceBlock",
    "RadialEmbeddingBlock",
    "ZBLBasis",
    "LinearNodeEmbeddingBlock",
    "LinearReadoutBlock",
    "EquivariantProductBasisBlock",
    "ScaleShiftBlock",
    "LinearDipoleReadoutBlock",
    "LinearDipolePolarReadoutBlock",
    "NonLinearDipoleReadoutBlock",
    "NonLinearDipolePolarReadoutBlock",
    "InteractionBlock",
    "NonLinearReadoutBlock",
    "PerElementLinearReadoutBlock",
    "PerElementNonLinearReadoutBlock",
    "PolynomialCutoff",
    "BesselBasis",
    "GaussianBasis",
    "MACE",
    "ScaleShiftMACE",
    "AtomicDipolesMACE",
    "AtomicDielectricMACE",
    "AtomicXDMMACE",
    "XDMDispersionEnergy",
    "MACEXDMDispersion",
    "build_atomic_xdm_mace_from_args",
    "load_atomic_xdm_mace_checkpoint",
    "load_xdm_model",
    "EnergyDipolesMACE",
    "PolarMACE",
    "WeightedEnergyForcesLoss",
    "WeightedForcesLoss",
    "WeightedEnergyForcesVirialsLoss",
    "WeightedEnergyForcesStressLoss",
    "DipoleSingleLoss",
    "WeightedEnergyForcesDipoleLoss",
    "WeightedHuberEnergyForcesStressLoss",
    "UniversalLoss",
    "WeightedEnergyForcesL1L2Loss",
    "SymmetricContraction",
    "interaction_classes",
    "compute_mean_std_atomic_inter_energy",
    "compute_avg_num_neighbors",
    "compute_statistics",
    "compute_fixed_charge_dipole",
    "compute_fixed_charge_dipole_polar",
    "compute_dielectric_gradients",
]
