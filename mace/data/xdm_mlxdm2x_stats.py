###########################################################################################
# Fixed per-element reference mean/std for XDM moments (M1, M2, M3) and effective
# volume (Veff), matching the values used by RowleyGroup/MLXDM's ANI-2x-based
# dispersion model (torchanipbe0/resources/dispersion_2x/{m1,m2,m3,v}/best.param).
#
# MLXDM's "Shifter" module computes physical = b0 + b1 * standardized, with b0
# the per-element mean and b1 the per-element std, for elements in the fixed
# order used by its ANI-2x species vector: H, C, N, O, S, F, Cl. This exactly
# matches AtomicElementReferenceBlock's `mean[Z] + std[Z] * standardized`, so
# these constants can be dropped in directly in place of statistics computed
# from a given training set.
###########################################################################################

from typing import Dict, Sequence

import numpy as np
from ase.data import atomic_numbers as ase_atomic_numbers

from mace.tools import AtomicNumberTable

CANONICAL_PROPERTIES = ("M1", "M2", "M3", "Veff")

MLXDM_2X_MEAN: Dict[str, Dict[str, float]] = {
    "H": {"M1": 1.549966421, "M2": 12.66061817, "M3": 216.4468257, "Veff": 6.12519483},
    "C": {"M1": 4.29554016, "M2": 55.21207036, "M3": 1028.215929, "Veff": 31.91292422},
    "N": {"M1": 4.587581115, "M2": 44.51803025, "M3": 586.4837632, "Veff": 26.08595383},
    "O": {"M1": 4.834865433, "M2": 37.29377243, "M3": 382.8379115, "Veff": 21.86281485},
    "S": {"M1": 10.17757022, "M2": 157.2811396, "M3": 2785.902118, "Veff": 72.96078457},
    "F": {"M1": 4.645134794, "M2": 29.14714591, "M3": 242.2360933, "Veff": 17.81828315},
    "Cl": {"M1": 9.79339843, "M2": 129.2453106, "M3": 1878.21107, "Veff": 63.43526165},
}

MLXDM_2X_STD: Dict[str, Dict[str, float]] = {
    "H": {"M1": 0.375665429, "M2": 3.984412001, "M3": 98.25085767, "Veff": 1.259947858},
    "C": {"M1": 1.904449149, "M2": 13.98791997, "M3": 411.7789378, "Veff": 3.087075378},
    "N": {"M1": 1.672409685, "M2": 19.48145538, "M3": 213.5113146, "Veff": 2.585836319},
    "O": {"M1": 0.925116758, "M2": 9.705241715, "M3": 137.1489884, "Veff": 2.137157997},
    "S": {"M1": 3.867515285, "M2": 55.27651422, "M3": 974.0683876, "Veff": 8.639045398},
    "F": {"M1": 0.40839295, "M2": 3.464255641, "M3": 38.66531072, "Veff": 1.025135058},
    "Cl": {"M1": 0.850317483, "M2": 14.66876091, "M3": 279.588133, "Veff": 3.083419526},
}

# Free-atom reference static dipole polarizability (alpha_free) and free-atom
# volume (V_free), both in atomic units, transcribed from the constructor
# arguments of RowleyGroup/MLXDM's torchanipbe0/models.py::XDM_2x_CC (the
# DispersionLayer for this same H/C/N/O/S/F/Cl ANI-2x element set). Atom-in-
# molecule polarizability is alpha_A = Veff_A * alpha_free_A / V_free_A.
MLXDM_2X_ALPHA_FREE: Dict[str, float] = {
    "H": 4.4997895,
    "C": 11.87706887,
    "N": 7.42316804,
    "O": 5.41216434,
    "S": 19.57017029,
    "F": 3.75882236,
    "Cl": 14.71136939,
}

MLXDM_2X_V_FREE: Dict[str, float] = {
    "H": 8.2794385587230224,
    "C": 35.403450375407488,
    "N": 26.774856262986901,
    "O": 22.577665436425793,
    "S": 75.344227406670839,
    "F": 18.604506038051770,
    "Cl": 65.219744182377752,
}

# Becke-Johnson damping parameters (a1, a2) fitted for this PBE0-XDM/ANI-2x
# combination, and the dispersion-sum cutoff radius (Angstrom), both from the
# same XDM_2x_CC constructor call.
MLXDM_2X_BJ_A1 = 0.4186
MLXDM_2X_BJ_A2 = 2.6791
MLXDM_2X_DISPERSION_CUTOFF = 14.0

MLXDM_2X_ATOMIC_NUMBERS = sorted(ase_atomic_numbers[s] for s in MLXDM_2X_MEAN)

_Z_TO_SYMBOL = {ase_atomic_numbers[s]: s for s in MLXDM_2X_MEAN}


def default_mlxdm_2x_atomic_number_table() -> AtomicNumberTable:
    """The 7 elements (H, C, N, O, S, F, Cl) MLXDM's ANI-2x dispersion model
    was trained on, sorted by atomic number."""
    return AtomicNumberTable(MLXDM_2X_ATOMIC_NUMBERS)


def mlxdm_2x_reference_stats(
    z_table: AtomicNumberTable,
    target_keys: Sequence[str] = CANONICAL_PROPERTIES,
) -> Dict[str, np.ndarray]:
    """Build ``mean``/``std`` arrays of shape ``[len(z_table), 4]`` from MLXDM's
    fixed ANI-2x reference statistics, ready to feed into
    ``AtomicElementReferenceBlock``/``AtomicXDMMACE``.

    ``target_keys`` only needs to have length 4 and is interpreted positionally
    as (M1, M2, M3, Veff) regardless of the actual HDF5 dataset key names used
    for those four properties (``--target_keys`` just renames the columns read
    from disk; the underlying physical quantities are always this same four).
    """
    if len(target_keys) != len(CANONICAL_PROPERTIES):
        raise ValueError(
            f"MLXDM reference statistics cover exactly the 4 properties "
            f"{CANONICAL_PROPERTIES} (M1, M2, M3, Veff); got {len(target_keys)} "
            f"target keys: {target_keys}."
        )

    missing = [z for z in z_table.zs if z not in _Z_TO_SYMBOL]
    if missing:
        raise ValueError(
            f"MLXDM ANI-2x reference statistics are only available for atomic "
            f"numbers {MLXDM_2X_ATOMIC_NUMBERS} (H, C, N, O, S, F, Cl); no "
            f"reference values for atomic numbers {missing}. Pass "
            f"--element_stats dataset to compute mean/std from your own "
            f"training data instead, or restrict --atomic_numbers to the "
            f"supported elements."
        )

    mean = np.zeros((len(z_table), len(CANONICAL_PROPERTIES)))
    std = np.ones((len(z_table), len(CANONICAL_PROPERTIES)))
    for idx, z in enumerate(z_table.zs):
        symbol = _Z_TO_SYMBOL[z]
        for j, key in enumerate(CANONICAL_PROPERTIES):
            mean[idx, j] = MLXDM_2X_MEAN[symbol][key]
            std[idx, j] = MLXDM_2X_STD[symbol][key]
    return {"mean": mean, "std": std}


def mlxdm_2x_polarizability_reference(z_table: AtomicNumberTable) -> Dict[str, np.ndarray]:
    """Build ``alpha_free``/``v_free`` arrays of shape ``[len(z_table)]`` from
    MLXDM's fixed free-atom reference polarizability/volume, ready to feed
    into an XDM dispersion-energy module (atom-in-molecule polarizability is
    then ``alpha_A = Veff_A * alpha_free[Z] / v_free[Z]``).
    """
    missing = [z for z in z_table.zs if z not in _Z_TO_SYMBOL]
    if missing:
        raise ValueError(
            f"MLXDM ANI-2x free-atom polarizability/volume reference values are "
            f"only available for atomic numbers {MLXDM_2X_ATOMIC_NUMBERS} "
            f"(H, C, N, O, S, F, Cl); no reference values for atomic numbers "
            f"{missing}."
        )
    alpha_free = np.array(
        [MLXDM_2X_ALPHA_FREE[_Z_TO_SYMBOL[z]] for z in z_table.zs]
    )
    v_free = np.array([MLXDM_2X_V_FREE[_Z_TO_SYMBOL[z]] for z in z_table.zs])
    return {"alpha_free": alpha_free, "v_free": v_free}
