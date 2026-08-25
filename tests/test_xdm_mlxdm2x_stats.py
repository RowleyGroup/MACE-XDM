import numpy as np
import pytest

from mace.data import default_mlxdm_2x_atomic_number_table, mlxdm_2x_reference_stats
from mace.data.xdm_mlxdm2x_stats import MLXDM_2X_ATOMIC_NUMBERS
from mace.tools import AtomicNumberTable


def test_mlxdm_2x_atomic_numbers():
    # H, C, N, O, F, S, Cl sorted by atomic number
    assert MLXDM_2X_ATOMIC_NUMBERS == [1, 6, 7, 8, 9, 16, 17]


def test_default_mlxdm_2x_atomic_number_table():
    z_table = default_mlxdm_2x_atomic_number_table()
    assert z_table.zs == [1, 6, 7, 8, 9, 16, 17]


def test_mlxdm_2x_reference_stats_values():
    z_table = default_mlxdm_2x_atomic_number_table()
    stats = mlxdm_2x_reference_stats(z_table)
    assert stats["mean"].shape == (7, 4)
    assert stats["std"].shape == (7, 4)

    # spot-check H (index 0) and Cl (index 6), columns are (M1, M2, M3, Veff)
    h_idx = z_table.z_to_index(1)
    cl_idx = z_table.z_to_index(17)
    np.testing.assert_allclose(
        stats["mean"][h_idx], [1.549966421, 12.66061817, 216.4468257, 6.12519483]
    )
    np.testing.assert_allclose(
        stats["std"][h_idx], [0.375665429, 3.984412001, 98.25085767, 1.259947858]
    )
    np.testing.assert_allclose(
        stats["mean"][cl_idx], [9.79339843, 129.2453106, 1878.21107, 63.43526165]
    )
    np.testing.assert_allclose(
        stats["std"][cl_idx], [0.850317483, 14.66876091, 279.588133, 3.083419526]
    )


def test_mlxdm_2x_reference_stats_reindexes_by_z_table_order():
    # Same elements, different order in the table; values should follow Z, not position.
    forward = mlxdm_2x_reference_stats(AtomicNumberTable([1, 6, 8]))
    reversed_table = mlxdm_2x_reference_stats(AtomicNumberTable([8, 6, 1]))
    np.testing.assert_allclose(forward["mean"][0], reversed_table["mean"][2])  # H
    np.testing.assert_allclose(forward["mean"][2], reversed_table["mean"][0])  # O


def test_mlxdm_2x_reference_stats_subset():
    z_table = AtomicNumberTable([1, 6, 8])
    stats = mlxdm_2x_reference_stats(z_table)
    assert stats["mean"].shape == (3, 4)


def test_mlxdm_2x_reference_stats_unsupported_element_raises():
    z_table = AtomicNumberTable([1, 6, 35])  # Br unsupported
    with pytest.raises(ValueError, match="35"):
        mlxdm_2x_reference_stats(z_table)
