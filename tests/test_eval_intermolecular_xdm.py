import json
import sys

import h5py
import numpy as np
import torch
from e3nn import o3

from mace.cli.eval_intermolecular_xdm import discover_triplets, main
from mace.modules import (
    MACE,
    AtomicXDMMACE,
    gate_dict,
    interaction_classes,
)
from mace.tools import AtomicNumberTable

torch.set_default_dtype(torch.float64)

SYMBOL_TO_Z = {"H": 1, "O": 8}


def _build_xdm_model(z_table, num_xdm_targets=4):
    n_elements = len(z_table)
    rng = np.random.RandomState(0)
    return AtomicXDMMACE(
        r_max=5.0, num_bessel=8, num_polynomial_cutoff=5, max_ell=2,
        interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=interaction_classes["RealAgnosticInteractionBlock"],
        num_interactions=2, num_elements=n_elements,
        hidden_irreps=o3.Irreps("16x0e + 16x1o"), MLP_irreps=o3.Irreps("16x0e"),
        avg_num_neighbors=5.0, atomic_numbers=z_table.zs, correlation=2,
        gate=gate_dict["silu"],
        element_means=rng.rand(n_elements, num_xdm_targets) + 1.0,
        element_stds=rng.rand(n_elements, num_xdm_targets) * 0.1 + 0.1,
        num_xdm_targets=num_xdm_targets,
    )


def _build_short_range_model(z_table, atomic_energies):
    return MACE(
        r_max=5.0, num_bessel=8, num_polynomial_cutoff=5, max_ell=2,
        interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=interaction_classes["RealAgnosticInteractionBlock"],
        num_interactions=2, num_elements=len(z_table),
        hidden_irreps=o3.Irreps("16x0e + 16x1o"), MLP_irreps=o3.Irreps("16x0e"),
        atomic_energies=np.array(atomic_energies), avg_num_neighbors=5.0,
        atomic_numbers=z_table.zs, correlation=2, gate=torch.nn.functional.silu,
    )


def _write_dataset(path, n_complexes=4, seed=0):
    rng = np.random.RandomState(seed)
    with h5py.File(path, "w") as f:
        wrapper = f.create_group("deshaw_test_xdm")
        for i in range(n_complexes):
            complex_pos = rng.randn(6, 3) * 2.0 + np.array(
                [[0, 0, 0], [0, 0.76, 0.59], [0, -0.76, 0.59],
                 [3, 0, 0], [3, 0.76, 0.59], [3, -0.76, 0.59]]
            )
            complex_z = np.array([8, 1, 1, 8, 1, 1], dtype=np.uint8)
            frag_z = np.array([8, 1, 1], dtype=np.uint8)

            e_complex, e_f1, e_f2 = -152.0 + rng.randn() * 0.01, -76.0 + rng.randn() * 0.01, -76.0 + rng.randn() * 0.01
            exdm_complex, exdm_f1, exdm_f2 = -0.005 + rng.randn() * 1e-4, -0.001 + rng.randn() * 1e-5, -0.001 + rng.randn() * 1e-5

            # Deliberately mismatched prefixes between complex and fragment groups,
            # mirroring parse_tmpdir.py's own "deshaw370k_"/"deshaw15k_...frag_N" naming.
            gc = wrapper.create_group(f"complexprefix_{i}")
            gc.create_dataset("atomic_numbers", data=complex_z)
            gc.create_dataset("coordinates", data=complex_pos[None].astype(np.float32))
            gc.create_dataset("energies", data=np.array([e_complex]))
            gc.create_dataset("e_xdm", data=np.array([exdm_complex]))

            g1 = wrapper.create_group(f"fragprefix_{i}.frag_1")
            g1.create_dataset("atomic_numbers", data=frag_z)
            g1.create_dataset("coordinates", data=complex_pos[None, :3].astype(np.float32))
            g1.create_dataset("energies", data=np.array([e_f1]))
            g1.create_dataset("e_xdm", data=np.array([exdm_f1]))

            g2 = wrapper.create_group(f"fragprefix_{i}.frag_2")
            g2.create_dataset("atomic_numbers", data=frag_z)
            g2.create_dataset("coordinates", data=complex_pos[None, 3:].astype(np.float32))
            g2.create_dataset("energies", data=np.array([e_f2]))
            g2.create_dataset("e_xdm", data=np.array([exdm_f2]))


def test_discover_triplets_matches_by_index_despite_prefix_mismatch(tmp_path):
    path = tmp_path / "data.h5"
    _write_dataset(path, n_complexes=3)

    triplets = discover_triplets([str(path)])
    assert len(triplets) == 3
    for roles in triplets:
        assert set(roles.keys()) == {"complex", "frag1", "frag2"}
        complex_leaf = roles["complex"][1]
        frag1_leaf = roles["frag1"][1]
        # same trailing index, different prefix text
        assert complex_leaf.rsplit("_", 1)[-1] == frag1_leaf.split(".frag_1")[0].rsplit("_", 1)[-1]


def test_discover_triplets_skips_incomplete_sets(tmp_path):
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group("wrapper")
        gc = g.create_group("complex_0")
        gc.create_dataset("atomic_numbers", data=np.array([1], dtype=np.uint8))
        gc.create_dataset("coordinates", data=np.zeros((1, 1, 3), dtype=np.float32))
        gc.create_dataset("energies", data=np.array([0.0]))
        gc.create_dataset("e_xdm", data=np.array([0.0]))
        g1 = g.create_group("frag_0.frag_1")
        g1.create_dataset("atomic_numbers", data=np.array([1], dtype=np.uint8))
        g1.create_dataset("coordinates", data=np.zeros((1, 1, 3), dtype=np.float32))
        g1.create_dataset("energies", data=np.array([0.0]))
        g1.create_dataset("e_xdm", data=np.array([0.0]))
        # frag_0.frag_2 deliberately missing

    try:
        triplets = discover_triplets([str(path)])
    except ValueError:
        triplets = []
    assert len(triplets) == 0


def test_main_cli_writes_intermolecular_report(tmp_path, monkeypatch):
    xdm_z_table = AtomicNumberTable([1, 8])
    xdm_model = _build_xdm_model(xdm_z_table)
    xdm_path = tmp_path / "xdm.model"
    torch.save(xdm_model, xdm_path)

    sr_z_table = AtomicNumberTable([1, 8])
    sr_model = _build_short_range_model(sr_z_table, atomic_energies=[-0.5, -75.0])
    sr_path = tmp_path / "short_range.model"
    torch.save(sr_model, sr_path)

    data_path = tmp_path / "data.h5"
    _write_dataset(data_path, n_complexes=4)

    output_dir = tmp_path / "out"
    argv = [
        "mace_eval_intermolecular_xdm",
        "--short_range_model", str(sr_path),
        "--xdm_model", str(xdm_path),
        "--data_files", str(data_path),
        "--output_dir", str(output_dir),
        "--no_plots",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()

    report = json.loads((output_dir / "intermolecular_report.json").read_text())
    assert set(report.keys()) == {"PBE0", "XDM", "PBE0+XDM"}
    for key in report:
        assert report[key]["n"] == 4
        for metric in ("mae_kcal_mol", "rmse_kcal_mol", "r2", "pearson_r"):
            assert metric in report[key]

    csv_lines = (output_dir / "intermolecular_predictions.csv").read_text().strip().splitlines()
    assert len(csv_lines) == 5  # header + 4 complexes
