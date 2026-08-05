import json
import sys

import h5py
import numpy as np
import torch

from mace.cli.test_xdm_dispersion import collect_dispersion_energies, main
from mace.data import mlxdm_2x_polarizability_reference
from mace.modules import build_atomic_xdm_mace_from_args
from mace.modules.xdm_dispersion import XDMDispersionEnergy
from mace.tools import AtomicNumberTable
from mace.tools.torch_geometric.dataloader import DataLoader
from mace.tools.torch_geometric.batch import Batch
from mace.data.xdm import build_xdm_atomic_data

torch.set_default_dtype(torch.float64)

SYMBOL_TO_Z = {"H": 1, "C": 6, "N": 7, "O": 8}


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
        "epoch": 1,
        "best_valid_loss": 0.1,
        "args": args,
        "z_table": z_table.zs,
        "element_mean": np.asarray(element_means).tolist(),
        "element_std": np.asarray(element_stds).tolist(),
        "avg_num_neighbors": avg_num_neighbors,
    }


def _write_dataset(path, z_table, n_molecules=6, seed=0):
    rng = np.random.RandomState(seed)
    with h5py.File(path, "w") as f:
        for i in range(n_molecules):
            species = ["O", "H", "H"] if i % 2 == 0 else ["C", "H", "H", "H", "H"]
            grp = f.create_group(f"mol_{i}")
            n_atoms = len(species)
            grp.create_dataset(
                "atomic_numbers",
                data=np.array([SYMBOL_TO_Z[s] for s in species], dtype=np.uint8),
            )
            grp.create_dataset(
                "coordinates", data=(rng.randn(1, n_atoms, 3) * 2.0).astype(np.float32)
            )
            for key, scale in (("M1", 3.0), ("M2", 20.0), ("M3", 300.0), ("Veff", 15.0)):
                grp.create_dataset(key, data=(rng.rand(1, n_atoms) + 1.0) * scale)


def test_collect_dispersion_energies_components_are_additive():
    z_table = AtomicNumberTable([1, 6, 8])
    args = _make_args_dict()
    rng = np.random.RandomState(0)
    element_means = rng.rand(len(z_table), 4) + 1.0
    element_stds = rng.rand(len(z_table), 4) * 0.1 + 0.1

    model = build_atomic_xdm_mace_from_args(
        args=args, z_table=z_table, avg_num_neighbors=4.0,
        element_means=element_means, element_stds=element_stds,
    )
    model.eval()

    polar_ref = mlxdm_2x_polarizability_reference(z_table)
    dispersion_energy = XDMDispersionEnergy(
        alpha_free=polar_ref["alpha_free"], v_free=polar_ref["v_free"]
    )

    graphs = []
    for atomic_numbers, positions, targets in (
        (
            np.array([8, 1, 1]),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.76, 0.59], [0.0, -0.76, 0.59]]),
            (rng.rand(3, 4) + 1.0) * np.array([3.0, 20.0, 300.0, 15.0]),
        ),
        (
            np.array([6, 1, 1, 1, 1]),
            rng.randn(5, 3) * 1.5,
            (rng.rand(5, 4) + 1.0) * np.array([3.0, 20.0, 300.0, 15.0]),
        ),
    ):
        graphs.append(build_xdm_atomic_data(atomic_numbers, positions, targets, z_table, cutoff=5.0))
    loader = DataLoader(graphs, batch_size=2, shuffle=False)

    out = collect_dispersion_energies(model, dispersion_energy, loader, "cpu")

    for prefix in ("pred", "true"):
        total = out[f"{prefix}_total"]
        parts_sum = out[f"{prefix}_e6"] + out[f"{prefix}_e8"] + out[f"{prefix}_e10"]
        assert np.allclose(total, parts_sum, atol=1e-10)
    assert out["n_atoms"].tolist() == [3, 5]


def test_main_cli_writes_dispersion_report(tmp_path, monkeypatch):
    z_table = AtomicNumberTable([1, 6, 7, 8])
    args = _make_args_dict()
    rng = np.random.RandomState(2)
    element_means = rng.rand(len(z_table), 4) + 1.0
    element_stds = rng.rand(len(z_table), 4) * 0.1 + 0.1

    model = build_atomic_xdm_mace_from_args(
        args=args, z_table=z_table, avg_num_neighbors=4.0,
        element_means=element_means, element_stds=element_stds,
    )
    checkpoint = _make_checkpoint_dict(model, args, z_table, 4.0, element_means, element_stds)
    model_path = tmp_path / "xdm_test_best.pt"
    torch.save(checkpoint, model_path)

    data_path = tmp_path / "dataset.h5"
    _write_dataset(data_path, z_table)
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"train": [], "valid": [], "test": [f"mol_{i}" for i in range(6)]}))

    output_dir = tmp_path / "out"
    argv = [
        "mace_test_xdm_dispersion",
        "--model", str(model_path),
        "--test_files", str(data_path),
        "--split_file", str(split_path),
        "--output_dir", str(output_dir),
        "--no_plots",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()

    report = json.loads((output_dir / "dispersion_report.json").read_text())
    assert set(report.keys()) == {"total", "e6", "e8", "e10"}
    for key in report:
        assert report[key]["n"] == 6
        for metric in ("mae_hartree", "rmse_hartree", "mae_kcal_mol", "rmse_kcal_mol", "r2", "pearson_r"):
            assert metric in report[key]
