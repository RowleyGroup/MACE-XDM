import sys

import torch

from mace.cli.run_train_xdm import main
from tests.test_run_train_xdm import _write_master_file


def _run(tmp_path, monkeypatch, name, extra_args, max_num_epochs):
    argv = [
        "mace_run_train_xdm",
        "--name", name,
        "--train_files", str(tmp_path / "master.h5"),
        "--atomic_numbers", "1,6,8",
        "--element_stats", "mlxdm_2x",
        "--hidden_irreps", "8x0e + 8x1o",
        "--num_interactions", "2",
        "--correlation", "2",
        "--max_ell", "2",
        "--batch_size", "4",
        "--max_num_epochs", str(max_num_epochs),
        "--eval_interval", "1",
        "--device", "cpu",
        "--default_dtype", "float64",
        "--results_dir", str(tmp_path / "results"),
        "--checkpoints_dir", str(tmp_path / "checkpoints"),
        "--log_dir", str(tmp_path / "logs"),
        "--model_dir", str(tmp_path / "models"),
        *extra_args,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()


def test_checkpoint_saves_and_restores_scheduler_and_patience_state(tmp_path, monkeypatch):
    torch.set_default_dtype(torch.float64)
    _write_master_file(tmp_path / "master.h5", n_molecules=30)

    _run(tmp_path, monkeypatch, "xdm_restart_test", extra_args=[], max_num_epochs=2)

    latest_path = tmp_path / "checkpoints" / "xdm_restart_test_latest.pt"
    assert latest_path.exists()
    checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
    assert "scheduler_state_dict" in checkpoint
    assert "epochs_without_improvement" in checkpoint
    assert "num_bad_epochs" in checkpoint["scheduler_state_dict"]

    # A restart should pick the scheduler/patience state back up rather than
    # resetting it, and must not crash continuing for more epochs.
    _run(
        tmp_path,
        monkeypatch,
        "xdm_restart_test",
        extra_args=["--restart_latest"],
        max_num_epochs=4,
    )
    resumed_checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
    assert resumed_checkpoint["epoch"] == 3


def test_restart_from_checkpoint_missing_scheduler_state_falls_back_gracefully(
    tmp_path, monkeypatch
):
    torch.set_default_dtype(torch.float64)
    _write_master_file(tmp_path / "master.h5", n_molecules=30)

    _run(tmp_path, monkeypatch, "xdm_restart_legacy", extra_args=[], max_num_epochs=2)

    latest_path = tmp_path / "checkpoints" / "xdm_restart_legacy_latest.pt"
    checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
    del checkpoint["scheduler_state_dict"]
    del checkpoint["epochs_without_improvement"]
    torch.save(checkpoint, latest_path)

    # Simulates resuming a run whose checkpoints predate this fix -- must not
    # crash, and should just start the scheduler/patience state fresh.
    _run(
        tmp_path,
        monkeypatch,
        "xdm_restart_legacy",
        extra_args=["--restart_latest"],
        max_num_epochs=4,
    )
    resumed_checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
    assert resumed_checkpoint["epoch"] == 3
