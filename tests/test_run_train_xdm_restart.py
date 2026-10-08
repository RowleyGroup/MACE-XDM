import logging
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


def test_foundation_model_warm_start_skipped_warns_instead_of_silent(tmp_path, monkeypatch, caplog):
    # A leftover _latest.pt from a prior run previously disabled
    # --foundation_model warm-start AND --restart_latest resuming (neither
    # condition was met) with no indication anything was skipped -- the
    # model trained from random init silently. Must now warn instead.
    torch.set_default_dtype(torch.float64)
    _write_master_file(tmp_path / "master.h5", n_molecules=30)

    _run(tmp_path, monkeypatch, "xdm_warmstart_test", extra_args=[], max_num_epochs=1)
    latest_path = tmp_path / "checkpoints" / "xdm_warmstart_test_latest.pt"
    assert latest_path.exists()

    with caplog.at_level(logging.WARNING):
        # Neither --restart_latest nor a valid foundation model is needed to
        # exercise the warning branch -- it fires, and returns, before the
        # (nonexistent) foundation model path would ever be opened.
        _run(
            tmp_path,
            monkeypatch,
            "xdm_warmstart_test",
            extra_args=["--foundation_model", str(tmp_path / "does_not_exist.model")],
            max_num_epochs=2,
        )
    assert any(
        "foundation_model warm-start is being skipped" in rec.message for rec in caplog.records
    )


def test_nan_validation_loss_raises_instead_of_crashing_on_missing_best_path(
    tmp_path, monkeypatch
):
    # best_path is only ever written when loss improves on best_valid_loss;
    # "nan < anything" is always False in Python, so a NaN validation loss
    # previously meant best_path was never created, and the training script
    # crashed afterwards with a confusing FileNotFoundError instead of
    # reporting the actual divergence. Must now fail fast and clearly.
    import mace.cli.run_train_xdm as run_train_xdm_module

    torch.set_default_dtype(torch.float64)
    _write_master_file(tmp_path / "master.h5", n_molecules=20)

    def fake_evaluate(model, data_loader, device, loss_weights, num_properties):
        metrics = {"loss": float("nan")}
        for i in range(num_properties):
            metrics[f"mae_{i}"] = float("nan")
            metrics[f"rmse_{i}"] = float("nan")
        return metrics

    monkeypatch.setattr(run_train_xdm_module, "evaluate", fake_evaluate)

    try:
        _run(tmp_path, monkeypatch, "xdm_nan_test", extra_args=[], max_num_epochs=2)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "NaN" in str(exc) and "diverged" in str(exc)
    # The crash site this replaces was torch.load(best_path, ...) raising
    # FileNotFoundError after the training loop -- confirm that's gone too.
    assert not (tmp_path / "checkpoints" / "xdm_nan_test_best.pt").exists()


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
