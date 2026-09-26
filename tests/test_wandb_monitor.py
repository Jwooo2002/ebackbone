import json

import pytest

from ebackbone_v3.wandb_monitor import completed, epoch_metrics, latest_progress, public_config, validate_history


def row():
    return {"epoch": 1, "learning_rate": .05,
            "train": {"samples": 124395, "loss": 2., "top1": .4, "top5": .7},
            "validation": {"samples": 5000, "loss": 2.1, "top1": .373, "top5": .7206}}


def test_epoch_units_scope_and_rejection():
    r = row()
    values = epoch_metrics(r)
    assert values["validation/top1_pct"] == pytest.approx(37.3)
    assert values["validation/top5_pct"] == pytest.approx(72.06)
    assert values["epoch"] == values["sync/completed_epochs"] == 1
    validate_history([r])
    r["validation"]["samples"] = 4999
    with pytest.raises(ValueError):
        validate_history([r])
    r["train"]["loss"] = float("nan")
    with pytest.raises(ValueError):
        epoch_metrics(r)


def test_log_filter_and_partial_line(tmp_path):
    path = tmp_path / "log"
    a = {"event": "optimizer_step", "model": "hierarchy", "epoch": 3}
    b = {"event": "optimizer_step", "model": "frame", "epoch": 90}
    path.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n" + json.dumps({**a, "epoch": 4}))
    assert latest_progress(path, "hierarchy")["epoch"] == 3
    assert latest_progress(path, "frame")["epoch"] == 90
    assert latest_progress(path, "same_arch") is None


def test_completion_requires_report_and_reload():
    h = [row()]
    assert not completed(None, h, 1)
    report = {"status": "complete", "completed_epochs": 1, "checkpoint_verification": {"logits_bit_exact": True}}
    assert completed(report, h, 1)
    assert not completed(report, h, 100)
    report["checkpoint_verification"]["logits_bit_exact"] = False
    assert not completed(report, h, 1)


def test_public_config_excludes_raw_cache_and_unlisted_fields():
    value = public_config({"architecture": "hierarchy", "settings": {"batch_size": 32},
                           "raw_cache": "/private/path", "unexpected": "not-to-upload"}, {"physical_gpu_index": 0})
    assert "raw_cache" not in value and "unexpected" not in value
    assert value["settings"]["batch_size"] == 32
    assert value["accuracy_units"] == "percent"
