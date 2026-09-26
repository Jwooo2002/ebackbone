"""Opt-in tests of the local study queue; no subprocesses or GPUs are launched."""

import importlib.util
import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.local_integration
QUEUE_ROOT = (
    Path(__file__).resolve().parents[2] / "ebackbone_v3_hierarchy_ddp_b64_e50_artifacts"
)


def queue(tmp_path, monkeypatch):
    required = (QUEUE_ROOT / "supervise.py", QUEUE_ROOT / "plan.json")
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        pytest.skip(f"local study queue assets are unavailable: {', '.join(missing)}")
    spec = importlib.util.spec_from_file_location("queue_under_test", required[0])
    q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)
    plan = json.loads(required[1].read_text())
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    for name, value in (
        ("cpu_verification.json", {"status": "PASS"}),
        ("protected_before.json", {}),
        ("completed_current_protected.json", {}),
    ):
        (tmp_path / name).write_text(json.dumps(value))
    monkeypatch.setattr(q, "ROOT", tmp_path)
    monkeypatch.setattr(q, "STATUS", tmp_path / "queue_status.json")
    monkeypatch.setattr(q, "export_results", lambda plan: None)
    events = []
    monkeypatch.setattr(q, "wait_current", lambda plan: events.append("current_finished"))
    monkeypatch.setattr(q, "run_child", lambda plan, name, mode: events.append((mode, name)))
    return q, events, plan


def test_all_preflights_precede_any_training(tmp_path, monkeypatch):
    q, events, plan = queue(tmp_path, monkeypatch)
    q.main()
    names = [variant["name"] for variant in plan["variants"]]
    assert events == (
        ["current_finished"]
        + [("preflight", name) for name in names]
        + [("train", name) for name in names]
    )
    assert json.loads(q.STATUS.read_text())["stage"] == "complete"
    with pytest.raises(RuntimeError, match="Existing queue"):
        q.main()


def test_failed_preflight_prevents_all_training(tmp_path, monkeypatch):
    q, events, _ = queue(tmp_path, monkeypatch)

    def fail(plan, name, mode):
        events.append((mode, name))
        raise RuntimeError("preflight failed")

    monkeypatch.setattr(q, "run_child", fail)
    with pytest.raises(RuntimeError, match="preflight failed"):
        q.main()
    assert events == ["current_finished", ("preflight", "hierarchy")]
