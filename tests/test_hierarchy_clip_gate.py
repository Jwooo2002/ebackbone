"""Readiness safety tests use fake processes/GPUs and never import CUDA."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from ebackbone_v3.hierarchy_clip_gate import audit_mini_manifests, check_gpu_readiness, sha256_file


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def ready_study(tmp_path):
    put(tmp_path / "plan.json", {"cuda_visible_devices": "GPU-a,GPU-b"})
    put(tmp_path / "queue_status.json", {"stage": "paused", "supervisor_pid": 7, "torchrun_pid": 8})
    (tmp_path / "supervisor.lock").touch()
    run = tmp_path / "runs/hierarchy_ts_residual"
    run.mkdir(parents=True)
    verification = {}
    for kind in ("best", "last"):
        verification[kind] = {}
        for prefix in ("backbone", "checkpoint"):
            path = run / f"{prefix}_{kind}.pt"
            path.write_bytes(b"mock completed export")
            verification[kind][f"{prefix}_sha256"] = sha256_file(path)
    put(run / "report.json", dict(status="complete", completed_epochs=50, backbone_exported=True,
                                 trained_model_reload_bit_exact=True, checkpoint_verification=verification))
    return tmp_path


def gpu_runner(occupancy="", uuids="GPU-a\nGPU-b\n"):
    def run(command, **kwargs):
        assert command[0] == "nvidia-smi" and kwargs["check"] is True
        return SimpleNamespace(stdout=uuids if "--query-gpu=uuid" in command else occupancy)
    return run


def check(root, runner=None, process_probe=lambda pid: False):
    return check_gpu_readiness(root, runner=runner or gpu_runner(), process_probe=process_probe)


def test_finished_exported_paused_idle_is_ready(tmp_path):
    result = check(ready_study(tmp_path))
    assert result["ready"] and len(result["verified_artifacts"]) == 4
    assert not result["full_training_authorized"]


@pytest.mark.parametrize("stage", ["train", "train_pause_armed", "pause_monitor_failed", "failed"])
def test_nonfinal_queue_is_blocked_even_with_idle_gpus(tmp_path, stage):
    root = ready_study(tmp_path)
    put(root / "queue_status.json", {"stage": stage, "supervisor_pid": 7})
    assert not check(root)["ready"]


def test_running_supervisor_blocks_paused_snapshot(tmp_path):
    assert not check(ready_study(tmp_path), process_probe=lambda pid: pid == 7)["ready"]


def test_other_process_on_either_study_gpu_blocks(tmp_path):
    root = ready_study(tmp_path)
    for gpu in ("GPU-a", "GPU-b"):
        assert not check(root, gpu_runner(f"{gpu}, 88, python, 100\n"))["ready"]
    assert check(root, gpu_runner("GPU-unrelated, 99, python, 100\n"))["ready"]


def test_missing_uuid_or_invalid_process_output_blocks(tmp_path):
    root = ready_study(tmp_path)
    assert not check(root, gpu_runner(uuids="GPU-a\n"))["ready"]
    assert not check(root, gpu_runner("No running processes found.\n"))["ready"]


def test_nvml_unavailable_fails_closed(tmp_path):
    def failed(command, **kwargs):
        raise subprocess.CalledProcessError(9, command, stderr="driver unavailable")
    result = check(ready_study(tmp_path), failed)
    assert not result["ready"] and any("host GPU readiness unavailable" in x for x in result["reasons"])


@pytest.mark.parametrize("damage", ["report", "epochs", "export", "digest"])
def test_missing_or_incomplete_completion_evidence_blocks(tmp_path, damage):
    root = ready_study(tmp_path)
    run = root / "runs/hierarchy_ts_residual"
    if damage == "report":
        (run / "report.json").unlink()
    elif damage == "epochs":
        report = json.loads((run / "report.json").read_text())
        report["completed_epochs"] = 49
        put(run / "report.json", report)
    elif damage == "export":
        (run / "backbone_last.pt").unlink()
    else:
        (run / "backbone_last.pt").write_bytes(b"corrupt")
    assert not check(root)["ready"]


def test_supervisor_lock_prevents_namespace_pid_false_negative(tmp_path):
    import fcntl
    root = ready_study(tmp_path)
    with (root / "supervisor.lock").open("rb") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = check(root)
        assert not result["ready"]
        assert "study supervisor lock is still held" in result["reasons"]


def mini_manifest(tmp_path, *, overlap=False):
    metadata = {}
    for split in ("train", "validation"):
        identity = "train" if overlap else split
        row = dict(schema_version=1, split=split, source_split="train", class_id="n0001",
                   class_label=0, class_index=0, sample_id=identity, source_locator=identity,
                   raw_content_sha256=identity)
        path = tmp_path / f"{split}.jsonl"
        path.write_text(json.dumps(row) + "\n")
        metadata[split] = dict(sha256=sha256_file(path), sample_count=1)
    put(tmp_path / "provenance.json", dict(class_to_index={"n0001": 0}, manifest_files=metadata))
    return tmp_path


def test_manifest_audit_never_opens_test(tmp_path):
    root = mini_manifest(tmp_path)
    # A nonexistent final-test manifest makes accidental access a hard failure.
    result = audit_mini_manifests(root)
    assert result["status"] == "PASS" and not result["final_test_accessed"]
    assert result["downstream_action_dataset"] is None


def test_manifest_overlap_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        audit_mini_manifests(mini_manifest(tmp_path, overlap=True))


def test_manifest_tampering_is_rejected(tmp_path):
    root = mini_manifest(tmp_path)
    (root / "train.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="SHA256"):
        audit_mini_manifests(root)
