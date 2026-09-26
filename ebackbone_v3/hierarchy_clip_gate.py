"""Read-only readiness and split audits for bounded hierarchy/CLIP checks.

This module never imports torch, allocates CUDA memory, modifies a study, or
launches training. A successful readiness result is an observation, not a GPU
reservation: callers must check immediately before their bounded GPU work.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def process_alive(pid):
    """Probe a PID without sending a signal; inaccessible state fails closed."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError(f"invalid process ID: {pid!r}")
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return False
    return state != "Z"


def check_gpu_readiness(study_root, *, runner=subprocess.run, process_probe=process_alive):
    """Fail closed until V2 is exported, queue stopped, and both GPUs are idle.

    ``runner`` and ``process_probe`` are injectable only for pure CPU unit tests.
    Production checks must run with host GPU/process visibility. Acquiring the
    existing supervisor lock also checks shutdown across PID namespaces.
    """
    root = Path(study_root).resolve()
    result = dict(ready=False, checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  study_root=str(root), reasons=[], full_training_authorized=False)
    reasons = result["reasons"]
    try:
        plan, queue = _json(root / "plan.json"), _json(root / "queue_status.json")
        result["queue_stage"] = queue.get("stage")
        if queue.get("stage") not in {"paused", "complete", "completed"}:
            reasons.append("study queue is not paused or complete")
        expected_uuids = plan.get("cuda_visible_devices", "").split(",")
        if len(expected_uuids) != 2 or len(set(expected_uuids)) != 2 or not all(
                value.startswith("GPU-") for value in expected_uuids):
            raise ValueError("study must name two distinct GPU UUIDs")
        result["expected_gpu_uuids"] = expected_uuids
        report_path = root / "runs" / "hierarchy_ts_residual" / "report.json"
        report = _json(report_path)
        result["residual_completed_epochs"] = report.get("completed_epochs")
        if report.get("status") != "complete" or report.get("completed_epochs") != 50:
            reasons.append("TS residual V2 has not completed its 50-epoch report")
        if not report.get("backbone_exported") or not report.get("trained_model_reload_bit_exact"):
            reasons.append("TS residual V2 checkpoint export/reload verification is incomplete")
        result["verified_artifacts"] = {}
        for kind in ("best", "last"):
            verification = report.get("checkpoint_verification", {}).get(kind, {})
            for prefix, field in (("checkpoint", "checkpoint_sha256"), ("backbone", "backbone_sha256")):
                path = report_path.parent / f"{prefix}_{kind}.pt"
                expected_hash = verification.get(field)
                if not expected_hash or not path.is_file():
                    reasons.append(f"missing verified residual artifact: {path.name}")
                elif sha256_file(path) != expected_hash:
                    reasons.append(f"residual artifact checksum mismatch: {path.name}")
                else:
                    result["verified_artifacts"][path.name] = expected_hash
        # Both supervisors use this same advisory lock. Read-only open does not
        # change study bytes; nonblocking lock acquisition never pauses a job.
        with (root / "supervisor.lock").open("rb") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                reasons.append("study supervisor lock is still held")
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        pids = {key: queue[key] for key in ("supervisor_pid", "torchrun_pid") if key in queue}
        pid_file = root / "supervisor_pid.json"
        if pid_file.is_file():
            for key, value in _json(pid_file).items():
                if "pid" in key:
                    pids[f"original_{key}"] = value
        request_path = root / "pause_after_ts_v2" / "request.json"
        if request_path.is_file():
            request = _json(request_path)
            pids.update({key: request[key] for key in ("old_supervisor_pid", "torchrun_pid") if key in request})
        if not pids:
            reasons.append("no study supervisor process identities available")
        result["live_study_processes"] = {key: pid for key, pid in pids.items() if process_probe(pid)}
        if result["live_study_processes"]:
            reasons.append("a study supervisor or torchrun process is still alive")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        reasons.append(f"study completion cannot be verified: {exc}")

    # Always query both inventory and compute processes; an empty/error inventory
    # must never be interpreted as free GPUs (including sandbox NVML failures).
    try:
        inventory = runner(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True, timeout=15)
        available = {line.strip() for line in inventory.stdout.splitlines() if line.strip()}
        expected = result.get("expected_gpu_uuids", [])
        result["visible_gpu_uuids"] = sorted(available)
        if len(expected) != 2 or not set(expected).issubset(available):
            reasons.append("both study GPU UUIDs are not visible")
        occupancy = runner(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True, timeout=15)
        processes = []
        for row in csv.reader(io.StringIO(occupancy.stdout)):
            if not row or not any(part.strip() for part in row):
                continue
            if len(row) != 4 or not row[0].strip().startswith("GPU-"):
                raise ValueError("unrecognized nvidia-smi compute-process output")
            uuid, pid, name, memory = (part.strip() for part in row)
            processes.append(dict(gpu_uuid=uuid, pid=int(pid), process_name=name, used_memory_mib=memory))
        result["study_gpu_processes"] = [p for p in processes if p["gpu_uuid"] in expected]
        if result["study_gpu_processes"]:
            reasons.append("one or both study GPUs still have compute processes")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        reasons.append(f"host GPU readiness unavailable: {exc}")
    result["ready"] = not reasons
    return result


def assert_gpu_ready(study_root, **kwargs):
    result = check_gpu_readiness(study_root, **kwargs)
    if not result["ready"]:
        raise RuntimeError("GPU testing deferred: " + "; ".join(result["reasons"]))
    return result


def audit_mini_manifests(manifest_dir):
    """Audit immutable train/internal-validation metadata, without opening test.

    The two entire JSONL manifests are small metadata scans; event payloads and
    test.jsonl are never opened. Identity/content overlap is exact-byte only.
    """
    root = Path(manifest_dir)
    provenance = _json(root / "provenance.json")
    class_map = provenance["class_to_index"]
    if sorted(class_map.values()) != list(range(len(class_map))):
        raise ValueError("class mapping must be contiguous and unique")
    sets, summaries = {}, {}
    for split in ("train", "validation"):
        path = root / f"{split}.jsonl"
        metadata = provenance["manifest_files"][split]
        digest = sha256_file(path)
        if digest != metadata["sha256"]:
            raise ValueError(f"{split} manifest SHA256 differs from immutable provenance")
        identities = {key: set() for key in ("sample_id", "source_locator", "raw_content_sha256")}
        counts = {key: 0 for key in class_map}
        with path.open() as stream:
            for count, line in enumerate(stream, 1):
                row = json.loads(line)
                if row["schema_version"] != 1 or row["split"] != split or row["source_split"] != "train":
                    raise ValueError(f"{split} has incorrect split/schema/source identity")
                label = class_map.get(row["class_id"])
                if label is None or row["class_label"] != label or row["class_index"] != label:
                    raise ValueError(f"{split} class labels differ from immutable mapping")
                counts[row["class_id"]] += 1
                for key, values in identities.items():
                    if row[key] in values:
                        raise ValueError(f"duplicate {key} within {split}")
                    values.add(row[key])
        count = len(identities["sample_id"])
        if count != metadata["sample_count"] or any(value == 0 for value in counts.values()):
            raise ValueError(f"{split} count/class coverage differs from immutable provenance")
        sets[split] = identities
        summaries[split] = dict(samples=count, sha256=digest, class_counts=counts, source_split="train")
    intersections = {key: len(sets["train"][key] & sets["validation"][key]) for key in sets["train"]}
    if any(intersections.values()):
        raise ValueError(f"train/validation overlap: {intersections}")
    return dict(status="PASS", dataset="N-ImageNet Mini", purpose="bounded engineering checks only",
                downstream_action_dataset=None, final_test_accessed=False, classes=len(class_map),
                class_to_index=class_map, splits=summaries, train_validation_intersections=intersections,
                overlap_scope="sample identity, source locator, exact stored NPZ SHA256; no semantic deduplication",
                provenance_sha256=sha256_file(root / "provenance.json"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=float, default=0,
                        help="bounded wait only; this command never launches GPU work")
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args(argv)
    if args.wait_seconds < 0 or not 1 <= args.poll_seconds <= 60:
        parser.error("wait must be nonnegative and poll interval must be 1..60 seconds")
    deadline = time.monotonic() + args.wait_seconds
    while True:
        result = check_gpu_readiness(args.study_root)
        print(json.dumps(result, sort_keys=True), flush=True)
        remaining = deadline - time.monotonic()
        if result["ready"] or remaining <= 0:
            return 0 if result["ready"] else 2
        time.sleep(min(args.poll_seconds, remaining))


if __name__ == "__main__":
    raise SystemExit(main())
