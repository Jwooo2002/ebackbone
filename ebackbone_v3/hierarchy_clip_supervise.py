"""Sequential, separately locked supervisor for the authorized Mini study."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

from .hierarchy_clip_gate import check_gpu_readiness, sha256_file

VARIANTS = ("hierarchy_temporal", "hierarchy_text", "hierarchy_clip_vit")


def atomic_json(path, value):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def verify_protected(study):
    hashes = json.loads((study / "protected_before.json").read_text())
    changed = [name for name, digest in hashes.items() if sha256_file(name) != digest]
    if changed:
        raise RuntimeError(f"protected prior study or weights changed: {changed}")


def check_launch_contract(study, config):
    gate = json.loads(Path(config["training_gate_path"]).read_text())
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if gate.get("status") != "PASS" or gate.get("all_variants_passed") is not True or gate.get("config_sha256") != digest:
        raise ValueError("all-variant preflight gate does not match the training configuration")
    if set(gate.get("variants", [])) != set(VARIANTS):
        raise ValueError("preflight gate does not cover the three comparisons")
    for name, digest in gate["source_sha256"].items():
        if sha256_file(Path(__file__).parent / name) != digest:
            raise ValueError(f"preflight source changed: {name}")
    for name, record in gate["reports"].items():
        path = Path(record["path"])
        if sha256_file(path) != record["sha256"] or json.loads(path.read_text()).get("status") != "PASS":
            raise ValueError(f"preflight report changed or failed: {name}")
    verify_protected(study)


def supervise(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    study = Path(config["study_root"])
    study.mkdir(parents=True, exist_ok=True)
    with (study / "supervisor.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (study / "queue_status.json").exists():
            raise FileExistsError("existing new-study queue needs explicit resume inspection")
        check_launch_contract(study, config)

        def status(stage, **extra):
            record = {"stage": stage, "supervisor_pid": os.getpid(),
                      "updated_at_utc": datetime.now(timezone.utc).isoformat(), **extra}
            atomic_json(study / "queue_status.json", record)
            print(json.dumps(record), flush=True)

        completed = []
        try:
            for variant in VARIANTS:
                check_launch_contract(study, config)
                readiness = check_gpu_readiness(config["wait_for_study"])
                if not readiness["ready"]:
                    raise RuntimeError("GPUs or paused prior queue are not ready: " + "; ".join(readiness["reasons"]))
                atomic_json(study / f"readiness_before_{variant}.json", readiness)
                environment = os.environ.copy()
                environment.update(CUDA_VISIBLE_DEVICES=",".join(readiness["expected_gpu_uuids"]),
                                   CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
                command = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                           "--nproc-per-node=2", "--max-restarts=0", "-m", "ebackbone_v3.hierarchy_clip_train",
                           "--config", str(config_path), "--variant", variant, "--output-dir",
                           str(study / "runs" / variant), "--mode", "train"]
                with (study / f"train_{variant}.log").open("x") as log:
                    process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    status("training", variant=variant, torchrun_pid=process.pid, command=command,
                           completed_variants=completed, remaining_variants=list(VARIANTS[len(completed) + 1:]))
                    code = process.wait()
                if code:
                    raise RuntimeError(f"{variant} exited {code}; remaining new-study variants are held")
                report = json.loads((study / "runs" / variant / "report.json").read_text())
                if report.get("status") != "complete" or report.get("completed_epochs") != 50:
                    raise RuntimeError(f"{variant} did not produce a completed 50-epoch report")
                completed.append(variant)
                verify_protected(study)
            status("complete", completed_variants=completed, full_training_epochs_per_variant=50)
        except BaseException as error:
            status("failed", error=f"{type(error).__name__}: {error}", completed_variants=completed)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    supervise(args.config)


if __name__ == "__main__":
    main()
