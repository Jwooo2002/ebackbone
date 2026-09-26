"""Read-only W&B sidecar for existing Mini V1 training artifacts.

No torch import, trainer restart, checkpoint upload or dataset access. Completed
epochs are backfilled from atomic history files; progress follows the JSON log.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import statistics
import time
from pathlib import Path


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def epoch_metrics(record):
    result = {"epoch": record["epoch"], "learning_rate": record["learning_rate"],
              "sync/completed_epochs": record["epoch"]}
    for split in ("train", "validation"):
        values = record[split]
        for key in ("loss", "samples", "seconds", "samples_per_second", "forward_ms_per_sample"):
            if key in values:
                result[f"{split}/{key}"] = values[key]
        for key in ("top1", "top5"):
            result[f"{split}/{key}_pct"] = 100 * values[key]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in result.values()):
        raise ValueError("nonfinite or invalid epoch metric")
    return result


def latest_progress(path, model):
    path = Path(path)
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 262144))
        lines = handle.read().splitlines(keepends=True)
    for line in reversed(lines):
        if not line.endswith(b"\n"):
            continue  # Never consume an in-progress write.
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(row, dict) and row.get("event") == "optimizer_step" and row.get("model") == model:
            return row
    return None


def validate_history(history):
    if [r["epoch"] for r in history] != list(range(1, len(history) + 1)):
        raise ValueError("noncontiguous epoch history")
    if any(r["train"]["samples"] != 124395 or r["validation"]["samples"] != 5000 for r in history):
        raise ValueError("expected complete Mini train/internal-validation epochs")


def completed(report, history, budget):
    return bool(report and report.get("status") == "complete" and report.get("completed_epochs") == budget
                and len(history) == budget and report.get("checkpoint_verification", {}).get("logits_bit_exact"))


def public_config(identity, spec):
    keys = ("architecture", "model_version", "settings", "renderer", "point_contract", "dataset",
            "train_manifest_sha256", "validation_manifest_sha256", "source_sha256", "optimizer", "scheduler",
            "objective", "augmentation", "torch_version", "numpy_version")
    return {**{k: identity[k] for k in keys if k in identity},
            "physical_gpu_index": spec["physical_gpu_index"], "tracking_mode": "read-only artifact sidecar",
            "accuracy_units": "percent", "final_test_evaluated": False}


def monitor(registry_path, *, follow=True):
    import wandb

    registry = read_json(registry_path)
    output = Path(registry["state_dir"])
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "monitor.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = output / "status.json"
    previous = read_json(state_path, {}).get("runs", {})
    state = {"project_url": f"https://wandb.ai/{registry['entity']}/{registry['project']}", "runs": {},
             "poll_seconds": registry.get("poll_seconds", 30), "wandb_version": wandb.__version__}
    handles, cursors, progress_keys = {}, {}, {}
    while True:
        terminal = 0
        for spec in registry["runs"]:
            name = spec["model"]
            try:
                path = Path(spec["run_dir"])
                identity = read_json(path / "config.json")
                if identity is None:
                    failed_parent = read_json(spec["supervisor_status"], {}).get("stage") == "failed"
                    state["runs"][name] = {"status": "blocked" if failed_parent else "queued", "url": None}
                    if failed_parent:
                        terminal += 1
                    continue
                if identity["architecture"] != name or identity["diagnostic_samples"] is not None:
                    raise ValueError("unexpected architecture or diagnostic run")
                history = read_json(path / "history.json", [])
                validate_history(history)
                report = read_json(path / "report.json")
                done = completed(report, history, identity["settings"]["epochs"])
                old = state["runs"].get(name, previous.get(name, {}))
                if done and old.get("status") == "finished":
                    state["runs"][name] = old
                    terminal += 1
                    continue
                if old.get("status") == "stopped" and read_json(spec["supervisor_status"], {}).get("stage") == "stopped_by_user":
                    state["runs"][name] = old
                    terminal += 1
                    continue
                if old.get("status") == "failed" and read_json(spec["supervisor_status"], {}).get("stage") == "failed":
                    state["runs"][name] = old
                    terminal += 1
                    continue
                if name not in handles:
                    run_id = "v1" + hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:14]
                    run = wandb.init(entity=registry["entity"], project=registry["project"], id=run_id,
                                     name=spec["display_name"], group=registry["group"], job_type="supervised-classification",
                                     resume="allow", reinit="create_new", mode="online", dir=str(output),
                                     config=public_config(identity, spec), tags=["N-ImageNet-Mini", "V1", "seed20260908"],
                                     notes="Historical epochs backfilled from training artifacts. Live fields update every 30s. System auto-stats are disabled because this is a separate CPU logger. Training settings and source snapshots are unchanged.",
                                     settings=wandb.Settings(console="off", disable_git=True, disable_code=True,
                                                            x_disable_stats=True, x_disable_meta=True,
                                                            x_save_requirements=False, init_timeout=60))
                    run.define_metric("epoch")
                    for prefix in ("train/*", "validation/*"):
                        run.define_metric(prefix, step_metric="epoch")
                    run.define_metric("learning_rate", step_metric="epoch")
                    run.define_metric("progress_epoch")
                    run.define_metric("live/*", step_metric="progress_epoch")
                    handles[name] = run
                    # The remote resumed summary is authoritative, not a local
                    # cursor that might have advanced before upload finished.
                    cursors[name] = int(run.summary.get("sync/completed_epochs", 0))
                run = handles[name]
                for row in history:
                    if row["epoch"] > cursors[name]:
                        run.log(epoch_metrics(row))
                        cursors[name] = row["epoch"]
                summary = {"training/status": "complete" if done else "running",
                           "sync/completed_epochs": len(history), "sync/last_poll_unix": time.time()}
                if history:
                    best = max(history, key=lambda r: (r["validation"]["top1"], -r["validation"]["loss"]))
                    summary.update({"best/epoch": best["epoch"], "best/val_top1_pct": 100 * best["validation"]["top1"],
                                    "best/val_top5_pct": 100 * best["validation"]["top5"],
                                    "best/val_loss": best["validation"]["loss"],
                                    "latest/val_top1_pct": 100 * history[-1]["validation"]["top1"]})
                progress = latest_progress(spec["log_path"], name)
                if progress and not done:
                    key = (progress["epoch"], progress["update_in_epoch"])
                    fraction = max(len(history), progress["epoch"] - 1 + progress["samples_seen"] / 124395)
                    summary.update({"live/current_epoch": progress["epoch"], "live/progress_percent": fraction,
                                    "live/source_update_age_seconds": max(0, time.time() - progress["unix_time"])})
                    if history:
                        seconds = statistics.median(r["train"]["seconds"] + r["validation"]["seconds"] for r in history[-5:])
                        remaining = max(0, (identity["settings"]["epochs"] - fraction) * seconds)
                        summary.update({"live/eta_remaining_hours": remaining / 3600,
                                        "live/estimated_finish_unix": time.time() + remaining})
                    if progress_keys.get(name) != key:
                        run.log({"progress_epoch": fraction, "live/batch_loss": progress["batch_mean_loss"],
                                 "live/current_epoch": progress["epoch"], "live/samples_seen_in_epoch": progress["samples_seen"]})
                        progress_keys[name] = key
                supervisor = read_json(spec["supervisor_status"], {})
                stopped = not done and supervisor.get("stage") == "stopped_by_user"
                failed = not done and supervisor.get("stage") == "failed"
                if stopped:
                    summary.update({"training/status": "stopped_early", "training/stop_reason": "user_requested",
                                    "training/stopped_at_completed_epoch": len(history),
                                    "live/eta_remaining_hours": 0, "live/estimated_finish_unix": None})
                if failed:
                    summary["training/status"] = "failed"
                run.summary.update(summary)
                state["runs"][name] = {"url": run.url, "id": run.id, "completed_epochs": len(history),
                                       "status": "finished" if done else "stopped" if stopped else "failed" if failed else "running"}
                if done or failed or stopped:
                    run.finish(exit_code=1 if failed else 0)
                    terminal += 1
                    handles.pop(name)
            except Exception as exc:
                state["runs"][name] = {**state["runs"].get(name, {}), "status": "sync_error", "error": str(exc)}
                print(json.dumps({"model": name, "sync_error": str(exc)}), flush=True)
        state["updated_unix"] = time.time()
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(state_path)
        if not follow or terminal == len(registry["runs"]):
            # --once is a debugging flush, not proof that training is complete.
            for run in handles.values():
                run.finish()
            break
        time.sleep(state["poll_seconds"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    monitor(args.registry, follow=not args.once)


if __name__ == "__main__":
    main()
