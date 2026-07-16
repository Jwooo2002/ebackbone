"""Train-only, bounded B0 throughput benchmark and pilot on cuda:1."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from ebackbone_v3.b0_models import B0_CLASS_COUNT, build_b0_model
from ebackbone_v3.b0_production import _seed_worker, collate_production_b0
from ebackbone_v3.errors import TrainingError
from ebackbone_v3.n_imagenet_mini_dataset import DEFAULT_N_IMAGENET_MINI_DATASET_ROOT, open_dataset


def run_b0_throughput_benchmark(
    *, manifest_dir: str | Path, output_dir: str | Path,
    dataset_root: str | Path = DEFAULT_N_IMAGENET_MINI_DATASET_ROOT,
    cache_root: str | Path, seed: int = 20260716, per_class: int = 1,
) -> dict[str, Any]:
    """Benchmark only project train B0 frames, then run a cached three-epoch pilot."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise TrainingError("the bounded B0 benchmark requires available cuda:1")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise TrainingError("benchmark output directory must be new")
    output.mkdir(parents=True)
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # This command intentionally has no validation/test argument or dataset construction.
    uncached = open_dataset(manifest_dir=manifest_dir, dataset_root=dataset_root, split="train", baseline="b0", cache="off")
    del per_class  # The cache benchmark contract is exactly 512 project-train samples.
    subset, subset_audit = _deterministic_512_subset(uncached, seed=seed)
    cached = open_dataset(manifest_dir=manifest_dir, dataset_root=dataset_root, split="train", baseline="b0", cache="on", cache_root=cache_root)
    cached_subset = Subset(cached, subset.indices)  # type: ignore[attr-defined]

    # Explicit warm-up creates only B0 frame entries atomically before timing cache hits.
    for index in subset.indices:  # type: ignore[attr-defined]
        item = cached[index]
        if set(item.tensors) != {"event_frame"}:
            raise TrainingError("B0 cache exposed non-frame tensors")

    uncached_measurements: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for workers in (0, 2):
        for batch_size in (8, 16):
            try:
                uncached_measurements.append(_measure(
                    dataset=subset, batch_size=batch_size, workers=workers,
                    device=device, seed=seed, limit_batches=8,
                ))
                candidates.append(_measure(
                    dataset=cached_subset, batch_size=batch_size, workers=workers,
                    device=device, seed=seed, limit_batches=8,
                ))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                candidates.append({"batch_size": batch_size, "num_workers": workers, "status": "OOM"})
    stable = [row for row in candidates if row.get("status") == "PASS"]
    if not stable:
        raise TrainingError("no requested cached configuration completed safely")
    selected = max(stable, key=lambda row: float(row["samples_per_second"]))
    pilot = _pilot(cached_subset, selected, device=device, seed=seed)
    report = {
        "status": "PASS",
        "device": "cuda:1",
        "project_splits_opened": ["train"],
        "representation": "event_frame only",
        "subset": subset_audit,
        "uncached": uncached_measurements,
        "candidates": candidates,
        "selected": selected,
        "pilot": pilot,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _loader(dataset: Any, *, batch_size: int, workers: int, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    kwargs: dict[str, Any] = {
        "dataset": dataset, "batch_size": batch_size, "shuffle": False,
        "num_workers": workers, "pin_memory": True, "collate_fn": collate_production_b0,
        "worker_init_fn": _seed_worker, "generator": generator,
    }
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs)


def _measure(*, dataset: Any, batch_size: int, workers: int, device: torch.device, seed: int, limit_batches: int) -> dict[str, Any]:
    loader = _loader(dataset, batch_size=batch_size, workers=workers, seed=seed)
    model = build_b0_model(class_count=B0_CLASS_COUNT).to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    torch.cuda.reset_peak_memory_stats(device)
    archive = frame = wait = transfer = compute = 0.0
    utilization: list[int] = []
    samples = 0
    iterator = iter(loader)
    started = time.perf_counter()
    for _ in range(limit_batches):
        utilization_process = subprocess.Popen(
            ["nvidia-smi", "--id=1", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        before = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        wait += time.perf_counter() - before
        archive += sum(batch["archive_decode_seconds"])
        frame += sum(batch["frame_stage_seconds"])
        before = time.perf_counter()
        inputs = batch["event_frames"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        torch.cuda.synchronize(device)
        transfer += time.perf_counter() - before
        before = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(inputs), labels)
        loss.backward(); optimizer.step(); torch.cuda.synchronize(device)
        compute += time.perf_counter() - before
        samples += len(labels)
        stdout, _ = utilization_process.communicate()
        try: utilization.append(int(stdout.strip()))
        except ValueError: utilization.append(0)
    elapsed = time.perf_counter() - started
    del model, optimizer
    torch.cuda.empty_cache()
    return {
        "status": "PASS", "batch_size": batch_size, "num_workers": workers,
        "pin_memory": True, "persistent_workers": bool(workers), "prefetch_factor": 2 if workers else None,
        "samples": samples, "samples_per_second": samples / elapsed,
        "timing_seconds": {"archive_read_decode": archive, "frame_render_or_cache_read": frame,
                           "data_wait": wait, "host_to_device": transfer, "forward_backward_optimizer": compute},
        "gpu_utilization_percent": sum(utilization) / len(utilization) if utilization else 0.0,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def _pilot(dataset: Any, selected: dict[str, Any], *, device: torch.device, seed: int) -> dict[str, Any]:
    loader = _loader(dataset, batch_size=int(selected["batch_size"]), workers=int(selected["num_workers"]), seed=seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = build_b0_model().to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    losses: list[float] = []
    for _epoch in range(3):
        total = count = 0
        for batch in loader:
            inputs = batch["event_frames"].to(device, non_blocking=True); labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = F.cross_entropy(model(inputs), labels)
            loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(labels); count += len(labels)
        losses.append(total / count)
    if not losses[-1] < losses[0]:
        raise TrainingError(f"three-epoch B0 train loss did not decrease: {losses}")
    return {"epochs": 3, "train_loss": losses}


def _deterministic_512_subset(dataset: Any, *, seed: int) -> tuple[Subset[Any], dict[str, Any]]:
    """Pick exactly 512 immutable project-train IDs without opening samples."""
    sample_ids = dataset.sample_ids
    ranked = sorted(
        range(len(dataset)),
        key=lambda index: __import__("hashlib").sha256(
            b"ebackbone-v3/b0/frame-cache-512/v1\0" + str(seed).encode("ascii") + b"\0"
            + sample_ids[index].encode("utf-8")
        ).digest(),
    )
    if len(ranked) < 512:
        raise TrainingError("project-train manifest has fewer than the required 512 samples")
    indices = ranked[:512]
    return Subset(dataset, indices), {
        "selection": "sha256-ranked immutable project-train sample IDs",
        "seed": seed, "sample_count": len(indices),
        "sample_id_sha256": __import__("hashlib").sha256(
            b"".join(sample_ids[index].encode("utf-8") + b"\n" for index in indices)
        ).hexdigest(),
    }
