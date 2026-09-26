"""CPU-only windowing ablation with a frozen complete hierarchy checkpoint.

Compare whole clips with four partitions using either local or original global
time coordinates. Every path retains the original spatial pool and class head.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .hierarchy_data import HierarchyDataset, collate, prepare_points
from .hierarchy_models import HierarchyV1
from .hierarchy_clip_data import prepare_sequence, sequence_collate, split_event_windows
from .hierarchy_clip_preservation import (
    _capture_forward, _exact, _protected_hashes, _state_digest, _tensor_digest,
    _verify_hashes, load_full_hierarchy_checkpoint,
)
from .hierarchy_clip_runtime import RuntimeHierarchySequenceEncoder
from .hierarchy_clip_staging import file_sha256
from .hierarchy_clip_temporal import HierarchySequenceEncoder
from .v1_data import raw


def prepare_window_control(fields, source, *, num_windows=4, time_mode="local"):
    """Keep assignments fixed; change only point-time and temporal voxel routing.

    Global mode subsets the original full-clip tensors before collate adds each
    window's voxel offset. Empty temporal bins are retained within each window.
    """
    if time_mode not in ("local", "global"):
        raise ValueError("time_mode must be local or global")
    result = prepare_sequence(fields, source, num_windows=num_windows)
    if time_mode == "global":
        original = prepare_points(fields, source)
        assignments = split_event_windows(fields, source, num_windows=num_windows)
        for window, partition in zip(result["windows"], assignments):
            indices = torch.from_numpy(partition["event_indices"])
            for key in ("points", "voxel_lower", "voxel_upper", "alpha"):
                window[key] = original[key][indices]
    return {**result, "time_mode": time_mode}


def masked_window_mean(features, mask):
    if (features.ndim != 3 or mask.dtype != torch.bool or mask.shape != features.shape[:2]
            or features.device != mask.device):
        raise ValueError("expected window features [B,K,D] and boolean mask [B,K]")
    if not bool(mask.any(dim=1).all()):
        raise ValueError("all-empty samples have no original hierarchy reference")
    return features.masked_fill(~mask[..., None], 0).sum(dim=1) / mask.sum(dim=1, keepdim=True)


class FrozenWindowReadout(nn.Module):
    """No new parameters: original spatial pool, masked mean, original head.

    The caller freezes the complete hierarchy for the diagnostic. No Transformer,
    feature normalization, learned temporal positions, or new classifier is used.
    """

    def __init__(self, hierarchy, num_windows=4, encoder=None):
        super().__init__()
        if encoder is None:
            encoder = HierarchySequenceEncoder(hierarchy, num_windows=num_windows)
        if encoder.hierarchy is not hierarchy or encoder.num_windows != num_windows:
            raise ValueError("encoder must use the same hierarchy and num_windows")
        self.encoder = encoder

    @property
    def hierarchy(self):
        return self.encoder.hierarchy

    def window_features(self, inputs):
        maps = self.encoder(inputs)
        batch, windows = maps.shape[:2]
        return self.hierarchy.pool(maps.flatten(0, 1)).flatten(1).reshape(batch, windows, -1)

    def forward_features(self, inputs):
        return masked_window_mean(self.window_features(inputs), inputs["window_mask"])

    def forward(self, inputs):
        return self.hierarchy.classifier(self.forward_features(inputs))


def _capture_windows(model, inputs):
    captured = {}

    def hook(module, args, output):
        batch, windows = inputs["window_mask"].shape
        captured["windows"] = output.flatten(1).reshape(batch, windows, -1)

    handle = model.hierarchy.pool.register_forward_hook(hook)
    try:
        logits = model(inputs)
    finally:
        handle.remove()
    pooled = masked_window_mean(captured["windows"], inputs["window_mask"])
    # Since the trained classifier is linear, feature averaging should match
    # logit averaging to FP32 roundoff; this independently checks head/bias use.
    per_window_logits = model.hierarchy.classifier(captured["windows"])
    averaged_logits = masked_window_mean(per_window_logits, inputs["window_mask"])
    torch.testing.assert_close(logits, averaged_logits, atol=2e-5, rtol=2e-5)
    values = {"features": pooled, "window_features": captured["windows"], "logits": logits,
              "window_logits": per_window_logits, "mask": inputs["window_mask"]}
    if any(not torch.isfinite(value).all() for value in values.values()):
        raise FloatingPointError("windowing diagnostic produced nonfinite outputs")
    return values


def _metrics(logits, labels):
    values = torch.from_numpy(logits)
    targets = torch.from_numpy(labels)
    correct = int((values.argmax(dim=1) == targets).sum())
    top5 = int((values.topk(5, dim=1).indices == targets[:, None]).any(dim=1).sum())
    return {"samples": len(labels), "correct": correct, "top1": correct / len(labels),
            "top5": top5 / len(labels), "cross_entropy": float(F.cross_entropy(values, targets))}


def _paired(reference_logits, candidate_logits, reference_features, candidate_features, labels):
    ref, cand = reference_logits.argmax(axis=1), candidate_logits.argmax(axis=1)
    cosine = F.cosine_similarity(torch.from_numpy(reference_features),
                                 torch.from_numpy(candidate_features), dim=1).numpy()
    relative = np.linalg.norm(candidate_features - reference_features, axis=1) / np.maximum(
        np.linalg.norm(reference_features, axis=1), 1e-12)
    return {"prediction_agreement": int((ref == cand).sum()),
            "prediction_changes": int((ref != cand).sum()),
            "correct_to_wrong": int(((ref == labels) & (cand != labels)).sum()),
            "wrong_to_correct": int(((ref != labels) & (cand == labels)).sum()),
            "both_correct": int(((ref == labels) & (cand == labels)).sum()),
            "both_wrong": int(((ref != labels) & (cand != labels)).sum()),
            "feature_cosine_mean": float(cosine.mean()), "feature_cosine_min": float(cosine.min()),
            "feature_relative_l2_mean": float(relative.mean()),
            "logit_mean_absolute_difference": float(np.abs(candidate_logits - reference_logits).mean()),
            "logit_max_absolute_difference": float(np.abs(candidate_logits - reference_logits).max())}


def run_window_probe(config, reference_dir, output_dir, *, cpu_threads=2):
    if not 1 <= cpu_threads <= 2:
        raise ValueError("CPU diagnostic is bounded to one or two threads")
    reference_dir, output = Path(reference_dir).resolve(), Path(output_dir).resolve()
    previous = json.loads((reference_dir / "report.json").read_text())
    selection = json.loads((reference_dir / "selection.json").read_text())
    if (previous["status"] != "PASS" or previous["device"] != "cpu"
            or previous["precision"] != "float32" or previous["num_windows"] != 1
            or previous["batch_size"] != 1 or not 1 <= previous["samples"] <= 64):
        raise ValueError("requires a passed CPU FP32 K=1 bounded reference")
    if selection != previous["selection"]:
        raise ValueError("prior selection differs from its report")
    if file_sha256(reference_dir / "outputs.npz") != previous["outputs_sha256"]:
        raise ValueError("prior saved outputs changed")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(selection["seed"])
    checkpoint = Path(previous["source"]["path"])
    protected = _protected_hashes(config, checkpoint)
    for path in reference_dir.iterdir():
        if path.is_file():
            protected[str(path)] = file_sha256(path)
    _verify_hashes(protected)
    (output / "protected_before.json").write_text(json.dumps(protected, indent=2) + "\n")
    original = HierarchyV1().cpu().eval().requires_grad_(False)
    provenance = load_full_hierarchy_checkpoint(original, checkpoint)
    if provenance["sha256"] != previous["source"]["sha256"]:
        raise ValueError("checkpoint does not match preservation reference")
    for name, digest in previous["probe_source_sha256"].items():
        if file_sha256(Path(__file__).parent / name) != digest:
            raise ValueError(f"source changed since preservation probe: {name}")

    models = {}
    for mode in ("global", "local"):
        hierarchy = HierarchyV1().cpu().eval().requires_grad_(False)
        loaded = load_full_hierarchy_checkpoint(hierarchy, checkpoint)
        if loaded["sha256"] != provenance["sha256"]:
            raise ValueError("checkpoint changed while constructing models")
        # Match the reference's batch=1 convolutions for every window.
        encoder = RuntimeHierarchySequenceEncoder(hierarchy, num_windows=4,
                    hierarchy_chunk_windows=1, activation_checkpointing=False)
        models[mode] = FrozenWindowReadout(hierarchy, num_windows=4, encoder=encoder).eval()
    before = {"original": _state_digest(original),
              **{mode: _state_digest(model.hierarchy) for mode, model in models.items()}}
    if len(set(before.values())) != 1 or before["original"] != previous["model_state_sha256_before"]["original"]:
        raise ValueError("full model states differ from the prior reference")

    dataset = HierarchyDataset(config["manifest_dir"], config["dataset_root"], "validation",
              raw_cache=config.get("raw_cache"), limit=previous["samples"], seed=selection["seed"])
    sample_ids = [row.sample_id for row in dataset.rows]
    if (sample_ids != selection["sample_ids"] or dataset.manifest_sha256 != selection["manifest_sha256"]
            or [row.raw_content_sha256 for row in dataset.rows] != selection["raw_content_sha256"]):
        raise ValueError("window diagnostic must use exactly the prior 64-sample selection")
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    prior_outputs = np.load(reference_dir / "outputs.npz", allow_pickle=False)
    if not np.array_equal(prior_outputs["sample_ids"], np.asarray(sample_ids)):
        raise ValueError("saved reference sample order differs")
    chunks, records, target_values = {}, [], []
    with torch.inference_mode():
        for index, row in enumerate(dataset.rows):
            if row.split != "validation" or row.source_split != "train":
                raise ValueError("sample crossed the internal-validation/source-train boundary")
            fields = raw._decode_event_fields(dataset.payload(row), sample_id=row.sample_id)
            source = raw._source_identity(row, fields)
            metadata = {"label": row.class_label, "sample_id": row.sample_id, "split": row.split,
                        "source_split": row.source_split, "decode_seconds": 0.0, "render_seconds": 0.0}
            packed = prepare_points(fields, source)
            reference_batch = collate([{**metadata, "inputs": packed}])
            ref = _capture_forward(original, reference_batch["inputs"])
            _exact(ref["logits"], torch.from_numpy(prior_outputs["original_logits"][index:index + 1]),
                   "replayed reference logits")
            _exact(ref["pooled"], torch.from_numpy(prior_outputs["original_features"][index:index + 1]),
                   "replayed reference features")
            if int(prior_outputs["labels"][index]) != row.class_label:
                raise ValueError("reference labels differ")
            partitions = split_event_windows(fields, source, num_windows=4)
            indices = np.concatenate([part["event_indices"] for part in partitions])
            if not np.array_equal(np.sort(indices), np.arange(source.event_count)):
                raise ValueError("partitions did not conserve every original event exactly once")
            sequences = {mode: prepare_window_control(fields, source, num_windows=4, time_mode=mode)
                         for mode in models}
            _exact(sequences["local"]["window_mask"], sequences["global"]["window_mask"], "window masks")
            for position, part in enumerate(partitions):
                chosen = torch.from_numpy(part["event_indices"])
                local, global_window = (sequences[mode]["windows"][position] for mode in ("local", "global"))
                _exact(local["event_counts"], global_window["event_counts"], "window counts")
                _exact(local["points"][:, [0, 1, 3]], global_window["points"][:, [0, 1, 3]], "x y polarity")
                for key in ("points", "voxel_lower", "voxel_upper", "alpha"):
                    _exact(packed[key][chosen], global_window[key], f"global-time conservation {key}")
            chunks.setdefault("original_logits", []).append(ref["logits"].numpy().copy())
            chunks.setdefault("original_features", []).append(ref["pooled"].numpy().copy())
            target_values.append(row.class_label)
            record = {"sample_id": row.sample_id, "event_count": source.event_count,
                      "window_counts": [part["event_count"] for part in partitions],
                      "window_mask": sequences["local"]["window_mask"].tolist(),
                      "original_feature_sha256": _tensor_digest(ref["pooled"]),
                      "original_logits_sha256": _tensor_digest(ref["logits"]),
                      "event_conservation": True, "global_time_coordinates_preserved": True,
                      "time_ranges_by_mode": {}}
            for mode, model in models.items():
                sequence = sequences[mode]
                batch = sequence_collate([{**metadata, **sequence}])
                values = _capture_windows(model, batch["inputs"])
                for key, value in values.items():
                    chunks.setdefault(f"k4_{mode}_{key}", []).append(value.numpy().copy())
                record["time_ranges_by_mode"][mode] = [
                    [float(window["points"][:, 2].min()), float(window["points"][:, 2].max())]
                    if len(window["points"]) else None for window in sequence["windows"]]
            records.append(record)
            if (index + 1) % 8 == 0 or index + 1 == len(dataset):
                print(json.dumps({"event": "window_probe_progress", "samples": index + 1,
                                  "total": len(dataset), "seconds": round(time.perf_counter() - started, 2)}), flush=True)
    prior_outputs.close()
    after = {"original": _state_digest(original),
             **{mode: _state_digest(model.hierarchy) for mode, model in models.items()}}
    if before != after or any(p.grad is not None for model in [original, *models.values()] for p in model.parameters()):
        raise ValueError("zero-update diagnostic changed model state or gradients")
    if torch.cuda.is_initialized():
        raise ValueError("CPU-only diagnostic initialized CUDA")
    _verify_hashes(protected)
    arrays = {key: np.concatenate(values) for key, values in chunks.items()}
    labels = np.asarray(target_values, dtype=np.int64)
    arrays.update(labels=labels, sample_ids=np.asarray(sample_ids))
    np.savez_compressed(output / "outputs.npz", **arrays)
    metrics = {name: _metrics(arrays[name + "_logits"], labels)
               for name in ("original", "k4_global", "k4_local")}
    paired = {name: _paired(arrays["original_logits"], arrays[name + "_logits"],
                           arrays["original_features"], arrays[name + "_features"], labels)
              for name in ("k4_global", "k4_local")}
    paired["local_vs_global"] = _paired(arrays["k4_global_logits"], arrays["k4_local_logits"],
                                       arrays["k4_global_features"], arrays["k4_local_features"], labels)
    report = {"status": "PASS", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "frozen-checkpoint diagnostic of partitioning and temporal coordinates",
              "reference_dir": str(reference_dir), "reference_outputs_replayed_bit_exact": True,
              "source": provenance, "selection": selection, "samples": len(dataset),
              "device": "cpu", "precision": "float32", "cpu_threads": cpu_threads,
              "sample_batch_size": 1, "window_execution_batch_size": 1,
              "native_resolution": [480, 640], "all_events_used": True,
              "aggregation": "equal masked mean of original pooled features; original trained linear classifier",
              "new_trainable_parameters": 0, "optimizer_steps": 0, "gradients_created": False,
              "metrics": metrics, "paired_comparisons": paired, "records": records,
              "all_windows_nonempty_samples": sum(all(r["window_mask"]) for r in records),
              "model_state_sha256_before": before, "model_state_sha256_after": after,
              "model_weights_unchanged": True, "protected_files": {"checked": len(protected), "changed": []},
              "cuda_initialized": False, "final_test_rows_or_payloads_accessed": False,
              "HARDVS_accessed": False, "paired_RGB_used": False, "training_queue_modified": False,
              "limitations": ["64 validation examples, not a full validation score or causal estimate for trained runs",
                  "K1 versus K4 changes context, event-mass statistics, nonlinear encoding and aggregation",
                  "local versus global changes point time and temporal voxel routing together",
                  "global-time windows retain empty temporal bins and remain different from whole clips",
                  "no CLIP, temporal Transformer, classifier replacement or fine-tuning in this diagnostic"],
              "probe_source_sha256": {p.name: file_sha256(p) for p in Path(__file__).parent.glob("*.py")},
              "outputs_sha256": file_sha256(output / "outputs.npz"), "seconds": time.perf_counter() - started}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "report": str(output / "report.json"), "metrics": metrics,
                      "seconds": round(report["seconds"], 2)}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    run_window_probe(json.loads(args.config.read_text()), args.reference_dir, args.output_dir,
                     cpu_threads=args.cpu_threads)


if __name__ == "__main__":
    main()
