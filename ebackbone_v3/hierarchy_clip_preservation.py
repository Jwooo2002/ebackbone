"""Bounded, CPU-only check that a K=1 wrapper preserves the full hierarchy.

This diagnostic has no optimizer, training entry point, or GPU execution path.
The original classifier is mandatory; backbone-only transfer is insufficient.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn

from .hierarchy_data import HierarchyDataset, POINT_CONTRACT_SHA256, collate
from .hierarchy_models import HierarchyV1
from .hierarchy_clip_data import HierarchySequenceDataset, sequence_collate
from .hierarchy_clip_runtime import RuntimeHierarchySequenceEncoder
from .hierarchy_clip_staging import file_sha256
from .hierarchy_clip_temporal import HierarchySequenceEncoder


class PreservedHierarchyReadout(nn.Module):
    """K=1 sequence interface with the existing spatial pool and trained head."""

    def __init__(self, hierarchy, encoder=None):
        super().__init__()
        if encoder is None:
            encoder = HierarchySequenceEncoder(hierarchy, num_windows=1)
        if encoder.num_windows != 1 or encoder.hierarchy is not hierarchy:
            raise ValueError("encoder must use the same hierarchy and num_windows=1")
        self.encoder = encoder

    @property
    def hierarchy(self):
        return self.encoder.hierarchy

    def forward_features(self, inputs):
        # The original hierarchy does not define standalone empty samples.
        # Do not manufacture a claimed preservation guarantee for those inputs.
        if not bool(inputs["window_mask"].all()):
            raise ValueError("checkpoint preservation requires nonempty samples; empty mask found")
        return self.encoder(inputs)[:, 0]

    def forward(self, inputs):
        features = self.forward_features(inputs)
        return self.hierarchy.classifier(self.hierarchy.pool(features).flatten(1))


def load_full_hierarchy_checkpoint(hierarchy, path):
    """Strictly restore a trusted local full checkpoint, including its CE head."""
    path = Path(path).resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    identity = checkpoint.get("identity", {})
    variant = checkpoint.get("variant", identity.get("variant"))
    if variant != "hierarchy":
        raise ValueError("checkpoint must identify the hierarchy-only variant")
    contract = identity.get("point_contract_sha256")
    if contract is not None and contract != POINT_CONTRACT_SHA256:
        raise ValueError("hierarchy point contract mismatch")
    state = checkpoint.get("model")
    if not isinstance(state, dict) or not state:
        raise ValueError("full model checkpoint required, including classifier")
    prefixed = [key.startswith("module.") for key in state]
    if any(prefixed) and not all(prefixed):
        raise ValueError("mixed DDP prefixes in full model checkpoint")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    expected = hierarchy.state_dict()
    if set(state) != set(expected):
        raise ValueError("full model keys mismatch (classifier is mandatory): "
                         f"missing={sorted(set(expected) - set(state))}, "
                         f"unexpected={sorted(set(state) - set(expected))}")
    for key, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.shape != expected[key].shape
                or value.dtype != expected[key].dtype or not torch.isfinite(value).all()):
            raise ValueError(f"invalid full model tensor: {key}")
    hierarchy.load_state_dict(state, strict=True)
    if any(not torch.equal(value.cpu(), state[key]) for key, value in hierarchy.state_dict().items()):
        raise ValueError("full checkpoint reload was not bit-exact")
    return {"path": str(path), "sha256": file_sha256(path), "epoch": checkpoint.get("epoch"),
            "variant": variant, "strict_full_model": True, "classifier_transferred": True,
            "model_tensors": len(state), "identity": identity,
            "historical_best": checkpoint.get("best")}


def _tensor_digest(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(value.dtype), list(value.shape))).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state_digest(model):
    return hashlib.sha256(json.dumps({key: _tensor_digest(value)
                                    for key, value in model.state_dict().items()},
                                   sort_keys=True).encode()).hexdigest()


def _exact(reference, candidate, label):
    if (reference.dtype != candidate.dtype or reference.shape != candidate.shape
            or not torch.equal(reference, candidate)
            or _tensor_digest(reference) != _tensor_digest(candidate)):
        error = None
        if reference.shape == candidate.shape and reference.numel():
            error = float((reference.double() - candidate.double()).abs().max())
        raise AssertionError(f"{label} differs; maximum absolute error={error}")


def _capture_forward(model, inputs):
    captured = {}
    hierarchy = model if isinstance(model, HierarchyV1) else model.hierarchy
    feature_module = hierarchy.frame_stage if isinstance(model, HierarchyV1) else model.encoder

    def capture_map(module, args, value):
        captured["maps"] = value if isinstance(model, HierarchyV1) else value[:, 0]

    def capture_pool(module, args, value):
        captured["pooled"] = value.flatten(1)

    handles = [feature_module.register_forward_hook(capture_map),
               hierarchy.pool.register_forward_hook(capture_pool)]
    try:
        captured["logits"] = model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    captured["predictions"] = captured["logits"].argmax(dim=-1)
    if any(not torch.isfinite(value).all() for value in captured.values()):
        raise FloatingPointError("nonfinite feature map, pooled feature, or logit")
    return captured


def _protected_hashes(config, checkpoint_path):
    study = Path(config["study_root"])
    expected = json.loads((study / "protected_before.json").read_text())
    for relative, digest in json.loads((study / "snapshot_hashes.json").read_text()).items():
        expected[str(study / "snapshot" / relative)] = digest
    for name in ("config.json", "training_gate.json", "snapshot_hashes.json", "protected_before.json"):
        path = study / name
        expected[str(path)] = file_sha256(path)
    expected[str(checkpoint_path)] = file_sha256(checkpoint_path)
    return expected


def _verify_hashes(expected):
    changed = [path for path, digest in expected.items() if file_sha256(path) != digest]
    if changed:
        raise ValueError(f"protected files changed: {changed}")


def run_preservation_probe(config, checkpoint_path, output_dir, *, limit=64,
                           batch_size=1, cpu_threads=2, seed=20260922):
    if not 1 <= limit <= 64 or not 1 <= batch_size <= 2 or not 1 <= cpu_threads <= 2:
        raise ValueError("bounded probe requires 1..64 samples, batch 1..2, CPU threads 1..2")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    checkpoint_path = Path(checkpoint_path).resolve()
    protected = _protected_hashes(config, checkpoint_path)
    _verify_hashes(protected)
    (output / "protected_before.json").write_text(json.dumps(protected, indent=2) + "\n")

    reference = HierarchyV1().cpu().eval().requires_grad_(False)
    provenance = load_full_hierarchy_checkpoint(reference, checkpoint_path)
    source_dir = Path(__file__).parent
    original_sources = {}
    for name in ("__init__.py", "hierarchy_models.py", "hierarchy_data.py", "v1_models.py", "v1_data.py",
                 "representations.py", "n_imagenet_mini_dataset.py", "n_imagenet_mini_index.py",
                 "contracts.py", "splits.py", "errors.py"):
        digest = file_sha256(source_dir / name)
        if digest != provenance["identity"]["source_sha256"][name]:
            raise ValueError(f"original checkpoint implementation changed: {name}")
        original_sources[name] = digest

    wrappers = {}
    for name in ("sequence_k1", "runtime_k1"):
        hierarchy = HierarchyV1().cpu().eval().requires_grad_(False)
        loaded = load_full_hierarchy_checkpoint(hierarchy, checkpoint_path)
        if loaded["sha256"] != provenance["sha256"]:
            raise ValueError("checkpoint changed during independent model construction")
        encoder = None if name == "sequence_k1" else RuntimeHierarchySequenceEncoder(
            hierarchy, num_windows=1, hierarchy_chunk_windows=16, activation_checkpointing=True)
        wrappers[name] = PreservedHierarchyReadout(hierarchy, encoder=encoder).eval()
    state_before = {"original": _state_digest(reference),
                    **{name: _state_digest(model.hierarchy) for name, model in wrappers.items()}}
    if len(set(state_before.values())) != 1:
        raise ValueError("independently loaded models have different states")

    arguments = dict(manifest_dir=config["manifest_dir"], dataset_root=config["dataset_root"],
                     split="validation", raw_cache=config.get("raw_cache"), limit=limit, seed=seed)
    original_data = HierarchyDataset(**arguments)
    sequence_data = HierarchySequenceDataset(**arguments, num_windows=1)
    manifest_sha = provenance["identity"]["validation_manifest_sha256"]
    if original_data.manifest_sha256 != manifest_sha or sequence_data.manifest_sha256 != manifest_sha:
        raise ValueError("validation manifest does not match pretrained checkpoint")
    sample_ids = [row.sample_id for row in original_data.rows]
    if sample_ids != [row.sample_id for row in sequence_data.rows] or len(set(sample_ids)) != limit:
        raise ValueError("diagnostic sample identities differ or repeat")
    selection = {"split": "validation", "source_split": "train", "seed": seed,
                 "rule": "first 64 (or requested limit) by SHA256(v1-diagnostic + NUL + seed + NUL + sample_id)",
                 "label_independent": True, "manifest_sha256": manifest_sha, "sample_ids": sample_ids,
                 "ordered_sample_ids_sha256": hashlib.sha256(json.dumps(sample_ids).encode()).hexdigest(),
                 "raw_content_sha256": [row.raw_content_sha256 for row in original_data.rows]}
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")

    records, logits, features, labels = [], {name: [] for name in ("original", *wrappers)}, {}, []
    with torch.inference_mode():
        for start in range(0, limit, batch_size):
            indices = range(start, min(start + batch_size, limit))
            original_samples = [original_data[index] for index in indices]
            sequence_samples = [sequence_data[index] for index in indices]
            for left, right in zip(original_samples, sequence_samples):
                for key in ("sample_id", "label", "source", "split", "source_split"):
                    if left[key] != right[key]:
                        raise ValueError(f"sample identity mismatch: {key}")
                if left["split"] != "validation" or left["source_split"] != "train":
                    raise ValueError("probe crossed internal-validation/source-train boundary")
            original = collate(original_samples)
            sequence = sequence_collate(sequence_samples)
            _exact(original["labels"], sequence["labels"], "labels")
            for key, value in original["inputs"].items():
                _exact(value, sequence["inputs"]["packed"][key], f"packed {key}")
            ref = _capture_forward(reference, original["inputs"])
            outcomes = {"original": ref}
            for name, wrapper in wrappers.items():
                outcomes[name] = candidate = _capture_forward(wrapper, sequence["inputs"])
                for key in ("maps", "pooled", "logits", "predictions"):
                    _exact(ref[key], candidate[key], f"{name} {key}")
            for name, values in outcomes.items():
                logits[name].append(values["logits"].numpy().copy())
                features.setdefault(name, []).append(values["pooled"].numpy().copy())
            labels.append(original["labels"].numpy().copy())
            records.append({"sample_ids": original["sample_ids"],
                            "event_counts": original["inputs"]["event_counts"].tolist(),
                            "sources": [sample["source"] for sample in original_samples],
                            "input_shapes": {k: list(v.shape) for k, v in original["inputs"].items()},
                            "input_dtypes": {k: str(v.dtype) for k, v in original["inputs"].items()},
                            "point_channel_min": original["inputs"]["points"].amin(dim=0).tolist(),
                            "point_channel_max": original["inputs"]["points"].amax(dim=0).tolist(),
                            "feature_shape": list(ref["maps"].shape), "logit_shape": list(ref["logits"].shape),
                            "input_sha256": {k: _tensor_digest(v) for k, v in original["inputs"].items()},
                            "feature_sha256": _tensor_digest(ref["maps"]),
                            "logits_sha256": _tensor_digest(ref["logits"]),
                            "all_three_paths_bit_exact": True})
            if (start + len(original_samples)) % 8 == 0 or start + len(original_samples) == limit:
                print(json.dumps({"event": "preservation_progress", "samples": start + len(original_samples),
                                  "total": limit, "seconds": round(time.perf_counter() - started, 2)}), flush=True)

    state_after = {"original": _state_digest(reference),
                   **{name: _state_digest(model.hierarchy) for name, model in wrappers.items()}}
    if state_before != state_after or any(p.grad is not None for model in [reference, *wrappers.values()]
                                        for p in model.parameters()):
        raise ValueError("inference probe altered model state or created gradients")
    _verify_hashes(protected)
    if torch.cuda.is_initialized():
        raise ValueError("CPU-only probe unexpectedly initialized CUDA")
    labels_array = np.concatenate(labels)
    arrays = {"labels": labels_array, "sample_ids": np.asarray(sample_ids)}
    for name in logits:
        arrays[name + "_logits"] = np.concatenate(logits[name])
        arrays[name + "_features"] = np.concatenate(features[name])
    np.savez_compressed(output / "outputs.npz", **arrays)
    correct = int((arrays["original_logits"].argmax(axis=1) == labels_array).sum())
    report = {"status": "PASS", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "full hierarchy checkpoint behavior preservation, not architecture accuracy evaluation",
              "device": "cpu", "precision": "float32", "cpu_threads": cpu_threads,
              "samples": limit, "batch_size": batch_size, "num_windows": 1,
              "native_resolution": [480, 640], "all_events_used": True, "source": provenance,
              "original_source_hashes_match_checkpoint": original_sources,
              "probe_source_sha256": {p.name: file_sha256(p) for p in source_dir.glob("*.py")},
              "selection": selection, "comparisons": {name: {
                  "packed_inputs_bit_exact": True, "feature_maps_bit_exact": True,
                  "pooled_features_bit_exact": True, "logits_bit_exact": True,
                  "predictions_identical": limit, "max_absolute_error": 0.0} for name in wrappers},
              "subset_accuracy": {"correct": correct, "total": limit, "top1": correct / limit,
                                  "scope": "diagnostic subset only; not a new 5000-sample validation result"},
              "model_state_sha256_before": state_before, "model_state_sha256_after": state_after,
              "model_weights_unchanged": True, "optimizer_steps": 0, "gradients_created": False,
              "cuda_initialized": False, "final_test_accessed": False, "HARDVS_accessed": False,
              "paired_RGB_used": False, "training_queue_modified": False,
              "protected_files": {"checked": len(protected), "changed": []},
              "batches": records, "seconds": time.perf_counter() - started,
              "outputs_sha256": file_sha256(output / "outputs.npz")}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "samples": limit, "report": str(output / "report.json"),
                      "seconds": round(report["seconds"], 2)}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    run_preservation_probe(json.loads(args.config.read_text()), args.checkpoint, args.output_dir,
                           limit=args.limit, batch_size=args.batch_size,
                           cpu_threads=args.cpu_threads, seed=args.seed)


if __name__ == "__main__":
    main()
