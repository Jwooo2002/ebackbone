"""Frozen whole-clip hierarchy feature classification-head study.

This module deliberately separates rendering/feature extraction, fitting and
evaluation.  It is CPU/FP32 only.  Neither ``prepare_features`` nor
``train_all`` calculates an accuracy, selects a checkpoint, or reads final
test data.  A root-owned JSON protocol supplies the selected train/validation
sample IDs and the complete, preregistered condition matrix.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .hierarchy_clip_models import FixedPromptTextBank, load_pretrained_clip
from .hierarchy_clip_runtime import _class_contract
from .hierarchy_clip_preservation import load_full_hierarchy_checkpoint
from .hierarchy_clip_staging import file_sha256
from .hierarchy_data import HierarchyDataset, collate
from .hierarchy_models import HierarchyV1


ENGINE_VERSION = "hierarchy-clip-head-study-1"
FEATURE_DIM, CLIP_DIM, CLASS_COUNT = 256, 512, 100
_INTEROP_CONFIGURED = False


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _tensor_sha(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".npz", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_hashes(checkpoint_provenance):
    identity = checkpoint_provenance.get("identity", {})
    expected = identity.get("source_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("full hierarchy checkpoint must carry source_sha256 provenance")
    root = Path(__file__).parent
    actual = {}
    for name, digest in expected.items():
        path = root / name
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"hierarchy source changed since checkpoint: {name}")
        actual[name] = digest
    return actual


def _load_protocol(path):
    path = Path(path).resolve()
    protocol = json.loads(path.read_text())
    required = {"output_dir", "manifest_dir", "dataset_root", "hierarchy_checkpoint",
                "selection_json", "class_names_path", "clip_checkpoint", "conditions"}
    missing = required - set(protocol)
    if missing:
        raise ValueError(f"protocol missing keys: {sorted(missing)}")
    if not isinstance(protocol["conditions"], list) or not protocol["conditions"]:
        raise ValueError("protocol requires a nonempty preregistered conditions list")
    if protocol.get("status") != "SEALED_BEFORE_OUTCOMES":
        raise ValueError("condition runs require a SEALED_BEFORE_OUTCOMES protocol")
    source_hashes = protocol.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("sealed protocol requires nonempty source_sha256")
    for source, digest in source_hashes.items():
        candidate = Path(__file__).parent / source
        if not candidate.is_file() or file_sha256(candidate) != digest:
            raise ValueError(f"sealed source hash mismatch: {source}")
    integrity = protocol.get("integrity_files")
    if not isinstance(integrity, dict) or not integrity:
        raise ValueError("sealed protocol requires integrity_files")
    for source, digest in integrity.items():
        if not Path(source).is_file() or file_sha256(source) != digest:
            raise ValueError(f"sealed integrity file mismatch: {source}")
    protocol["_path"] = str(path)
    protocol["_sha256"] = file_sha256(path)
    return protocol


def _selection(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    # Root protocol writers may use either explicit split keys or a splits map.
    train = value.get("train_sample_ids", value.get("splits", {}).get("train"))
    validation = value.get("validation_sample_ids", value.get("val_sample_ids",
        value.get("splits", {}).get("validation")))
    if not isinstance(train, list) or not isinstance(validation, list):
        raise ValueError("selection requires train_sample_ids and validation_sample_ids")
    selected, records = {}, {}
    for name, ids in (("train", train), ("validation", validation)):
        # The frozen protocol uses records so that the selection carries label
        # and raw-byte provenance.  Plain ID lists remain useful for tiny tests.
        if ids and isinstance(ids[0], dict):
            if any(not {"sample_id", "class_label", "raw_content_sha256"} <= set(item) for item in ids):
                raise ValueError(f"selection {name} records require sample_id/class_label/raw_content_sha256")
            records[name] = {item["sample_id"]: item for item in ids}
            ids = [item["sample_id"] for item in ids]
        else:
            records[name] = {}
        if not ids or any(not isinstance(x, str) or not x for x in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"selection {name} IDs must be nonempty and unique")
        selected[name] = ids
    return value, selected, records, file_sha256(path)


def _setup_cpu(threads):
    global _INTEROP_CONFIGURED
    if threads != 2:
        raise ValueError("this study is fixed to exactly two CPU threads")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only study refuses an already initialized CUDA runtime")
    torch.set_num_threads(2)
    if not _INTEROP_CONFIGURED:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise
        _INTEROP_CONFIGURED = True
    torch.use_deterministic_algorithms(True)


def _output(protocol):
    return Path(protocol["output_dir"]).resolve()


def _feature_identity(protocol, selection_sha, checkpoint, source_hashes):
    return {"engine_version": ENGINE_VERSION, "protocol_sha256": protocol["_sha256"],
            "selection_sha256": selection_sha, "checkpoint": checkpoint,
            "hierarchy_source_sha256": source_hashes, "device": "cpu", "precision": "float32",
            "whole_clip": True, "batch_size": 1, "feature_dim": FEATURE_DIM,
            "final_test_accessed": False, "HARDVS_accessed": False, "paired_RGB_used": False}


def _load_cache(path, identity):
    if not path.is_file() or not path.with_suffix(".json").is_file():
        return None
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta.get("identity") != identity or meta.get("npz_sha256") != file_sha256(path):
        return None
    return meta


def _selected_dataset(protocol, split, ids, expected_manifest_sha=None):
    dataset = HierarchyDataset(protocol["manifest_dir"], protocol["dataset_root"], split,
                               raw_cache=protocol.get("raw_cache"))
    if expected_manifest_sha is not None and dataset.manifest_sha256 != expected_manifest_sha:
        raise ValueError(f"selection manifest SHA mismatch for {split}")
    rows = {row.sample_id: index for index, row in enumerate(dataset.rows)}
    absent = [sample_id for sample_id in ids if sample_id not in rows]
    if absent:
        raise ValueError(f"selection contains IDs absent from {split}: {absent[:3]}")
    return dataset, [rows[sample_id] for sample_id in ids]


def _extract_split(protocol, split, ids, selected_records, model, identity, expected_manifest_sha):
    root = _output(protocol)
    chunks = root / "features" / "chunks" / split
    final = root / "features" / f"{split}.npz"
    existing = _load_cache(final, identity)
    if existing:
        data = np.load(final, allow_pickle=False)
        if list(data["sample_ids"].astype(str)) == ids:
            return final
    if final.exists() or final.with_suffix(".json").exists():
        raise ValueError(f"existing feature cache identity or content mismatch: {final}")
    dataset, indices = _selected_dataset(protocol, split, ids, expected_manifest_sha)
    chunk_size = int(protocol.get("feature_chunk_size", 32))
    if chunk_size < 1:
        raise ValueError("feature_chunk_size must be positive")
    for start in range(0, len(indices), chunk_size):
        end = min(start + chunk_size, len(indices))
        expected_ids = ids[start:end]
        path = chunks / f"{start:06d}_{end:06d}.npz"
        meta_path = path.with_suffix(".json")
        chunk_identity = {**identity, "split": split, "start": start, "end": end,
                          "sample_ids_sha256": _sha(expected_ids)}
        valid = _load_cache(path, chunk_identity)
        if valid:
            loaded = np.load(path, allow_pickle=False)
            if list(loaded["sample_ids"].astype(str)) == expected_ids:
                continue
        if path.exists() or meta_path.exists():
            raise ValueError(f"existing chunk identity or content mismatch: {path}")
        features, labels, logits, sources = [], [], [], []
        with torch.inference_mode():
            for index, expected_id in zip(indices[start:end], expected_ids):
                sample = dataset[index]
                if sample["sample_id"] != expected_id or sample["split"] != split:
                    raise ValueError("dataset selection provenance mismatch")
                expected = selected_records.get(expected_id)
                if expected:
                    row = dataset.rows[index]
                    if int(sample["label"]) != int(expected["class_label"]) or getattr(row, "raw_content_sha256", None) != expected["raw_content_sha256"]:
                        raise ValueError(f"selection label/raw-content provenance mismatch: {expected_id}")
                packed = collate([sample])
                feature = model.pool(model.forward_features(packed["inputs"])).flatten(1)
                logit = model.classifier(feature)
                if feature.shape != (1, FEATURE_DIM) or logit.shape != (1, CLASS_COUNT):
                    raise ValueError("unexpected frozen hierarchy output shape")
                features.append(feature.numpy().copy()); logits.append(logit.numpy().copy())
                labels.append(int(sample["label"])); sources.append(_canonical(sample["source"]))
        _atomic_npz(path, sample_ids=np.asarray(expected_ids), labels=np.asarray(labels, dtype=np.int64),
                    features=np.concatenate(features), original_logits=np.concatenate(logits),
                    sources=np.asarray(sources))
        _atomic_json(meta_path, {"identity": chunk_identity, "npz_sha256": file_sha256(path),
                                 "samples": len(expected_ids), "complete": True})
        print(json.dumps({"event": "feature_chunk_complete", "split": split, "start": start,
                          "end": end, "total": len(indices)}), flush=True)
    arrays = {key: [] for key in ("sample_ids", "labels", "features", "original_logits", "sources")}
    for start in range(0, len(indices), chunk_size):
        end = min(start + chunk_size, len(indices))
        item = np.load(chunks / f"{start:06d}_{end:06d}.npz", allow_pickle=False)
        for key in arrays: arrays[key].append(item[key])
    _atomic_npz(final, **{key: np.concatenate(value) for key, value in arrays.items()})
    _atomic_json(final.with_suffix(".json"), {"identity": identity, "npz_sha256": file_sha256(final),
                                                "samples": len(ids), "complete": True})
    return final


def _verify_preservation(protocol, feature_paths):
    path = protocol.get("preservation_outputs_npz")
    if not path:
        raise ValueError("protocol must require preservation_outputs_npz for exact 64-sample comparison")
    reference = np.load(Path(path).resolve(), allow_pickle=False)
    if len(reference["sample_ids"]) != 64:
        raise ValueError("preservation output must contain exactly 64 samples")
    current = np.load(feature_paths["validation"], allow_pickle=False)
    lookup = {sample_id: index for index, sample_id in enumerate(current["sample_ids"].astype(str))}
    for old_index, sample_id in enumerate(reference["sample_ids"].astype(str)):
        if sample_id not in lookup:
            raise ValueError("all 64 preservation sample IDs must be selected for validation")
        index = lookup[sample_id]
        for old, new in ((reference["original_features"][old_index], current["features"][index]),
                         (reference["original_logits"][old_index], current["original_logits"][index])):
            if old.dtype != new.dtype or old.shape != new.shape or not np.array_equal(old, new):
                raise AssertionError(f"preservation output differs for {sample_id}")


def _prepare_text_bank(protocol):
    root = _output(protocol); path = root / "features" / "clip_text_bank.npz"
    names_path = Path(protocol["class_names_path"]).resolve()
    names, classes = _class_contract(protocol)
    provenance = {"class_names_sha256": file_sha256(names_path), "clip_checkpoint_sha256": file_sha256(protocol["clip_checkpoint"]),
                  "prompt_template": protocol.get("prompt_template", "a photo of a {class_name}."), "protocol_sha256": protocol["_sha256"]}
    meta = path.with_suffix(".json")
    if path.is_file() or meta.is_file():
        if (path.is_file() and meta.is_file() and json.loads(meta.read_text()).get("identity") == provenance
                and json.loads(meta.read_text()).get("npz_sha256") == file_sha256(path)):
            return path
        raise ValueError(f"existing text bank identity mismatch: {path}")
    pretrained = load_pretrained_clip(protocol["clip_checkpoint"], source_dir=protocol.get("clip_source_dir"))
    bank = FixedPromptTextBank(pretrained, names, provenance["prompt_template"])
    vectors = bank.embeddings.detach().cpu().numpy()
    if vectors.shape != (CLASS_COUNT, CLIP_DIM): raise ValueError("CLIP text bank must be [100,512]")
    _atomic_npz(path, embeddings=vectors.astype(np.float32), class_names=np.asarray(names), prompts=np.asarray(bank.prompts))
    _atomic_json(meta, {"identity": provenance, "clip": bank.provenance, "classes": classes, "contract": bank.contract(),
                        "npz_sha256": file_sha256(path), "frozen": True})
    return path


def _target_array(condition, clip_bank):
    kind, seed = condition["kind"], int(condition["bank_seed"])
    if kind in ("learned", "random"):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return F.normalize(torch.randn(CLASS_COUNT, CLIP_DIM, generator=generator), dim=1).numpy(), np.empty(0, np.int64)
    if kind == "clip":
        return np.asarray(clip_bank, dtype=np.float32), np.empty(0, np.int64)
    if kind == "permuted_clip":
        order = torch.randperm(CLASS_COUNT, generator=torch.Generator(device="cpu").manual_seed(seed)).numpy()
        return np.asarray(clip_bank, dtype=np.float32)[order], order
    raise ValueError(f"unknown condition kind: {kind}")


def _prepare_target_banks(protocol, text_path):
    root = _output(protocol) / "features" / "target_banks"; clip = np.load(text_path, allow_pickle=False)["embeddings"]
    outputs = {}
    for condition in protocol["conditions"]:
        path, meta = root / f"{condition['id']}.npz", root / f"{condition['id']}.json"
        values, permutation = _target_array(condition, clip)
        if values.shape != (CLASS_COUNT, CLIP_DIM) or not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values, axis=1), 1., rtol=0, atol=1e-6):
            raise ValueError("invalid normalized exported target bank")
        identity = {"protocol_sha256": protocol["_sha256"], "condition": condition, "text_bank_sha256": file_sha256(text_path)}
        if path.is_file() or meta.is_file():
            if not path.is_file() or not meta.is_file(): raise ValueError(f"partial target-bank artifact: {path}")
            old = json.loads(meta.read_text())
            if old.get("identity") != identity or old.get("npz_sha256") != file_sha256(path): raise ValueError(f"target-bank identity mismatch: {path}")
            outputs[condition["id"]] = {"path": str(path), "sha256": file_sha256(path)}; continue
        singular = np.linalg.svd(values, compute_uv=False)
        _atomic_npz(path, prototypes=values.astype(np.float32), permutation=permutation)
        _atomic_json(meta, {"identity": identity, "npz_sha256": file_sha256(path), "frozen": True,
                            "prototype_sha256": hashlib.sha256(values.astype(np.float32).tobytes()).hexdigest(),
                            "permutation": permutation.tolist(), "gram_sha256": hashlib.sha256((values @ values.T).astype(np.float32).tobytes()).hexdigest(),
                            "singular_values": singular.tolist(), "rank": int(np.linalg.matrix_rank(values))})
        outputs[condition["id"]] = {"path": str(path), "sha256": file_sha256(path)}
    return outputs


class CosineHead(nn.Module):
    def __init__(self, prototypes, *, learned, projector_seed):
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(int(projector_seed))
        self.projector = nn.Linear(FEATURE_DIM, CLIP_DIM, bias=True)
        with torch.no_grad():
            self.projector.weight.copy_(torch.randn(self.projector.weight.shape, generator=generator) * 0.02)
            self.projector.bias.zero_()
        prototype = F.normalize(torch.as_tensor(prototypes, dtype=torch.float32), dim=1)
        if learned: self.prototypes = nn.Parameter(prototype.clone())
        else: self.register_buffer("prototypes", prototype)
        self.learned = learned
        # This is deliberately a parameter (shared optimization treatment for
        # every cosine condition), initialized to log(10), not a forward clamp.
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(10.0))))

    def forward(self, features):
        return self.logit_scale.exp() * F.normalize(self.projector(features), dim=1) @ F.normalize(self.prototypes, dim=1).t()


def _condition(condition, bank, exported=None):
    needed = {"id", "kind", "optimization_seed", "bank_seed"}
    if missing := needed - set(condition): raise ValueError(f"condition missing {sorted(missing)}")
    kind = condition["kind"]
    if exported is not None:
        prototype = np.asarray(exported, dtype=np.float32)
        head = CosineHead(prototype, learned=kind == "learned", projector_seed=condition["optimization_seed"])
        permutation = None
        head.prototype_provenance = {"kind": kind, "bank_seed": int(condition["bank_seed"]), "permutation": permutation,
                                     "exported": True, "prototype_sha256": hashlib.sha256(prototype.tobytes()).hexdigest()}
        return head
    seed = int(condition["bank_seed"])
    if kind in ("learned", "random"):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        random = F.normalize(torch.randn(CLASS_COUNT, CLIP_DIM, generator=generator), dim=1).numpy()
        head = CosineHead(random, learned=kind == "learned", projector_seed=condition["optimization_seed"])
        head.prototype_provenance = {"kind": kind, "bank_seed": seed, "permutation": None}
        return head
    if kind == "clip":
        head = CosineHead(bank, learned=False, projector_seed=condition["optimization_seed"])
        head.prototype_provenance = {"kind": kind, "bank_seed": seed, "permutation": None}
        return head
    if kind == "permuted_clip":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        order = torch.randperm(CLASS_COUNT, generator=generator).numpy()
        head = CosineHead(bank[order], learned=False, projector_seed=condition["optimization_seed"])
        head.prototype_provenance = {"kind": kind, "bank_seed": seed, "permutation": order.tolist()}
        return head
    raise ValueError(f"unknown condition kind: {kind}")


def _fit_identity(protocol, condition, feature_meta, text_sha, target_sha=None):
    training = protocol["training"]
    return {"engine_version": ENGINE_VERSION, "protocol_sha256": protocol["_sha256"], "condition": condition,
            "feature_npz_sha256": feature_meta, "text_bank_sha256": text_sha, "target_bank_sha256": target_sha, "epochs": training["epochs"],
            "optimizer": training,
            "checkpoint_selection": "none_final_epoch_only"}


def _fit(protocol, condition, feature_paths, text_path, target_path=None):
    root = _output(protocol); target = root / "fits" / condition["id"] / "final.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    train = np.load(feature_paths["train"], allow_pickle=False)
    bank = np.load(text_path, allow_pickle=False)["embeddings"]
    # Training does not open, hash, or otherwise inspect the validation cache.
    feature_meta = {"train": file_sha256(feature_paths["train"])}
    identity = _fit_identity(protocol, condition, feature_meta, file_sha256(text_path),
                             file_sha256(target_path) if target_path is not None else None)
    if target.is_file():
        old = torch.load(target, map_location="cpu", weights_only=False)
        if (old.get("identity") == identity and old.get("complete") is True
                and old.get("final_epoch") == protocol["training"]["epochs"]
                and len(old.get("train_history", [])) == protocol["training"]["epochs"]):
            return target
        raise ValueError(f"existing fit identity mismatch: {target}")
    training = protocol["training"]
    if training["epochs"] != 50 or training["batch_size"] != 256 or training["learning_rate"] != 1e-3 or training["weight_decay"] != 1e-4:
        raise ValueError("protocol training contract differs from preregistered 50-epoch head study")
    exported_data = None if target_path is None else np.load(target_path, allow_pickle=False)
    exported = None if exported_data is None else exported_data["prototypes"]
    model = _condition(condition, bank, exported=exported).float().train()
    if exported_data is not None:
        model.prototype_provenance["permutation"] = exported_data["permutation"].astype(int).tolist()
    if isinstance(model, CosineHead):
        with torch.no_grad(): model.logit_scale.fill_(float(np.log(training["logit_scale_initial"])))
    initial_projector_digest = _tensor_sha(model.projector.weight)
    initial_prototype_digest = _tensor_sha(model.prototypes)
    ordinary = [parameter for name, parameter in model.named_parameters() if name != "logit_scale"]
    groups = [{"params": ordinary, "weight_decay": 1e-4}]
    groups.append({"params": [model.logit_scale], "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(groups, lr=training["learning_rate"], betas=tuple(training["betas"]), eps=training["eps"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training["epochs"])
    x, y = torch.from_numpy(train["features"]).float(), torch.from_numpy(train["labels"]).long()
    history = []; order_digest = hashlib.sha256(); generator = torch.Generator(device="cpu").manual_seed(int(condition["optimization_seed"]))
    lower, upper, bound_hits = float(np.log(training["logit_scale_min"])), float(np.log(training["logit_scale_max"])), 0
    print(json.dumps({"event": "fit_begin", "condition": condition["id"]}), flush=True)
    for epoch in range(training["epochs"]):
        order = torch.randperm(len(y), generator=generator); order_digest.update(order.numpy().tobytes()); total = 0.0
        for start in range(0, len(y), training["batch_size"]):
            take = order[start:start + training["batch_size"]]; loss = F.cross_entropy(model(x[take]), y[take])
            if not torch.isfinite(loss): raise FloatingPointError("nonfinite training loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                raise FloatingPointError("nonfinite head gradient")
            optimizer.step()
            if isinstance(model, CosineHead):
                with torch.no_grad():
                    before = float(model.logit_scale)
                    model.logit_scale.clamp_(lower, upper)
                    bound_hits += int(before != float(model.logit_scale))
            total += float(loss.detach()) * len(take)
        history.append({"epoch": epoch + 1, "train_cross_entropy": total / len(y)})
        scheduler.step()
    payload = {"identity": identity, "complete": True, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "train_history": history, "final_epoch": training["epochs"], "metrics_calculated": ["train_cross_entropy_only"],
                "diagnostics": ({"final_logit_scale": float(model.logit_scale.exp()), "logit_scale_bound_hits": bound_hits}
                                if isinstance(model, CosineHead) else {}),
                "pairing": {"projector_initial_digest": initial_projector_digest,
                            "batch_order_digest": order_digest.hexdigest(),
                            "prototype_initial_digest": initial_prototype_digest,
                            "prototype_final_digest": _tensor_sha(model.prototypes),
                            "prototype_provenance": model.prototype_provenance},
                "created_at_utc": datetime.now(timezone.utc).isoformat()}
    with tempfile.NamedTemporaryFile(suffix=".pt", dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary); os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"event": "fit_complete", "condition": condition["id"]}), flush=True)
    return target


def _metrics(logits, labels):
    order = np.argsort(-logits, axis=1); top1 = float((order[:, 0] == labels).mean())
    return {"samples": int(len(labels)), "top1": top1, "top5": float(np.any(order[:, :5] == labels[:, None], axis=1).mean()),
            "nll": float(F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(labels)).item())}


def prepare_features(protocol):
    _setup_cpu(int(protocol.get("cpu_threads", 2)))
    selection_value, selected, selected_records, selection_sha = _selection(protocol["selection_json"])
    hierarchy = HierarchyV1().cpu().eval().requires_grad_(False)
    checkpoint = load_full_hierarchy_checkpoint(hierarchy, protocol["hierarchy_checkpoint"])
    source_hashes = _source_hashes(checkpoint)
    identity = _feature_identity(protocol, selection_sha, checkpoint, source_hashes)
    manifest_sha = selection_value.get("manifest_sha256")
    if not isinstance(manifest_sha, dict) or set(manifest_sha) != {"train", "validation"}:
        raise ValueError("selection must declare train and validation manifest_sha256 values")
    paths = {split: _extract_split(protocol, split, ids, selected_records[split], hierarchy, {**identity, "split": split}, manifest_sha[split]) for split, ids in selected.items()}
    _verify_preservation(protocol, paths)
    text = _prepare_text_bank(protocol)
    target_banks = _prepare_target_banks(protocol, text)
    _atomic_json(_output(protocol) / "features" / "prepared.json", {"status": "COMPLETE", "identity": identity,
        "selection": {"sha256": selection_sha, "counts": {k: len(v) for k, v in selected.items()}},
        "features": {k: {"path": str(v), "sha256": file_sha256(v)} for k, v in paths.items()},
        "clip_text_bank": {"path": str(text), "sha256": file_sha256(text)}, "target_banks": target_banks, "preservation_64_exact": True,
        "accuracy_calculated": False, "final_test_accessed": False})
    return paths


def train_all(protocol):
    _setup_cpu(int(protocol.get("cpu_threads", 2)))
    prepared = _output(protocol) / "features" / "prepared.json"
    if not prepared.is_file() or json.loads(prepared.read_text()).get("status") != "COMPLETE":
        raise ValueError("prepare_features must complete before fitting")
    data = json.loads(prepared.read_text())
    if data.get("identity", {}).get("protocol_sha256") != protocol["_sha256"]:
        raise ValueError("prepared feature protocol identity mismatch")
    paths = {k: Path(v["path"]) for k, v in data["features"].items()}
    text = Path(data["clip_text_bank"]["path"])
    if not paths.get("train", Path()).is_file() or not text.is_file(): raise ValueError("prepared training artifact missing")
    if file_sha256(paths["train"]) != data["features"]["train"].get("sha256") or file_sha256(text) != data["clip_text_bank"].get("sha256"):
        raise ValueError("prepared train/text artifact hash mismatch")
    train_meta = _load_cache(paths["train"], {**data["identity"], "split": "train"})
    if train_meta is None:
        raise ValueError("prepared train cache identity mismatch")
    text_meta = paths.get("train").parent / "clip_text_bank.json"
    if not text_meta.is_file() or json.loads(text_meta.read_text()).get("npz_sha256") != file_sha256(text):
        raise ValueError("prepared text-bank metadata mismatch")
    targets = data.get("target_banks", {})
    if set(targets) != set(item["id"] for item in protocol["conditions"]):
        raise ValueError("prepared target-bank matrix mismatch")
    for condition in protocol["conditions"]:
        item = targets[condition["id"]]; target = Path(item["path"])
        if not target.is_file() or file_sha256(target) != item.get("sha256"):
            raise ValueError(f"prepared target-bank hash mismatch: {condition['id']}")
    ids = [item.get("id") for item in protocol["conditions"]]
    if len(ids) != len(set(ids)) or any(not isinstance(x, str) or not x for x in ids): raise ValueError("condition IDs must be unique")
    outputs = [_fit(protocol, condition, paths, text, Path(targets[condition["id"]]["path"])) for condition in protocol["conditions"]]
    _atomic_json(_output(protocol) / "fits" / "complete.json", {"status": "COMPLETE", "protocol_sha256": protocol["_sha256"],
        "fit_checkpoints": {c["id"]: {"path": str(p), "sha256": file_sha256(p)} for c, p in zip(protocol["conditions"], outputs)},
        "all_finals_complete": True, "validation_evaluated": False})
    return outputs


def evaluate(protocol):
    _setup_cpu(int(protocol.get("cpu_threads", 2)))
    complete = _output(protocol) / "fits" / "complete.json"
    if not complete.is_file():
        raise ValueError("evaluate refuses until every preregistered final fit is complete")
    completion = json.loads(complete.read_text())
    expected_ids = {condition["id"] for condition in protocol["conditions"]}
    if (completion.get("all_finals_complete") is not True or completion.get("protocol_sha256") != protocol["_sha256"]
            or set(completion.get("fit_checkpoints", {})) != expected_ids):
        raise ValueError("completion manifest does not validate the exact preregistered matrix")
    prepared_path = _output(protocol) / "features" / "prepared.json"
    if not prepared_path.is_file():
        raise ValueError("prepared feature manifest missing")
    prepared = json.loads(prepared_path.read_text())
    if prepared.get("status") != "COMPLETE" or prepared.get("identity", {}).get("protocol_sha256") != protocol["_sha256"]:
        raise ValueError("prepared feature manifest identity mismatch")
    paths = {k: Path(v["path"]) for k, v in prepared["features"].items()}; text = Path(prepared["clip_text_bank"]["path"])
    if set(paths) != {"train", "validation"} or not text.is_file() or any(not path.is_file() for path in paths.values()):
        raise ValueError("prepared cache artifacts missing")
    if file_sha256(text) != prepared["clip_text_bank"].get("sha256") or any(file_sha256(paths[key]) != prepared["features"][key].get("sha256") for key in paths):
        raise ValueError("prepared cache artifact hash mismatch")
    targets = prepared.get("target_banks", {})
    if set(targets) != expected_ids:
        raise ValueError("prepared target-bank matrix mismatch")
    if any(not Path(item.get("path", "")).is_file() or file_sha256(item["path"]) != item.get("sha256") for item in targets.values()):
        raise ValueError("prepared target-bank hash mismatch")
    feature_hashes = {"train": file_sha256(paths["train"])}
    text_sha = file_sha256(text)
    # This entire loop is validation-blind: no validation NPZ is opened and no
    # metric is calculated until every final checkpoint has been verified.
    for condition in protocol["conditions"]:
        checkpoint = _output(protocol) / "fits" / condition["id"] / "final.pt"
        recorded = completion["fit_checkpoints"][condition["id"]]
        if Path(recorded.get("path", "")).resolve() != checkpoint.resolve() or not checkpoint.is_file() or file_sha256(checkpoint) != recorded.get("sha256"):
            raise ValueError(f"final checkpoint manifest/hash mismatch: {condition['id']}")
        identity = _fit_identity(protocol, condition, feature_hashes, text_sha, targets[condition["id"]]["sha256"])
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (state.get("identity") != identity or state.get("complete") is not True
                or state.get("final_epoch") != protocol["training"]["epochs"]
                or len(state.get("train_history", [])) != protocol["training"]["epochs"]):
            raise ValueError(f"invalid final fit: {condition['id']}")
    bank = np.load(text, allow_pickle=False)["embeddings"]
    data = {key: np.load(path, allow_pickle=False) for key, path in paths.items()}
    rows = []; prediction_arrays = {}
    # The original trained CE head is intentionally evaluated only after the
    # full matrix gate above succeeds.
    for split, value in data.items():
        logits = value["original_logits"].astype(np.float32); labels = value["labels"]
        rows.append({"condition_id": "original_trained_head_reference", "kind": "original", "split": split, **_metrics(logits, labels)})
        prediction_arrays[f"original_trained_head_reference__{split}__logits"] = logits; prediction_arrays[f"original_trained_head_reference__{split}__labels"] = labels
    for condition in protocol["conditions"]:
        checkpoint = _output(protocol) / "fits" / condition["id"] / "final.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = _condition(condition, bank).float().eval(); model.load_state_dict(state["model"], strict=True)
        with torch.inference_mode():
            for split, value in data.items():
                logits = model(torch.from_numpy(value["features"]).float()).numpy(); labels = value["labels"]
                rows.append({"condition_id": condition["id"], "kind": condition["kind"], "split": split, **_metrics(logits, labels)})
                prediction_arrays[f"{condition['id']}__{split}__logits"] = logits; prediction_arrays[f"{condition['id']}__{split}__labels"] = labels; prediction_arrays[f"{condition['id']}__{split}__sample_ids"] = value["sample_ids"]
    evaluation = _output(protocol) / "evaluation"; evaluation.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", dir=evaluation, delete=False) as stream:
        writer = csv.DictWriter(stream, fieldnames=["condition_id", "kind", "split", "samples", "top1", "top5", "nll"]); writer.writeheader(); writer.writerows(rows); temporary = Path(stream.name)
    os.replace(temporary, evaluation / "all_rows.csv")
    _atomic_npz(evaluation / "predictions.npz", **prediction_arrays)
    _atomic_json(evaluation / "report.json", {"status": "COMPLETE", "protocol_sha256": protocol["_sha256"], "rows": len(rows),
        "all_matrix_finals_validated": True, "final_test_accessed": False, "metrics": ["top1", "top5", "nll"]})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare_features", "train_all", "evaluate"))
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args(argv); protocol = _load_protocol(args.protocol)
    result = {"prepare_features": prepare_features, "train_all": train_all, "evaluate": evaluate}[args.phase](protocol)
    print(json.dumps({"status": "COMPLETE", "phase": args.phase, "result": str(result)}, sort_keys=True))


if __name__ == "__main__": main()
