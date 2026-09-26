"""Bounded comparison verification only: one update per stage, no train mode."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

import torch
from torch import nn

from .hierarchy_clip_data import HierarchySequenceDataset, sequence_collate
from .hierarchy_clip_models import HierarchyTextAlignment, HierarchyCLIPViTAlignment, load_pretrained_clip
from .hierarchy_clip_temporal import HierarchySequenceEncoder, HierarchyTemporalBaseline
from .hierarchy_clip_staging import (configure_stage, file_sha256, gradient_report,
    load_hierarchy_backbone, load_comparison_checkpoint, save_comparison_checkpoint, set_staged_train)
from .hierarchy_models import HierarchyV1


VARIANTS = ("hierarchy_temporal", "hierarchy_text", "hierarchy_clip_vit")


def frozen_fingerprint(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def evaluation_logits(model, inputs):
    """Exact reload comparisons must not measure CUDA scatter ordering noise."""
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with torch.no_grad():
            return model(inputs).detach().cpu()
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def move_inputs(inputs, device):
    return {"packed": {k: v.to(device) for k, v in inputs["packed"].items()},
            "window_mask": inputs["window_mask"].to(device)}


def run_probe(config, output_dir, *, device="cpu"):
    if device != "cpu" and device not in ("cuda:0", "cuda:1"):
        raise ValueError("bounded device must be cpu, cuda:0 or cuda:1")
    gate = None
    if device.startswith("cuda"):
        # This precedes every CUDA availability query, allocation or model move.
        from .hierarchy_clip_gate import assert_gpu_ready
        gate = assert_gpu_ready(config["wait_for_study"])
        expected_devices = ",".join(gate["expected_gpu_uuids"])
        if os.environ.get("CUDA_VISIBLE_DEVICES", expected_devices) != expected_devices:
            raise ValueError("CUDA_VISIBLE_DEVICES must match both study UUIDs in order")
        os.environ["CUDA_VISIBLE_DEVICES"] = expected_devices
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite probe results: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    windows = config.get("num_windows", 4)
    if not 1 <= windows <= 4:
        raise ValueError("bounded tests permit 1-4 windows")
    seed = config.get("seed", 20260908)
    datasets = {split: HierarchySequenceDataset(config["manifest_dir"], config["dataset_root"], split,
                num_windows=windows, raw_cache=config["raw_cache"], limit=1, seed=seed)
                for split in ("train", "validation")}
    samples = {split: data[0] for split, data in datasets.items()}
    batches = {split: sequence_collate([sample]) for split, sample in samples.items()}
    if samples["train"]["sample_id"] == samples["validation"]["sample_id"]:
        raise ValueError("train/validation sample overlap")
    if any(s["source_split"] != "train" for s in samples.values()):
        raise ValueError("engineering probe crossed source-train boundary")
    labels_manifest = json.loads(Path(config["class_names_path"]).read_text())
    names = labels_manifest["class_names"]
    provenance = json.loads((Path(config["manifest_dir"]) / "provenance.json").read_text())
    class_order = [key for key, value in sorted(provenance["class_to_index"].items(), key=lambda pair: pair[1])]
    if labels_manifest["class_ids"] != class_order or len(names) != len(class_order):
        raise ValueError("text class order differs from immutable manifest labels")
    pretrained = load_pretrained_clip(config["clip_checkpoint"], source_dir=config["clip_source_dir"])
    identity = {"dataset": "N-ImageNet Mini engineering checks only", "downstream_dataset": None,
                "manifest_sha256": {k: v.manifest_sha256 for k, v in datasets.items()},
                "sequence_contract": datasets["train"].sequence_contract,
                "class_ids": class_order, "class_names": names, "prompt_template": config["prompt_template"],
                "class_names_sha256": file_sha256(config["class_names_path"]),
                "clip": pretrained.provenance,
                "source_sha256": {p.name: file_sha256(p) for p in Path(__file__).parent.glob("hierarchy_clip*.py")},
                "seed": seed, "stage_updates": 1, "device": device}
    results = {}
    report = {"status": "running", "evidence_kind": "bounded integration, not accuracy or full training",
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "gpu_gate": gate, "identity": identity,
              "final_test_accessed": False, "full_training_launched": False,
              "data": {split: {"sample_id": samples[split]["sample_id"], "source": samples[split]["source"],
                    "window_metadata": samples[split]["window_metadata"],
                    "packed_shapes": {k: list(v.shape) for k, v in batch["inputs"]["packed"].items()},
                    "mask": batch["inputs"]["window_mask"].tolist()}
                    for split, batch in batches.items()}, "comparisons": results}
    train_inputs = move_inputs(batches["train"]["inputs"], device)
    validation_inputs = move_inputs(batches["validation"]["inputs"], device)
    labels = batches["train"]["labels"].to(device)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for variant in VARIANTS:
        torch.manual_seed(seed)
        hierarchy = HierarchyV1(len(names))
        backbone_provenance = load_hierarchy_backbone(hierarchy, config["hierarchy_checkpoint"])
        encoder = HierarchySequenceEncoder(hierarchy, num_windows=windows)
        if variant == "hierarchy_temporal":
            model = HierarchyTemporalBaseline(encoder, num_classes=len(names))
        else:
            cls = HierarchyTextAlignment if variant == "hierarchy_text" else HierarchyCLIPViTAlignment
            model = cls(encoder, pretrained, names, prompt_template=config["prompt_template"])
        model.to(device)
        run_identity = {**identity, "variant": variant, "hierarchy": backbone_provenance}
        if hasattr(model, "text_bank"):
            run_identity["text"] = model.text_bank.contract()
        result = {"hierarchy": backbone_provenance, "stages": {},
                  "parameters": sum(p.numel() for p in model.parameters())}
        shapes = {}
        handles = []
        def capture(name):
            def hook(module, args, value):
                shapes[name] = {"input": list(args[0].shape) if isinstance(args[0], torch.Tensor) else None,
                                "output": list(value.shape)}
            return hook
        for name in ("encoder", "adapter", "temporal", "projection", "classifier"):
            module = getattr(model, name, None)
            if module is not None:
                handles.append(module.register_forward_hook(capture(name)))
        for stage in ("connectors", "selective"):
            optimizer = torch.optim.AdamW(configure_stage(model, stage, learning_rate=1e-4), weight_decay=0.01)
            frozen_before = frozen_fingerprint(model)
            set_staged_train(model)
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_inputs)
            if tuple(logits.shape) != (1, len(names)):
                raise ValueError("comparison output shape mismatch")
            loss = nn.functional.cross_entropy(logits.float(), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite bounded loss")
            loss.backward()
            gradients = gradient_report(model)
            for module_name in ("temporal", "adapter", "projection", "classifier"):
                details = gradients.get(module_name, {})
                if details.get("trainable_parameters", 0) and details.get("gradient_l1", 0) <= 0:
                    raise ValueError(f"missing {module_name} gradient")
            if stage == "selective" and gradients["encoder"]["gradient_l1"] <= 0:
                raise ValueError("missing selective hierarchy gradient")
            if stage == "selective" and variant == "hierarchy_clip_vit" and gradients["visual"]["gradient_l1"] <= 0:
                raise ValueError("missing selective pretrained ViT gradient")
            optimizer.step()
            if frozen_fingerprint(model) != frozen_before:
                raise ValueError("optimizer changed a frozen pretrained parameter")
            model.eval()
            expected = evaluation_logits(model, validation_inputs)
            if not torch.isfinite(expected).all():
                raise FloatingPointError("nonfinite validation forward")
            # Check actual disk serialization and strict loading after corrupting
            # a trainable tensor; scratch checkpoints are removed after success.
            with tempfile.TemporaryDirectory(prefix="reload_", dir=output) as temporary:
                path = Path(temporary) / "bounded.pt"
                save_comparison_checkpoint(path, model, optimizer, run_identity)
                digest = file_sha256(path)
                with torch.no_grad():
                    next(p for p in model.parameters() if p.requires_grad).add_(1)
                load_comparison_checkpoint(path, model, optimizer, run_identity)
                actual = evaluation_logits(model, validation_inputs)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            result["stages"][stage] = {"updates": 1, "train_loss": float(loss.detach()),
                "gradient_report": gradients, "logits_shape": list(logits.shape),
                "frozen_parameters_unchanged": True, "tensor_shapes": dict(shapes),
                "validation_forward_finite": True, "reload_logits_bit_exact": True,
                "checkpoint_evaluation_deterministic": True,
                "strict_loaded_state_tensors_bit_exact": True,
                "bounded_checkpoint_sha256": digest, "bounded_checkpoint_retained": False,
                "optimizer_groups": [{"name": g["name"], "lr": g["lr"],
                                      "parameters": sum(p.numel() for p in g["params"])}
                                     for g in optimizer.param_groups]}
            print(json.dumps({"variant": variant, "stage": stage, "status": "PASS", "device": device}), flush=True)
        results[variant] = result
        for handle in handles:
            handle.remove()
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        del model, optimizer, hierarchy, encoder, logits, loss
        gc.collect()
    report["status"] = "PASS"
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda:0", "cuda:1"))
    args = parser.parse_args()
    run_probe(json.loads(args.config.read_text()), args.output_dir, device=args.device)


if __name__ == "__main__":
    main()
