"""Profile or bounded CPU verification of polarity aggregation. No train command."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import torch

from .hierarchy_data import HierarchyDataset, collate
from .hierarchy_ts_data import HierarchyTSDataset, collate as ts_collate
from .hierarchy_polarity_models import BASE_MODELS, POLARITY_CONTRACT, make_model, profile_macs
from .v1 import load_config
from .v1_training import seed_everything


def verify(base, config_path, raw_cache, output):
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    config, locations = load_config(config_path)
    diagnostic = replace(config, device="cpu", precision="float32", batch_size=1,
                         accumulation_steps=1, num_workers=0, cpu_threads=2)
    seed_everything(diagnostic)
    dataset_type, batcher = (HierarchyDataset, collate) if base == "hierarchy" else (HierarchyTSDataset, ts_collate)
    data = dataset_type(**locations, split="train", raw_cache=raw_cache)
    sample = data[0]
    assert sample["split"] == "train" and sample["source_split"] == "train"
    inputs = batcher([sample])["inputs"]
    model = make_model(base)
    signed = inputs["points"][:, 3]
    polarity_counts = [int((signed == sign).sum()) for sign in (-1, 1)]
    assert all(n > 0 for n in polarity_counts)
    with torch.no_grad():
        grid = model.point_to_voxel(model.point(inputs["points"]), inputs)
        masses = [float(grid[:, channel].expm1().sum()) for channel in (32, 65)]
        for expected, observed in zip(polarity_counts, masses):
            assert abs(expected - observed) < .02
        assert torch.isfinite(grid).all()
    del grid
    gradients, losses = [], []
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate,
                                momentum=config.momentum, weight_decay=config.weight_decay)
    model.train()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([sample["label"]]))
        assert torch.isfinite(logits).all() and torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
        assert float(norm) > 0
        grad = {name: sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
                for name, module in model.named_children() if list(module.parameters())}
        for stage, value in grad.items():
            if step == 0 and stage in ("ts_encoder", "confidence_control"):
                assert value == 0
            else:
                assert value > 0, stage
        fusion = model.voxel_projection[0].weight.grad
        grad["negative_fusion_channels"] = float(fusion[:, :33].abs().sum())
        grad["positive_fusion_channels"] = float(fusion[:, 33:].abs().sum())
        assert grad["negative_fusion_channels"] > 0 and grad["positive_fusion_channels"] > 0
        gradients.append(grad); losses.append(float(loss.detach()))
        optimizer.step()
    output.mkdir(parents=True)
    checkpoint = output / "diagnostic_checkpoint.pt"
    torch.save({"base": base, "polarity_contract": POLARITY_CONTRACT, "model": model.state_dict()}, checkpoint)
    restored = make_model(base).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=False)["model"], strict=True)
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(inputs), restored(inputs))
    package = Path(__file__).parent
    result = {"status": "PASS", "evidence_kind": "one train sample; two CPU optimizer steps; not an accuracy experiment",
              "base": base, "training_settings_unchanged": asdict(config), "diagnostic_settings": asdict(diagnostic),
              "training_manifest_sha256": data.manifest_sha256, "raw_source": sample["source"],
              "negative_positive_events": polarity_counts, "negative_positive_interpolated_mass": masses,
              "profile": profile_macs(model, len(signed)), "profile_at_116790": profile_macs(model),
              "gradient_l1_by_stage": gradients, "losses": losses,
              "checkpoint_verification": {"strict_load": True, "logits_bit_exact": True,
                                          "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob("*.py")},
              "test_set_accessed": False, "validation_accessed": False, "full_training_launched": False,
              "gpu_used": False}
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("profile", "verify"):
        p = sub.add_parser(command)
        p.add_argument("--base", choices=tuple(BASE_MODELS), default="hierarchy")
        if command == "profile":
            p.add_argument("--events", type=int, default=116790)
        else:
            p.add_argument("--config", default="configs/hierarchy_v1.n_imagenet_mini.json")
            p.add_argument("--raw-cache", type=Path)
            p.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = (profile_macs(make_model(args.base), args.events) if args.command == "profile" else
              verify(args.base, args.config, args.raw_cache, args.output_dir))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
