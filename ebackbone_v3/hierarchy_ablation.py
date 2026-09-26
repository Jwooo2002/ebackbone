"""Independent hierarchy ablation profile, real-data probe, bounded diagnostic and training."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from .hierarchy_data import HierarchyDataset, POINT_CONTRACT, POINT_CONTRACT_SHA256
from .hierarchy_ablation_models import make_model, MODEL_NAMES, profile_macs
from .hierarchy_ablation_training import atomic_json, run_model
from .v1 import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    profile = commands.add_parser("profile")
    profile.add_argument("--model", required=True, choices=MODEL_NAMES)
    profile.add_argument("--events", type=int, default=116790,
                         help="events per sample; default is the verified first train sample")
    profile.add_argument("--output", type=Path)
    for command in ("probe", "diagnose", "train"):
        sub = commands.add_parser(command)
        sub.add_argument("--model", required=True, choices=MODEL_NAMES)
        sub.add_argument("--config", required=True)
        sub.add_argument("--raw-cache", type=Path)
        if command == "probe":
            sub.add_argument("--index", type=int, default=0)
            sub.add_argument("--output", type=Path)
        else:
            sub.add_argument("--output-dir", type=Path, required=True)
            sub.add_argument("--resume", action="store_true")
            sub.add_argument("--stop-after-epoch", type=int)
        if command == "diagnose":
            sub.add_argument("--samples", type=int, default=2, choices=range(1, 17))
            sub.add_argument("--epochs", type=int, default=1, choices=range(1, 4))
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    if args.command == "profile":
        result = profile_macs(make_model(args.model), args.events)
    else:
        config, locations = load_config(args.config)
        if args.command == "probe":
            dataset = HierarchyDataset(**locations, split="train", raw_cache=args.raw_cache)
            sample = dataset[args.index]
            result = {"status": "PASS", "source": sample["source"], "train_samples": len(dataset),
                      "point_contract": POINT_CONTRACT, "point_contract_sha256": POINT_CONTRACT_SHA256,
                      "final_test_accessed": False, "profile": profile_macs(make_model(args.model), sample["source"]["event_count"]),
                      "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                                      "min": float(v.min()), "max": float(v.max()),
                                      "finite": bool(torch.isfinite(v).all())}
                                  for k, v in sample["inputs"].items()}}
        else:
            diagnostic = args.command == "diagnose"
            if diagnostic:
                config = replace(config, epochs=args.epochs, batch_size=1, accumulation_steps=1,
                                 device="cpu", precision="float32", num_workers=0)
            result = run_model(args.model, config, **locations, output_dir=args.output_dir,
                               raw_cache=args.raw_cache, resume=args.resume,
                               stop_after_epoch=args.stop_after_epoch,
                               diagnostic_samples=args.samples if diagnostic else None)
    output = getattr(args, "output", None)
    if output:
        if output.exists():
            raise ValueError(f"refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
