"""Standalone V1 CLI; existing B0 commands remain unchanged.

Run ``python -m ebackbone_v3.v1 --help`` from the repository root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .v1_data import RENDERER, RENDERER_SHA256, V1Dataset, prepare_raw_cache
from .v1_models import MODEL_NAMES, V1Backbone, match_frame_width, profile_macs
from .v1_training import TrainConfig, atomic_json, compare


def load_config(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if set(value) != {"training", "manifest_dir", "dataset_root"}:
        raise ValueError("config requires exactly training, manifest_dir and dataset_root")
    training = TrainConfig(**value["training"])
    training.validate()
    locations = {k: str((path.parent / value[k]).resolve()) for k in ("manifest_dir", "dataset_root")}
    return training, locations


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    profile = commands.add_parser("profile", help="MAC/parameter comparison; no dataset access")
    profile.add_argument("--output", type=Path)
    for command in ("probe", "diagnose", "compare", "prepare-raw"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", default="configs/v1.n_imagenet_mini.json")
        sub.add_argument("--raw-cache", type=Path, required=command == "prepare-raw")
        if command in {"diagnose", "compare"}:
            sub.add_argument("--output-dir", type=Path, required=True)
            sub.add_argument("--resume", action="store_true")
            sub.add_argument("--stop-after-epoch", type=int)
        if command == "diagnose":
            sub.add_argument("--samples", type=int, default=2)
            sub.add_argument("--epochs", type=int, default=1)
        if command == "probe":
            sub.add_argument("--index", type=int, default=0)
            sub.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "profile":
        torch.set_num_threads(2)
        matched = match_frame_width()
        result = {"matched": matched, "models": {name: profile_macs(V1Backbone(
            name, frame_width=matched["frame_width"] if name == "frame_matched" else 32)) for name in MODEL_NAMES}}
    else:
        config, locations = load_config(args.config)
        if args.command == "prepare-raw":
            result = prepare_raw_cache(**locations, cache_root=args.raw_cache)
        elif args.command == "probe":
            multi = V1Dataset(**locations, split="train", multiview=True, raw_cache=args.raw_cache)[args.index]
            frame = V1Dataset(**locations, split="train", multiview=False, raw_cache=args.raw_cache)[args.index]
            if multi["source"] != frame["source"] or not torch.equal(multi["inputs"]["event_frame"], frame["inputs"]["event_frame"]):
                raise ValueError("cross-representation source/frame identity mismatch")
            result = {"status": "PASS", "source": multi["source"], "renderer": RENDERER,
                      "renderer_sha256": RENDERER_SHA256, "frame_bit_exact_across_paths": True,
                      "same_raw_subset_and_interval": True, "final_test_accessed": False,
                      "decode_seconds": multi["decode_seconds"], "render_seconds": multi["render_seconds"],
                      "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                                      "min": float(v.min()), "max": float(v.max()),
                                      "finite": bool(torch.isfinite(v).all())} for k, v in multi["inputs"].items()}}
        else:
            diagnostic = args.command == "diagnose"
            if diagnostic:
                config = TrainConfig(epochs=args.epochs, batch_size=1, accumulation_steps=1,
                                     learning_rate=config.learning_rate, momentum=config.momentum,
                                     weight_decay=config.weight_decay, seed=config.seed,
                                     num_workers=0, device="cpu", precision="float32", cpu_threads=4)
            result = compare(config, **locations, output_dir=args.output_dir, raw_cache=args.raw_cache,
                             diagnostic_samples=args.samples if diagnostic else None,
                             resume=args.resume, stop_after_epoch=args.stop_after_epoch)
    output = getattr(args, "output", None)
    if output:
        if output.exists():
            raise ValueError(f"refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
