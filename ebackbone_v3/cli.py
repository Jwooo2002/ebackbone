"""Command-line interface with lazy imports for optional runtime dependencies."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from ebackbone_v3.errors import EBackboneV3Error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ebackbone_V3 project utilities",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe_parser = subparsers.add_parser(
        "probe",
        help="Inspect one real raw-event sample and verify representation provenance.",
    )
    probe_parser.add_argument(
        "--config",
        required=True,
        help="Path to a resolved JSON probe-provider configuration.",
    )
    probe_parser.set_defaults(handler=_handle_probe)

    build_splits_parser = subparsers.add_parser(
        "build-splits",
        help="Build immutable supervised N-ImageNet mini split manifests.",
    )
    build_splits_parser.add_argument(
        "--config",
        required=True,
        help="Path to a resolved JSON split-build configuration.",
    )
    build_splits_parser.set_defaults(handler=_handle_build_splits)

    verify_splits_parser = subparsers.add_parser(
        "verify-splits",
        help="Verify immutable split manifests, provenance, and leakage invariants.",
    )
    verify_splits_parser.add_argument(
        "--manifest-dir",
        required=True,
        help="Directory containing train/validation/test JSONL and provenance files.",
    )
    verify_splits_parser.set_defaults(handler=_handle_verify_splits)

    inspect_sample_parser = subparsers.add_parser(
        "inspect-sample",
        help=(
            "Resolve one manifest-backed raw-event sample and report CPU tensor metadata only."
        ),
    )
    inspect_sample_parser.add_argument(
        "--manifest-dir",
        required=True,
        help="Directory containing the immutable supervised manifest artifacts.",
    )
    inspect_sample_parser.add_argument(
        "--index",
        required=True,
        type=int,
        help="Zero-based row index in the selected immutable manifest.",
    )
    inspect_sample_parser.add_argument(
        "--baseline",
        required=True,
        choices=("b0", "b1"),
        help="Return the B0 frame only or the aligned B1 representation bundle.",
    )
    inspect_sample_parser.add_argument(
        "--cache",
        required=True,
        choices=("off", "on"),
        help="Disable caching or validate/use one explicit representation-cache entry.",
    )
    inspect_sample_parser.add_argument(
        "--dataset-root",
        default="/mnt/hdd1/datasets/event/n_imagenet",
        help=(
            "N-ImageNet root containing mini_zenodo/archives; defaults to the checked-in "
            "local configuration."
        ),
    )
    inspect_sample_parser.add_argument(
        "--cache-root",
        help="Required with --cache on; destination for on-demand representation entries.",
    )
    inspect_sample_parser.add_argument(
        "--split",
        required=True,
        choices=("train", "validation", "test"),
        help="Explicit project role for the selected immutable manifest.",
    )
    inspect_sample_parser.add_argument(
        "--allow-final-test",
        action="store_true",
        help="Required in addition to --split test before any final-test access.",
    )
    inspect_sample_parser.set_defaults(handler=_handle_inspect_sample)

    smoke_parser = subparsers.add_parser(
        "smoke",
        help="Run a synthetic CPU-only forward, cross-entropy, and backward check.",
    )
    smoke_parser.add_argument(
        "--baseline",
        required=True,
        choices=("b0", "b1"),
        help="Synthetic baseline contract to exercise.",
    )
    smoke_parser.set_defaults(handler=_handle_smoke)

    train_b0_parser = subparsers.add_parser(
        "train-b0",
        help="Train production B0 on full immutable train/validation manifests.",
    )
    train_b0_parser.add_argument(
        "--manifest-dir",
        required=True,
        help="Directory containing the immutable supervised manifest artifacts.",
    )
    train_b0_parser.add_argument(
        "--dataset-root",
        default="/mnt/hdd1/datasets/event/n_imagenet",
        help="N-ImageNet root containing mini_zenodo/archives.",
    )
    train_b0_parser.add_argument(
        "--output-dir",
        required=True,
        help="New output directory, or the existing run directory with --resume.",
    )
    train_b0_parser.add_argument("--epochs", type=int, required=True)
    train_b0_parser.add_argument("--batch-size", type=int, default=4)
    train_b0_parser.add_argument("--learning-rate", type=float, default=0.05)
    train_b0_parser.add_argument("--momentum", type=float, default=0.9)
    train_b0_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_b0_parser.add_argument("--seed", type=int, default=20260715)
    train_b0_parser.add_argument("--num-workers", type=int, default=0)
    train_b0_parser.add_argument("--prefetch-factor", type=int, default=2)
    train_b0_parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA bfloat16 autocast (autocast is enabled by default on CUDA).",
    )
    train_b0_parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Execution device; CPU is the fail-safe default and CUDA requires explicit selection.",
    )
    train_b0_parser.add_argument(
        "--stop-after-epoch",
        type=int,
        help="Stop cleanly after this epoch to exercise or schedule resume.",
    )
    train_b0_parser.add_argument(
        "--resume",
        help="Resume strictly from checkpoint_last.pt in the selected output directory.",
    )
    train_b0_parser.set_defaults(handler=_handle_train_b0)

    debug_b0_parser = subparsers.add_parser(
        "train-b0-debug",
        help="Run only the explicit 4--16 sample CPU tiny-overfit engineering diagnostic.",
    )
    debug_b0_parser.add_argument("--manifest-dir", required=True)
    debug_b0_parser.add_argument(
        "--dataset-root", default="/mnt/hdd1/datasets/event/n_imagenet"
    )
    debug_b0_parser.add_argument("--output-dir", required=True)
    debug_b0_parser.add_argument("--subset-size", type=int, default=8)
    debug_b0_parser.add_argument("--epochs", type=int, default=120)
    debug_b0_parser.add_argument("--batch-size", type=int, default=4)
    debug_b0_parser.add_argument("--learning-rate", type=float, default=0.01)
    debug_b0_parser.add_argument("--seed", type=int, default=20260715)
    debug_b0_parser.add_argument("--target-train-accuracy", type=float, default=0.95)
    debug_b0_parser.set_defaults(handler=_handle_train_b0_debug)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = args.handler(args)
    except EBackboneV3Error as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


def entrypoint() -> None:
    raise SystemExit(main())


def _handle_probe(args: argparse.Namespace) -> dict[str, Any]:
    from ebackbone_v3.probe import run_probe

    return run_probe(args.config)


def _handle_smoke(args: argparse.Namespace) -> dict[str, Any]:
    from ebackbone_v3.smoke import run_synthetic_smoke

    return run_synthetic_smoke(args.baseline)


def _handle_build_splits(args: argparse.Namespace) -> dict[str, Any]:
    from ebackbone_v3.splits import build_splits

    return build_splits(args.config)


def _handle_verify_splits(args: argparse.Namespace) -> dict[str, Any]:
    from ebackbone_v3.splits import verify_splits

    return verify_splits(args.manifest_dir)


def _handle_inspect_sample(args: argparse.Namespace) -> dict[str, Any]:
    from ebackbone_v3.n_imagenet_mini_dataset import inspect_sample

    return inspect_sample(
        manifest_dir=args.manifest_dir,
        index=args.index,
        baseline=args.baseline,
        cache=args.cache,
        dataset_root=args.dataset_root,
        cache_root=args.cache_root,
        split=args.split,
        allow_final_test=args.allow_final_test,
    )


def _handle_train_b0(args: argparse.Namespace) -> dict[str, Any]:
    from ebackbone_v3.b0_production import run_production_b0

    return run_production_b0(
        manifest_dir=args.manifest_dir,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        amp=not args.no_amp,
        device_name=args.device,
        stop_after_epoch=args.stop_after_epoch,
        resume=args.resume,
    )


def _handle_train_b0_debug(args: argparse.Namespace) -> dict[str, Any]:
    from ebackbone_v3.b0_training import run_tiny_overfit

    return run_tiny_overfit(
        manifest_dir=args.manifest_dir,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        subset_size=args.subset_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        target_train_accuracy=args.target_train_accuracy,
    )


__all__ = ["build_parser", "entrypoint", "main"]
