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
        "--manifest",
        required=True,
        help="Path to immutable train.jsonl, validation.jsonl, or test.jsonl.",
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
        choices=("train", "validation", "test"),
        help="Optional manifest role; --split test is required before final-test access.",
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
        manifest_path=args.manifest,
        index=args.index,
        baseline=args.baseline,
        cache=args.cache,
        dataset_root=args.dataset_root,
        cache_root=args.cache_root,
        split=args.split,
    )


__all__ = ["build_parser", "entrypoint", "main"]
