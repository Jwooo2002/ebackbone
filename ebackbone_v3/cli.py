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


__all__ = ["build_parser", "entrypoint", "main"]
