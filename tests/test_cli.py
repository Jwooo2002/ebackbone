from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "main.py", *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_root_help_lists_commands() -> None:
    result = _run_cli("--help")
    assert result.returncode == 0
    assert result.stderr == ""
    assert "{probe,smoke}" in result.stdout
    assert "Inspect one real raw-event sample" in result.stdout
    assert "synthetic CPU-only" in result.stdout


@pytest.mark.parametrize(
    ("command", "expected"),
    [("probe", "--config"), ("smoke", "--baseline {b0,b1}")],
)
def test_subcommand_help(command: str, expected: str) -> None:
    result = _run_cli(command, "--help")
    assert result.returncode == 0
    assert result.stderr == ""
    assert expected in result.stdout


@pytest.mark.parametrize("baseline", ["b0", "b1"])
def test_smoke_cli_returns_json_pass(baseline: str) -> None:
    result = _run_cli("smoke", "--baseline", baseline)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    report = json.loads(result.stdout)
    assert report["status"] == "PASS"
    assert report["baseline"] == baseline
    assert report["device"] == "cpu"


def test_example_probe_fails_with_exact_unresolved_keys() -> None:
    result = _run_cli("probe", "--config", "configs/probe.example.json")
    assert result.returncode == 2
    assert result.stdout == ""
    assert "provider.callable" in result.stderr
    assert "provider.kwargs.dataset_root" in result.stderr
    assert "provider.kwargs.sample_id" in result.stderr
    assert "Traceback" not in result.stderr


def test_probe_cli_reports_verified_test_fixture() -> None:
    result = _run_cli("probe", "--config", "tests/fixtures/probe.aligned.json")
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    report = json.loads(result.stdout)
    assert report["status"] == "PASS"
    assert report["dataset"]["name"] == "contract-test-fixture"
    assert report["raw_events"]["event_count"] == 4
    assert report["temporal_alignment"]["status"] == "VERIFIED_FROM_PROVIDER_EVIDENCE"


def test_importing_main_has_no_cli_side_effect() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import main; print('import-ok')"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "import-ok"
    assert result.stderr == ""
