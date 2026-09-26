"""Keep local dataset and experiment integrations out of the portable suite."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-local-integration",
        action="store_true",
        default=False,
        help="Run tests that require local datasets or sibling experiment artifacts.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-local-integration"):
        return
    skip = pytest.mark.skip(reason="requires --run-local-integration and local assets")
    for item in items:
        if item.get_closest_marker("local_integration") is not None:
            item.add_marker(skip)
