"""Skip integration tests whose credentials aren't available.

The `uses_tabpfn_client` marker indicates tests that call the client, thus require a
`TABPFN_CLIENT_API_KEY` to work. We skip these tests, if the secret isn't present.
"""

from __future__ import annotations

import os

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the opt-in flag for the example runner.

    The scripts in `examples/` are user-facing demos, not unit tests: they
    download data and fit real models, so they must not run as part of a plain
    `pytest tests`. Without this flag `tests/test_examples.py` is still
    collected but every case skips itself.
    """
    parser.addoption(
        "--run-examples",
        action="store_true",
        default=False,
        help="Run the example scripts in examples/ (see tests/test_examples.py).",
    )


def should_skip_client_tests() -> str | None:
    if os.getenv("TABPFN_CLIENT_API_KEY"):
        return None
    return "TABPFN_CLIENT_API_KEY is not set (expected on fork PRs)"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if item.get_closest_marker("uses_tabpfn_client") is None:
            continue
        reason = should_skip_client_tests()
        if reason is not None:
            item.add_marker(pytest.mark.skip(reason=reason))
