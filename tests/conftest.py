"""Skip integration tests whose credentials / model weights aren't available.

Two markers gate the integration tests (declared in `pyproject.toml`):

* `uses_tabpfn_client` needs `TABPFN_CLIENT_API_KEY` — a cloud call, so there is
  no offline substitute.
* `uses_tabpfn_local` needs the TabPFN checkpoint on disk. That is satisfied
  either by a checkpoint already in the model cache (CI restores one that `main`
  populated — see `.github/workflows/cache-models.yml`) or by a `TABPFN_TOKEN`
  to download it on the fly.

Fork PRs get neither secret, so without this hook the whole integration suite
fails there with `TabPFNLicenseError` / missing-key assertions on changes that
have nothing to do with TabPFN. Skipping keeps the signal honest for external
contributors while internal PRs — which do get the secrets — still run
everything.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

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


def skip_reason_for_client() -> str | None:
    if os.getenv("TABPFN_CLIENT_API_KEY"):
        return None
    return "TABPFN_CLIENT_API_KEY is not set (expected on fork PRs)"


@functools.lru_cache(maxsize=1)
def skip_reason_for_local() -> str | None:
    from tabpfn.model_loading import prepend_cache_path

    from tabpfn_time_series.defaults import TABPFN_V3_TS_CHECKPOINT

    checkpoint = Path(prepend_cache_path(TABPFN_V3_TS_CHECKPOINT))
    if checkpoint.exists():
        return None
    if os.getenv("TABPFN_TOKEN"):
        return None
    return (
        f"{checkpoint.name} is not in the model cache ({checkpoint.parent}) and "
        "TABPFN_TOKEN is not set to download it (expected on fork PRs)"
    )


_SKIP_REASON_BY_MARKER = {
    "uses_tabpfn_client": skip_reason_for_client,
    "uses_tabpfn_local": skip_reason_for_local,
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        for marker, skip_reason in _SKIP_REASON_BY_MARKER.items():
            if item.get_closest_marker(marker) is None:
                continue
            reason = skip_reason()
            if reason is not None:
                item.add_marker(pytest.mark.skip(reason=reason))
                break
