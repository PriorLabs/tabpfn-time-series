import os

import pytest


_CI_CREDENTIALS_BY_MARKER = {
    "uses_tabpfn_client": "TABPFN_CLIENT_API_KEY",
    "uses_tabpfn_local": "TABPFN_TOKEN",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip credential-dependent integration tests when fork CI has no secrets."""
    if os.getenv("GITHUB_ACTIONS") != "true":
        return

    for item in items:
        for marker_name, env_var in _CI_CREDENTIALS_BY_MARKER.items():
            if item.get_closest_marker(marker_name) and not os.getenv(env_var):
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"{env_var} is unavailable in this GitHub Actions run"
                    )
                )
                break
