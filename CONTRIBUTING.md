# Contributing to TabPFN Time Series

Thanks for contributing! Before building a feature or opening a PR, please open a
GitHub issue describing the bug or proposal so we can discuss it first — it helps
avoid unnecessary work.

## Development setup

We use [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage the
environment.

```bash
git clone https://github.com/PriorLabs/tabpfn-time-series.git
cd tabpfn-time-series

# Full install: package + all extras + dev tooling
uv sync --all-extras --dev

# Install the git hooks (lint/format on commit, blocks direct commits to main)
uv run pre-commit install
```

## Repository layout

```
tabpfn_time_series/   # the package
tests/                # pytest suite
examples/             # user-facing tutorial scripts
gift_eval/            # GIFT-Eval benchmark harness (not shipped in the wheel)
scripts/              # maintenance/CI helper scripts
docs/                 # documentation sources
```

## Linting & formatting

Ruff handles both; the exact version is pinned in `pyproject.toml` and configured
in `ruff.toml`. CI enforces both checks.

```bash
uv run ruff check .
uv run ruff format .
```

## Testing

```bash
uv run pytest tests -ra
```

Mark tests that rely on `tabpfn-client` with
```python
@pytest.mark.uses_tabpfn_client
def test__my_test() -> None:
    ...
```
This allows these tests to skip when access an API key or Internet access is not available.

## Pull requests

- Target `main`. CI must be green — the required checks are lint, the OS × Python ×
  dependency-resolution test matrix, and the wheel/sdist build.
- One approving review is required; the branch must be up to date with `main`
  before merging.
- Keep PRs focused; stacked PRs are welcome for larger changes.
- New user-facing behavior should come with tests, and with an example update if
  it changes how the package is used.

## Legal

- Only contribute code you have the rights to; all contributions are licensed
  under Apache 2.0.
- No model weights or large datasets in the repository.

## Need help?

Open an issue, or join the [Prior Labs Discord](https://discord.com/channels/1285598202732482621/).
