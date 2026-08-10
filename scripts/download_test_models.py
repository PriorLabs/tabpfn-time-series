"""Download the TabPFN checkpoints the test suite needs for LOCAL-mode inference.

Downloading a checkpoint requires a one-time license acceptance, which in CI
means a `TABPFN_TOKEN`. Fork PRs never get repository secrets, so this script is
run on `main` (see `.github/workflows/cache-models.yml`) and the resulting
directory is stored in the GitHub Actions cache; PR jobs restore that cache and
find the checkpoints already on disk, where `tabpfn` loads them without
contacting Hugging Face at all.

Set `TABPFN_MODEL_CACHE_DIR` (or pass `--cache-dir`) to control where the
checkpoints land; both this script and `tabpfn` itself honour it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tabpfn.model_loading import (
    download_model,
    get_cache_dir,
    resolve_model_version,
)

from tabpfn_time_series.defaults import TABPFN_V3_TS_CHECKPOINT

logger = logging.getLogger(__name__)

# Checkpoints exercised by the LOCAL-mode tests. Currently only the TS ckpt that
# `resolve_default_ckpt` picks; keep this in sync with `defaults.py`.
REQUIRED_CHECKPOINTS = [TABPFN_V3_TS_CHECKPOINT]

# A truncated or error-page "checkpoint" would be worse than no cache at all:
# the tests would fail on a corrupt file instead of skipping. Real ckpts are
# hundreds of MB, so anything this small means the download went wrong.
MIN_CHECKPOINT_BYTES = 10 * 1024 * 1024


def download_checkpoints(cache_dir: Path) -> None:
    """Download every required checkpoint that is not already in `cache_dir`."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    for ckpt_name in REQUIRED_CHECKPOINTS:
        path = cache_dir / ckpt_name
        if path.exists():
            logger.info("Already cached, skipping: %s", path)
            continue

        logger.info("Downloading %s -> %s", ckpt_name, path)
        result = download_model(
            to=path,
            version=resolve_model_version(str(path)),
            which="regressor",
            model_name=ckpt_name,
        )
        if result != "ok":
            raise RuntimeError(f"Failed to download {ckpt_name}: {result}")

        size = path.stat().st_size if path.exists() else 0
        if size < MIN_CHECKPOINT_BYTES:
            raise RuntimeError(
                f"{ckpt_name} downloaded but is only {size} bytes; refusing to "
                "cache what looks like a failed download"
            )
        logger.info("Downloaded %s (%.0f MB)", ckpt_name, size / 1024**2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Where to store the checkpoints (default: tabpfn's cache dir).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cache_dir = args.cache_dir or get_cache_dir()
    download_checkpoints(cache_dir)
    logger.info("All required checkpoints are in %s", cache_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
