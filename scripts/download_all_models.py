"""Download every TabPFN checkpoint so the test suite can run offline.

Downloading a checkpoint requires a one-time license acceptance, which in CI
means a `TABPFN_TOKEN`. Fork PRs never get repository secrets, so CI stores the
download directory in the GitHub Actions cache keyed on its own contents; PR
jobs restore that cache and find the checkpoints already on disk, where
`tabpfn` loads them without contacting Hugging Face at all.

Set `TABPFN_MODEL_CACHE_DIR` to control where the checkpoints land; both this
script and `tabpfn` itself honour it.
"""

from __future__ import annotations

import logging

from tabpfn.model_loading import download_all_models, get_cache_dir

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cache_dir = get_cache_dir()
    logger.info("Downloading all models to %s", cache_dir)
    download_all_models(cache_dir)
    logger.info("All models downloaded to %s", cache_dir)


if __name__ == "__main__":
    main()
