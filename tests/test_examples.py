"""Run the example files in ``examples/`` and check they work.

Each example is executed as a subprocess with a fixed per-example timeout.
There are two run modes:

* **Smoke** (default): an example that *errors* fails the test; one that simply
  doesn't finish in time *passes* -- we only assert it starts and runs without
  crashing. This is the fast, cheap per-PR guard against broken example scripts
  and import errors.
* **Full** (``EXAMPLE_FULL=1``): every example must run *to completion*, and a
  timeout is a failure -- exactly as a user would experience it. This is the
  scheduled GPU full run. The switch is an environment variable rather than a
  pytest CLI option so that command-line arguments cannot flip the pass/fail
  semantics of a run.

An example is **skipped** (not failed) when the credentials it needs aren't
available -- the same rule the unit suite applies via the ``uses_tabpfn_client``
marker, so fork PRs (which get no secrets) report honest skips instead of red
builds.

Usage:
    uv run --no-sync pytest tests/test_examples.py --run-examples
    EXAMPLE_FULL=1 uv run --no-sync pytest tests/test_examples.py --run-examples
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import should_skip_client_tests

# Full run: every example runs to completion; a timeout is a failure. Used by
# the scheduled GPU full run. In the default smoke mode a timeout still passes
# -- we only assert the example starts and runs without crashing.
FULL_RUN = os.environ.get("EXAMPLE_FULL", "0") == "1"

# Per-example timeout. The smoke budget is short since a timeout passes anyway;
# the full-run budget is sized so it only bites on genuine hangs. The slowest
# example (explainability_electricity) fits a TabPFN context per rolling window
# and downloads a parquet from S3 first.
EXAMPLE_TIMEOUT_SECONDS = 900 if FULL_RUN else 120

# Files under examples/ that are not themselves examples: shared helpers the
# example scripts import. Running them as a script is a no-op at best.
NOT_EXAMPLES = {"common.py"}

# Examples that call the TabPFN cloud API and so need TABPFN_CLIENT_API_KEY.
REQUIRES_CLIENT = {"tabpfn_family_model_as_backbone.py"}


def get_example_files() -> list[dict]:
    """Discover example files and attach the metadata the runner needs."""
    package_root = Path(__file__).parent.parent
    examples_dir = package_root / "examples"

    files = []
    for file_path in sorted(examples_dir.glob("**/*.py")):
        name = file_path.name
        if name in NOT_EXAMPLES:
            continue
        files.append(
            {
                "path": file_path,
                "name": name,
                "requires_client": name in REQUIRES_CLIENT,
            },
        )
    return files


@pytest.mark.parametrize(
    "example_file",
    [pytest.param(f, id=f["name"]) for f in get_example_files()],
)
def test_example(request, example_file):
    """Run a single example file as a subprocess and check the outcome."""
    name = example_file["name"]
    path = example_file["path"]

    if not request.config.getoption("--run-examples"):
        pytest.skip(f"Skipping {name} since --run-examples not set")

    if example_file["requires_client"]:
        reason = should_skip_client_tests()
        if reason is not None:
            pytest.skip(f"Example {name} needs the TabPFN cloud API: {reason}")

    # Examples are top-to-bottom scripts; run each in its own process so a hang
    # can be killed cleanly and state never leaks between examples. They import
    # sibling helpers (examples/common.py) by plain module name, so the working
    # directory has to be examples/.
    env = dict(os.environ)
    if FULL_RUN:
        # Full size, exactly as a user would run the example. Stripped rather
        # than inherited so a stray setting can't quietly shrink what the full
        # run verifies.
        env.pop("FAST_TEST_MODE", None)
    else:
        # Shrink the workload where an example supports it. None do today; the
        # smoke budget is what actually bounds runtime here.
        env["FAST_TEST_MODE"] = "1"
    # Headless plotting: an example calling plt.show() must never block on a GUI
    # window (on a machine with a display, the interactive backend blocks until
    # the window is closed -- fatal in full-run mode, where that reads as a
    # timeout).
    env.setdefault("MPLBACKEND", "Agg")

    try:
        proc = subprocess.run(  # noqa: S603 - trusted, repo-local example scripts
            [sys.executable, str(path)],
            cwd=str(path.parent),
            env=env,
            timeout=EXAMPLE_TIMEOUT_SECONDS,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        if FULL_RUN:
            pytest.fail(
                f"Example {name} did not complete within "
                f"{EXAMPLE_TIMEOUT_SECONDS}s (full run: a timeout is a failure)",
            )
        # Not a failure: the example started and ran without crashing, which is
        # all the smoke gate asserts. Completion is checked by the full run.
        print(
            f"{name}: ran {EXAMPLE_TIMEOUT_SECONDS}s without error "
            f"(smoke mode; completion not verified)",
        )
        return

    if proc.returncode != 0:
        output = (proc.stdout or b"").decode("utf-8", "replace")
        tail = "\n".join(output.strip().splitlines()[-100:])
        pytest.fail(f"Example {name} exited with code {proc.returncode}:\n{tail}")
