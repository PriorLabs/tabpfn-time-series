from pathlib import Path


def find_repo_root() -> Path:
    """Find repository root by locating LICENSE.txt file."""
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / "LICENSE.txt").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find repository root (LICENSE.txt not found)")


REPO_ROOT: Path = find_repo_root()
OUTPUT_ROOT: Path = REPO_ROOT / "output"
