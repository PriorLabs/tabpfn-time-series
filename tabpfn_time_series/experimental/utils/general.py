from pathlib import Path


def find_repo_root():
    """Find repository root by locating LICENSE.txt file."""
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / "LICENSE.txt").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find repository root (LICENSE.txt not found)")
