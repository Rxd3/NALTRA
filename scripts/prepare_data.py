"""Prepare checksum-locked CORDIS sources and derived NALTRA benchmark tracks."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from naltra.cli import prepare_data  # noqa: E402


def main() -> None:
    prepare_data()


if __name__ == "__main__":
    main()
