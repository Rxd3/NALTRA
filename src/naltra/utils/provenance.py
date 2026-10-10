"""Run provenance shared by the result writers: code revision and publishable paths."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
# Inherited from a calling git process, these would make git describe another repository.
GIT_LOCATION = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")


def commit() -> str | None:
    """``git describe --always --dirty`` of this checkout, or None without git or a repository."""
    try:
        described = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            check=True,
            env={name: value for name, value in os.environ.items() if name not in GIT_LOCATION},
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return described or None


def repo_path(path: str | Path) -> str:
    """``path`` as result files record it, without local machine details.

    A relative ``path`` is read from the repository root. Inside the repository it is recorded
    relative with forward slashes; outside, only its file name.
    """
    resolved = (REPO_ROOT / path).resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return resolved.relative_to(REPO_ROOT).as_posix()
    return resolved.name
