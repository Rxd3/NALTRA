"""Code revision and publishable paths recorded by the result writers."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import pytest

from naltra.utils import provenance


def strings(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, str):
        yield value


def absolute_paths(payload: Any, *roots: Path) -> list[str]:
    """Strings of a JSON payload that are absolute paths or name a local root."""
    named = [str(root) for root in (provenance.REPO_ROOT, *roots)]
    return [
        text
        for text in strings(payload)
        if PureWindowsPath(text).is_absolute()
        or PurePosixPath(text).is_absolute()
        or any(root in text for root in named)
    ]


def test_a_path_inside_the_repository_is_recorded_relative_with_forward_slashes() -> None:
    path = provenance.REPO_ROOT / "data" / "noisy" / "combined" / "test.jsonl"
    assert provenance.repo_path(path) == "data/noisy/combined/test.jsonl"


def test_a_path_outside_the_repository_is_reduced_to_its_file_name(tmp_path) -> None:
    assert provenance.repo_path(tmp_path / "release" / "test.jsonl") == "test.jsonl"


def test_a_relative_path_is_read_from_the_repository_not_the_working_directory(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert provenance.repo_path("data/noisy/test.jsonl") == "data/noisy/test.jsonl"


def test_commit_ignores_a_git_location_inherited_from_the_environment(monkeypatch) -> None:
    environments = []

    def describe(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        environments.append(kwargs.get("env"))
        return subprocess.CompletedProcess(command, 0, stdout="042ff9c\n")

    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.setenv(name, "elsewhere")
    monkeypatch.setenv("NALTRA_KEPT", "1")
    monkeypatch.setattr(provenance.subprocess, "run", describe)
    assert provenance.commit() == "042ff9c"
    env = environments[0]
    assert env is not None and env["NALTRA_KEPT"] == "1"
    assert not {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"} & set(env)


def test_commit_describes_the_checkout_including_uncommitted_changes(monkeypatch) -> None:
    calls = []

    def describe(command: list[str], **_: Any) -> subprocess.CompletedProcess:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="v1.1.0-3-g042ff9c-dirty\n")

    monkeypatch.setattr(provenance.subprocess, "run", describe)
    assert provenance.commit() == "v1.1.0-3-g042ff9c-dirty"
    assert calls[0][-3:] == ["describe", "--always", "--dirty"]


@pytest.mark.parametrize(
    "error", [FileNotFoundError("git"), subprocess.CalledProcessError(128, "git")]
)
def test_commit_is_none_without_git_or_a_repository(monkeypatch, error: Exception) -> None:
    def fail(*_: Any, **__: Any) -> None:
        raise error

    monkeypatch.setattr(provenance.subprocess, "run", fail)
    assert provenance.commit() is None
