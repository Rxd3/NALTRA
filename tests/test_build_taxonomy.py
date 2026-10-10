"""Rebuilding the committed EuroSciVoc taxonomy from the raw CORDIS archive."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from scripts import build_taxonomy

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = REPO_ROOT / "data/raw/cordis_h2020/cordis-h2020projects-csv.zip"
# The committed files as checked out with CRLF (.gitattributes eol=crlf): the bytes that the
# dataset manifests and the trained models pin.
COMMITTED_SHA256 = {
    "taxonomy.json": "0dda384c528199fc0dd4f170ce2a677bb5009f569802b963b5c9cd6ed1299b84",
    "label_map.json": "4cf881e6d78a9f9b972be861b684417ee2fdd2cab67903fe5ae40192ab4ec20a",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict]:
    code = build_taxonomy.main(argv)
    return code, json.loads(capsys.readouterr().out.strip().splitlines()[-1])


@pytest.mark.skipif(not ARCHIVE.is_file(), reason="raw CORDIS archive is not present")
def test_build_reproduces_the_committed_taxonomy(tmp_path, capsys, posix_text_mode) -> None:
    # Written as on Linux/macOS: the pinned bytes must not depend on the platform.
    # The committed alias order came from an unseeded set, so it cannot be rebuilt; a build
    # whose mapping equals the existing label_map.json leaves that file and its hash alone.
    shutil.copy(REPO_ROOT / "taxonomy/label_map.json", tmp_path)
    assert run(["--output-dir", str(tmp_path)], capsys) == (
        0,
        {
            "status": "success",
            "labels": 586,
            "direct_labels": 473,
            "aliases": 1752,
            "written": ["taxonomy.json"],
        },
    )
    assert {name: sha256(tmp_path / name) for name in COMMITTED_SHA256} == COMMITTED_SHA256


def test_missing_archive_fails_with_a_status_line(tmp_path, capsys) -> None:
    code, status = run(["--archive", str(tmp_path / "missing.zip")], capsys)
    assert code == 1 and status["status"] == "failed"
