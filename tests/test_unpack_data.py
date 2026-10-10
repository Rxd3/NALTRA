"""Restoring the committed gzip release into manifest-verified JSONL files."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest
from scripts import unpack_data

CONTENT = b'{"id": "cordis:test:0", "labels_direct": ["A1"]}\n'


def release(tmp_path: Path, content: bytes = CONTENT) -> Path:
    """One split directory holding test.jsonl.gz and the manifest that lists test.jsonl."""
    split = tmp_path / "processed/en"
    split.mkdir(parents=True)
    (split / "test.jsonl.gz").write_bytes(gzip.compress(content))
    entry = {
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
        "record_count": 1,
    }
    manifest = {"output_files": {"test.jsonl": entry}}
    (split / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path / "processed"


def run(root: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, dict]:
    code = unpack_data.main(["--root", str(root)])
    return code, json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def edit_manifest(root: Path, change) -> None:
    path = root / "en/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    change(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_unpack_restores_the_verified_jsonl(tmp_path, capsys) -> None:
    root = release(tmp_path)
    assert run(root, capsys) == (0, {"status": "success", "unpacked": 1, "skipped": 0})
    assert (root / "en/test.jsonl").read_bytes() == CONTENT
    assert sorted(p.name for p in (root / "en").iterdir()) == [
        "manifest.json",
        "test.jsonl",
        "test.jsonl.gz",
    ]


def test_second_run_skips_files_that_already_match(tmp_path, capsys) -> None:
    root = release(tmp_path)
    run(root, capsys)
    assert run(root, capsys) == (0, {"status": "success", "unpacked": 0, "skipped": 1})


def test_stale_jsonl_is_replaced(tmp_path, capsys) -> None:
    root = release(tmp_path)
    (root / "en/test.jsonl").write_bytes(b"stale\n")
    assert run(root, capsys) == (0, {"status": "success", "unpacked": 1, "skipped": 0})
    assert (root / "en/test.jsonl").read_bytes() == CONTENT


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m["output_files"]["test.jsonl"].update(sha256="0" * 64),
        lambda m: m["output_files"]["test.jsonl"].update(size_bytes=1),
        lambda m: m["output_files"].pop("test.jsonl"),
        lambda m: m["output_files"]["test.jsonl"].update(size_bytes="17"),
        lambda m: m["output_files"]["test.jsonl"].update(size_bytes=None),
        lambda m: m.update(output_files=list(m["output_files"].values())),
        lambda m: m.clear(),
        lambda m: m["output_files"].update({"test.jsonl": "sha256"}),
        lambda m: m["output_files"]["test.jsonl"].pop("sha256"),
        lambda m: m["output_files"]["test.jsonl"].update(sha256=None),
    ],
    ids=[
        "tampered_hash",
        "wrong_size",
        "missing_entry",
        "text_size",
        "null_size",
        "list_of_outputs",
        "no_outputs",
        "text_entry",
        "missing_hash",
        "null_hash",
    ],
)
def test_manifest_mismatch_fails_without_leaving_a_jsonl(tmp_path, capsys, change) -> None:
    root = release(tmp_path)
    edit_manifest(root, change)
    code, status = run(root, capsys)
    assert (code, status["status"]) == (1, "failed")
    assert "manifest.json" in status["error"]
    assert sorted(p.name for p in (root / "en").iterdir()) == ["manifest.json", "test.jsonl.gz"]


def test_archive_larger_than_its_manifest_entry_is_cut_off_before_hashing(
    tmp_path, capsys, monkeypatch
) -> None:
    root = release(tmp_path)
    # Same manifest entry, but the archive now inflates to about 8 MiB of padding.
    (root / "en/test.jsonl.gz").write_bytes(gzip.compress(CONTENT + b" " * (8 << 20)))
    monkeypatch.setattr(
        unpack_data, "compute_file_sha256", lambda path: pytest.fail(f"hashed {path}")
    )
    code, status = run(root, capsys)
    assert (code, status["status"]) == (1, "failed")
    assert str(len(CONTENT)) in status["error"]
    assert sorted(p.name for p in (root / "en").iterdir()) == ["manifest.json", "test.jsonl.gz"]


def test_missing_manifest_fails(tmp_path, capsys) -> None:
    root = release(tmp_path)
    (root / "en/manifest.json").unlink()
    code, status = run(root, capsys)
    assert (code, status["status"]) == (1, "failed")
    assert not (root / "en/test.jsonl").exists()


def test_manifest_that_is_not_json_is_named_in_the_failure(tmp_path, capsys) -> None:
    root = release(tmp_path)
    (root / "en/manifest.json").write_text("{", encoding="utf-8")
    code, status = run(root, capsys)
    assert (code, status["status"]) == (1, "failed")
    assert "manifest.json" in status["error"]


def test_corrupt_archive_fails_without_leaving_a_jsonl(tmp_path, capsys) -> None:
    root = release(tmp_path)
    archive = root / "en/test.jsonl.gz"
    archive.write_bytes(archive.read_bytes()[:-8])
    code, status = run(root, capsys)
    assert (code, status["status"]) == (1, "failed")
    assert sorted(p.name for p in (root / "en").iterdir()) == ["manifest.json", "test.jsonl.gz"]


def test_empty_root_fails(tmp_path, capsys) -> None:
    (tmp_path / "processed").mkdir()
    code, status = run(tmp_path / "processed", capsys)
    assert (code, status["status"]) == (1, "failed")
