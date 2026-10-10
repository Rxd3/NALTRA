"""Release writers emit CRLF on every platform: the bytes the shipped SHA-256 pins cover."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from naltra.data import preprocessing
from naltra.data.loader import save_jsonl
from naltra.data.manifest import compute_file_sha256
from naltra.data.noise import generate_noisy_benchmarks


def test_save_jsonl_writes_crlf_on_every_platform(tmp_path: Path, posix_text_mode) -> None:
    path = save_jsonl([{"id": "a", "text": "ğ"}, {"id": "b"}], tmp_path / "out.jsonl")
    assert path.read_bytes() == '{"id": "a", "text": "ğ"}\r\n{"id": "b"}\r\n'.encode()


def test_noisy_release_writes_crlf_on_every_platform(tmp_path: Path, posix_text_mode) -> None:
    record = {"id": "cordis:1", "text": "Solar cells convert light.", "language": "en"}
    save_jsonl([record], tmp_path / "processed/cordis_h2020/en/test.jsonl")

    generate_noisy_benchmarks(tmp_path / "processed", tmp_path / "noisy", ["cordis_h2020/en"])

    written = list((tmp_path / "noisy").rglob("*.json*"))
    assert sorted(p.name for p in written) == ["manifest.json", "manifest.json", "test.jsonl"]
    for path in written:
        data = path.read_bytes()
        assert data.endswith(b"\r\n") and data.count(b"\n") == data.count(b"\r\n"), path


def test_split_map_writes_crlf_on_every_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, posix_text_mode
) -> None:
    label = {"id": "optics", "parent": None, "euroscivoc_code": "/23", "is_direct_supported": True}
    (tmp_path / "taxonomy.json").write_text(json.dumps({"version": "0.4.0", "labels": [label]}))
    (tmp_path / "label_map.json").write_text(json.dumps({"aliases": {"/23": "optics"}}))
    rows = [";".join([str(i)] + [""] * 14 + [f"Optics study {i}."] + [""] * 6) for i in range(20)]
    (tmp_path / "project.csv").write_text("\n".join(["header", *rows]))
    (tmp_path / "euroSciVoc.csv").write_text("\n".join(["h", *(f"/23;;O;;{i}" for i in range(20))]))
    lock = {
        "project_sha256": compute_file_sha256(tmp_path / "project.csv"),
        "euroscivoc_sha256": compute_file_sha256(tmp_path / "euroSciVoc.csv"),
    }
    monkeypatch.setattr(preprocessing, "get_source_config", lambda name: lock)

    preprocessing.process_cordis_h2020(
        tmp_path,
        tmp_path / "en",
        tmp_path / "splits",
        tmp_path / "taxonomy.json",
        tmp_path / "label_map.json",
    )

    data = (tmp_path / "splits/project_splits.json").read_bytes()
    assert b"\n" in data and data.count(b"\n") == data.count(b"\r\n")
