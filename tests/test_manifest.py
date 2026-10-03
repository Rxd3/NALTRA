"""Unit tests for NALTRA reproducibility manifest utility."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from naltra.data.loader import save_jsonl
from naltra.data.manifest import (
    compute_file_sha256,
    create_manifest,
    get_environment_metadata,
    get_taxonomy_checksums,
)


def test_compute_file_sha256(tmp_path: Path) -> None:
    test_file = tmp_path / "sample.txt"
    test_file.write_text("Hello, NALTRA!", encoding="utf-8")

    sha = compute_file_sha256(test_file)
    assert isinstance(sha, str)
    assert len(sha) == 64
    expected = hashlib.sha256(b"Hello, NALTRA!").hexdigest()
    assert sha == expected


def test_get_taxonomy_checksums() -> None:
    meta = get_taxonomy_checksums()
    assert meta["taxonomy_version"] == "0.2.0"
    assert "taxonomy_sha256" in meta
    assert len(meta["taxonomy_sha256"]) == 64
    assert "label_map_sha256" in meta
    assert len(meta["label_map_sha256"]) == 64


def test_get_environment_metadata() -> None:
    env = get_environment_metadata()
    assert "python_version" in env
    assert "platform" in env
    assert "package_versions" in env
    assert isinstance(env["package_versions"], dict)


def test_create_manifest_required_fields(tmp_path: Path) -> None:
    sample_data = [{"id": "rec:1", "text": "sample text", "language": "en"}]
    save_jsonl(sample_data, tmp_path / "sample.jsonl")

    manifest = create_manifest(
        benchmark_name="test_benchmark",
        output_dir=tmp_path,
        generation_parameters={"param1": "val1", "seed": 42},
        source_metadata={"source": "test_src"},
    )
    assert manifest["benchmark_name"] == "test_benchmark"

    manifest_file = tmp_path / "manifest.json"
    assert manifest_file.exists()

    with open(manifest_file, encoding="utf-8") as f:
        loaded = json.load(f)

    for req_key in [
        "manifest_version",
        "benchmark_name",
        "created_at_utc",
        "generation_parameters",
        "source_metadata",
        "taxonomy",
        "environment",
        "output_files",
    ]:
        assert req_key in loaded, f"Missing required manifest field: {req_key}"

    assert loaded["manifest_version"] == "1.0.0"
    assert loaded["benchmark_name"] == "test_benchmark"
    assert loaded["generation_parameters"]["seed"] == 42
    assert "sample.jsonl" in loaded["output_files"]
    assert loaded["output_files"]["sample.jsonl"]["record_count"] == 1
    assert "sha256" in loaded["output_files"]["sample.jsonl"]


def test_create_manifest_determinism_output_files(tmp_path: Path) -> None:
    sample_data = [
        {"id": "rec:1", "text": "hello"},
        {"id": "rec:2", "text": "world"},
    ]
    save_jsonl(sample_data, tmp_path / "data.jsonl")

    m1 = create_manifest(
        benchmark_name="bench_1",
        output_dir=tmp_path,
        generation_parameters={"seed": 100},
    )

    m2 = create_manifest(
        benchmark_name="bench_1",
        output_dir=tmp_path,
        generation_parameters={"seed": 100},
    )

    assert m1["output_files"] == m2["output_files"]
    assert m1["taxonomy"] == m2["taxonomy"]
    assert m1["generation_parameters"] == m2["generation_parameters"]
