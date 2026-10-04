"""Unit tests for NALTRA reproducibility manifest utility."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from naltra.data.loader import save_jsonl
from naltra.data.manifest import (
    compute_file_sha256,
    create_manifest,
    get_environment_metadata,
    get_taxonomy_checksums,
    validate_manifest,
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

    assert loaded["manifest_version"] == "1.1.0"
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


@pytest.fixture
def release(tmp_path: Path) -> Path:
    output = tmp_path / "release"
    source = tmp_path / "source.jsonl"
    save_jsonl([{"id": "source:1", "text": "original"}], source)
    save_jsonl([{"id": "output:1", "text": "generated"}], output / "test.jsonl")
    create_manifest(
        "test_release", output, {"seed": 42}, {"source": "fixture"}, input_files=[source]
    )
    return output


def test_validate_manifest_accepts_intact_release(release: Path) -> None:
    assert (
        validate_manifest(release, "test_release", ["test.jsonl"])["output_files"]["test.jsonl"][
            "record_count"
        ]
        == 1
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("manifest_version", "0.0.0"),
        ("benchmark_name", "other_release"),
        ("created_at_utc", "not a timestamp"),
        ("taxonomy", {}),
        ("environment", {}),
        ("generation_parameters", []),
        ("source_metadata", {}),
        ("input_files", {}),
        ("output_files", {}),
        ("code", {}),
    ],
)
def test_validate_manifest_rejects_bad_metadata(release: Path, field: str, value: object) -> None:
    path = release / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[field] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_manifest(release, "test_release", ["test.jsonl"])


@pytest.mark.parametrize(
    "field,value", [("sha256", "0" * 64), ("size_bytes", 0), ("record_count", 2)]
)
def test_validate_manifest_rejects_stale_output_metadata(
    release: Path, field: str, value: object
) -> None:
    path = release / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["output_files"]["test.jsonl"][field] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        validate_manifest(release, "test_release", ["test.jsonl"])


@pytest.mark.parametrize("target", ["output", "source", "extra", "missing", "json"])
def test_validate_manifest_rejects_changed_release(release: Path, target: str) -> None:
    if target == "output":
        save_jsonl([{"id": "changed"}], release / "test.jsonl")
    elif target == "source":
        save_jsonl([{"id": "changed"}], release.parent / "source.jsonl")
    elif target == "extra":
        save_jsonl([{"id": "unexpected"}], release / "train.jsonl")
    elif target == "missing":
        (release / "test.jsonl").unlink()
    else:
        (release / "manifest.json").write_text("{invalid", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_manifest(release, "test_release", ["test.jsonl"])


def test_create_manifest_does_not_hide_invalid_jsonl(tmp_path: Path) -> None:
    (tmp_path / "bad.jsonl").write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError):
        create_manifest("bad", tmp_path, {"seed": 42})
    assert not (tmp_path / "manifest.json").exists()


def test_aggregate_manifest_covers_nested_outputs(tmp_path: Path) -> None:
    save_jsonl([{"id": "nested"}], tmp_path / "child/test.jsonl")
    manifest = create_manifest("aggregate", tmp_path, {"seed": 42})
    assert manifest["output_files"]["child/test.jsonl"]["record_count"] == 1
