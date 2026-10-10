"""Reproducibility manifest generation for NALTRA benchmark datasets."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from naltra.data.loader import load_jsonl

MANIFEST_SCHEMA_VERSION = "1.1.0"
REPO_ROOT = Path(__file__).resolve().parents[3]


class GenerationCodeMismatch(ValueError):
    """The checkout's data-generation code differs from the code a manifest recorded."""


def relative_path(path: str | Path, start: str | Path) -> str:
    """``path`` relative to the directory ``start``, with forward slashes.

    Manifest readers resolve every recorded source against the manifest's own directory, so a
    path os.path.relpath cannot relate to it (another Windows drive or UNC share) is refused.
    """
    source, base = Path(path).resolve(), Path(start).resolve()
    try:
        return Path(os.path.relpath(source, base)).as_posix()
    except ValueError as error:
        raise ValueError(
            f"Cannot record {source} relative to {base}: they are on different drives or "
            "shares. Keep sources, caches and outputs on one drive."
        ) from error


def get_source_config(name: str) -> dict[str, str]:
    """Read the versioned source lock used by every preparation entry point."""
    import yaml

    with (REPO_ROOT / "configs/data.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
        return config["translation"] if name == "translation" else config["sources"][name]


def compute_file_sha256(path: str | Path) -> str:
    """Compute the SHA-256 hex digest of a file in 64KB chunks."""
    file_path = Path(path)
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_taxonomy_checksums(taxonomy_dir: str | Path = "taxonomy") -> dict[str, str]:
    """Retrieve canonical taxonomy version and file checksums."""
    tax_path = Path(taxonomy_dir)
    taxonomy_file = tax_path / "taxonomy.json"
    label_map_file = tax_path / "label_map.json"

    with taxonomy_file.open(encoding="utf-8") as f:
        version = json.load(f)["version"]

    return {
        "taxonomy_version": version,
        "taxonomy_sha256": compute_file_sha256(taxonomy_file),
        "label_map_sha256": compute_file_sha256(label_map_file),
    }


def get_environment_metadata() -> dict[str, Any]:
    """Capture relevant execution environment and package versions."""
    tracked_packages = [
        "numpy",
        "torch",
        "transformers",
        "scikit-learn",
        "pyyaml",
        "huggingface-hub",
        "httpx",
    ]
    package_versions = {}
    for pkg in tracked_packages:
        try:
            package_versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass

    return {
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "package_versions": package_versions,
    }


def get_generation_code_hashes() -> dict[str, str]:
    paths = list((REPO_ROOT / "src/naltra/data").glob("*.py")) + [
        REPO_ROOT / "src/naltra/pipeline/preprocessing.py",
        REPO_ROOT / "scripts/prepare_data.py",
        REPO_ROOT / "src/naltra/cli.py",
        REPO_ROOT / "configs/data.yaml",
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "requirements.txt",
    ]
    return {p.relative_to(REPO_ROOT).as_posix(): compute_file_sha256(p) for p in paths}


def create_manifest(
    benchmark_name: str,
    output_dir: str | Path,
    generation_parameters: dict[str, Any],
    source_metadata: dict[str, Any] | None = None,
    taxonomy_dir: str | Path = "taxonomy",
    input_files: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Generate a structured reproducibility manifest for an output directory."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Include nested outputs, so the aggregate noisy manifest verifies every dataset.
    output_files: dict[str, dict[str, Any]] = {}
    for p in sorted(target_dir.rglob("*.jsonl")):
        output_files[p.relative_to(target_dir).as_posix()] = {
            "sha256": compute_file_sha256(p),
            "size_bytes": p.stat().st_size,
            "record_count": len(load_jsonl(p)),
        }

    source_files = {
        relative_path(p, target_dir): compute_file_sha256(p) for p in sorted(map(Path, input_files))
    }
    code_hashes = get_generation_code_hashes()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )

    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_SCHEMA_VERSION,
        "benchmark_name": benchmark_name,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "generation_parameters": generation_parameters,
        "taxonomy": get_taxonomy_checksums(taxonomy_dir),
        "source_metadata": source_metadata or {},
        "environment": get_environment_metadata(),
        "output_files": output_files,
        "input_files": source_files,
        "code": {"git_revision": revision, "dirty": dirty, "files": code_hashes},
    }

    manifest_file = target_dir / "manifest.json"
    with manifest_file.open("w", encoding="utf-8", newline="\r\n") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    return manifest


def validate_manifest(
    output_dir: str | Path,
    benchmark_name: str,
    expected_files: Sequence[str],
    taxonomy_dir: str | Path = "taxonomy",
) -> dict[str, Any]:
    """Reject malformed, stale or incomplete releases, including changed source files."""
    target = Path(output_dir)
    with (target / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if (
        not isinstance(manifest, dict)
        or manifest.get("manifest_version") != MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported manifest schema; regenerate the dataset.")
    if manifest.get("benchmark_name") != benchmark_name:
        raise ValueError(f"Manifest benchmark must be {benchmark_name!r}.")
    created = datetime.fromisoformat(manifest["created_at_utc"])
    if created.utcoffset() is None or created.utcoffset().total_seconds() != 0:
        raise ValueError("Manifest creation time must be UTC.")
    for field in (
        "generation_parameters",
        "source_metadata",
        "environment",
        "output_files",
        "input_files",
        "code",
    ):
        if not isinstance(manifest.get(field), dict) or not manifest[field]:
            raise ValueError(f"Manifest {field} must be a non-empty object.")
    environment = manifest["environment"]
    if (
        not environment.get("python_version")
        or not environment.get("platform")
        or not isinstance(environment.get("package_versions"), dict)
    ):
        raise ValueError("Manifest environment metadata is incomplete.")
    if manifest.get("taxonomy") != get_taxonomy_checksums(taxonomy_dir):
        raise ValueError("Manifest taxonomy version or hashes do not match current taxonomy.")
    actual_files = {p.relative_to(target).as_posix() for p in target.rglob("*.jsonl")}
    if set(manifest["output_files"]) != set(expected_files) or actual_files != set(expected_files):
        raise ValueError("Manifest output file inventory does not match the expected release.")
    for name, meta in manifest["output_files"].items():
        path = target / name
        if not isinstance(meta, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(meta.get("sha256"))):
            raise ValueError(f"Invalid manifest SHA-256 for {name}.")
        if type(meta.get("size_bytes")) is not int or type(meta.get("record_count")) is not int:
            raise ValueError(f"Invalid manifest size/count for {name}.")
        if meta != {
            "sha256": compute_file_sha256(path),
            "size_bytes": path.stat().st_size,
            "record_count": len(load_jsonl(path)),
        }:
            raise ValueError(f"Manifest hash, size or record count mismatch for {name}.")
    for name, digest in manifest["input_files"].items():
        if (
            not re.fullmatch(r"[0-9a-f]{64}", str(digest))
            or compute_file_sha256(target / name) != digest
        ):
            raise ValueError(f"Manifest source hash mismatch for {name}.")
    code = manifest["code"]
    if (
        not re.fullmatch(r"[0-9a-f]{40}", str(code.get("git_revision")))
        or type(code.get("dirty")) is not bool
    ):
        raise ValueError("Invalid manifest code revision.")
    if code.get("files") != get_generation_code_hashes():
        raise GenerationCodeMismatch("Manifest generation code inventory or hashes do not match.")
    return manifest
