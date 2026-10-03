"""Reproducibility manifest generation for NALTRA benchmark datasets."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from naltra.data.loader import load_jsonl

MANIFEST_SCHEMA_VERSION = "1.0.0"


def compute_file_sha256(path: str | Path) -> str:
    """Compute the SHA-256 hex digest of a file in 64KB chunks."""
    file_path = Path(path)
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_file_md5(path: str | Path) -> str:
    """Compute the MD5 hex digest of a file in 64KB chunks."""
    file_path = Path(path)
    h = hashlib.md5()
    with file_path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_taxonomy_checksums(taxonomy_dir: str | Path = "taxonomy") -> dict[str, str]:
    """Retrieve canonical taxonomy version and file checksums."""
    tax_path = Path(taxonomy_dir)
    taxonomy_file = tax_path / "taxonomy.json"
    label_map_file = tax_path / "label_map.json"

    version = "unknown"
    if taxonomy_file.exists():
        try:
            with taxonomy_file.open(encoding="utf-8") as f:
                data = json.load(f)
                version = data.get("version", "unknown")
        except Exception:
            pass

    return {
        "taxonomy_version": version,
        "taxonomy_sha256": compute_file_sha256(taxonomy_file) if taxonomy_file.exists() else "",
        "label_map_sha256": compute_file_sha256(label_map_file) if label_map_file.exists() else "",
    }


def get_environment_metadata() -> dict[str, Any]:
    """Capture relevant execution environment and package versions."""
    tracked_packages = [
        "datasets",
        "pandas",
        "numpy",
        "pyarrow",
        "torch",
        "transformers",
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


def create_manifest(
    benchmark_name: str,
    output_dir: str | Path,
    generation_parameters: dict[str, Any],
    source_metadata: dict[str, Any] | None = None,
    taxonomy_dir: str | Path = "taxonomy",
) -> dict[str, Any]:
    """Generate a structured reproducibility manifest for an output directory."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Collect hashes of all output .jsonl files (strictly excluding manifest.json itself)
    output_files: dict[str, dict[str, Any]] = {}
    for p in sorted(target_dir.glob("*.jsonl")):
        record_count = 0
        try:
            recs = load_jsonl(p)
            record_count = len(recs)
        except Exception:
            pass

        output_files[p.name] = {
            "sha256": compute_file_sha256(p),
            "size_bytes": p.stat().st_size,
            "record_count": record_count,
        }

    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_SCHEMA_VERSION,
        "benchmark_name": benchmark_name,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "generation_parameters": generation_parameters,
        "taxonomy": get_taxonomy_checksums(taxonomy_dir),
        "source_metadata": source_metadata or {},
        "environment": get_environment_metadata(),
        "output_files": output_files,
    }

    manifest_file = target_dir / "manifest.json"
    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    return manifest
