"""Restore the committed compressed release: every *.jsonl.gz becomes its verified .jsonl.

Each archive is decompressed into a temporary sibling whose SHA-256 and size must match the
sibling manifest.json entry before it atomically replaces the .jsonl, so a tampered archive,
a stale manifest or an interrupted run never leaves a partial or unverified dataset behind.
Decompression stops once the output grows past the size the manifest lists.
A .jsonl that already matches its manifest entry is skipped.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import zlib
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from naltra.data.manifest import compute_file_sha256  # noqa: E402

CHUNK_BYTES = 1 << 20


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", default="data/processed", help="Searched for *.jsonl.gz")
    return result


def unpack(archive: Path) -> bool:
    """Restore one archive; False when its .jsonl already matches the manifest."""
    target = archive.with_suffix("")
    manifest = archive.parent / "manifest.json"
    try:
        outputs = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{manifest} is not valid JSON ({exc}).") from None
    outputs = outputs.get("output_files") if isinstance(outputs, dict) else None
    if not isinstance(outputs, dict):
        raise ValueError(f"{manifest} has no output_files object.")
    entry = outputs.get(target.name)
    if not entry:
        raise ValueError(f"{manifest} does not list {target.name}.")
    if not isinstance(entry, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256"))):
        raise ValueError(f"{manifest} lists no valid SHA-256 for {target.name}.")
    if target.exists() and compute_file_sha256(target) == entry["sha256"]:
        return False
    size = entry.get("size_bytes")
    if type(size) is not int or size < 0:
        raise ValueError(f"{manifest} lists no valid size for {target.name}.")
    temporary = target.with_name(target.name + ".tmp")
    try:
        with gzip.open(archive, "rb") as source, temporary.open("wb") as sink:
            written = 0
            # Stop as soon as the output outgrows its manifest size, so a tampered archive
            # cannot fill the disk before the checks below run.
            while chunk := source.read(CHUNK_BYTES):
                written += len(chunk)
                if written > size:
                    raise ValueError(
                        f"{archive} inflates past the {size} bytes "
                        f"{manifest} lists for {target.name}."
                    )
                sink.write(chunk)
        if compute_file_sha256(temporary) != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {target} against {manifest}.")
        if temporary.stat().st_size != size:
            raise ValueError(f"Size mismatch for {target} against {manifest}.")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = REPO_ROOT / args.root  # an absolute --root replaces REPO_ROOT
    try:
        archives = sorted(root.rglob("*.jsonl.gz"))
        if not archives:
            raise ValueError(f"No *.jsonl.gz under {root}.")
        restored = [unpack(archive) for archive in archives]
    except (OSError, EOFError, zlib.error, ValueError, KeyError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), flush=True)
        return 1
    unpacked, skipped = sum(restored), restored.count(False)
    print(json.dumps({"status": "success", "unpacked": unpacked, "skipped": skipped}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
