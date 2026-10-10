"""Write fixed pair-hash samples of evaluation sets for members too slow to score in full.

Each set keeps the records whose pair_id is among the --pairs smallest SHA-256 hashes in
that file. EN, TR and code-switch files share pair_ids, so every language keeps the same
projects, and the selection depends only on the ids (not on file order or label counts).
Every member then scores the sampled files as named sets, e.g. en_test_kev=<path>.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.predict_all import parse_set, resolve_path  # noqa: E402

from naltra.data.loader import load_jsonl, save_jsonl  # noqa: E402


def pair_hash(pair_id: str) -> str:
    return hashlib.sha256(pair_id.encode("utf-8")).hexdigest()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--sets", nargs="+", type=parse_set, required=True)
    result.add_argument("--pairs", type=int, required=True, help="Pairs kept per set")
    result.add_argument("--output-dir", default="data/samples")
    return result


def sample(records: list[dict], pairs: int) -> list[dict]:
    if any(not r.get("pair_id") for r in records):
        raise ValueError("Every record needs a pair_id to be sampled consistently.")
    kept = set(sorted({r["pair_id"] for r in records}, key=pair_hash)[:pairs])
    return [r for r in records if r["pair_id"] in kept]


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.pairs < 1:
        parser().error("--pairs must be at least 1")
    try:
        samples = {name: sample(load_jsonl(path), args.pairs) for name, path in args.sets}
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), flush=True)
        return 1
    output = resolve_path(args.output_dir)
    report = {}
    for name, records in samples.items():
        target = save_jsonl(records, output / f"{name}.jsonl")
        report[name] = {"records": len(records), "path": str(target)}
    print(json.dumps({"status": "success", "pairs": args.pairs, "sets": report}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
