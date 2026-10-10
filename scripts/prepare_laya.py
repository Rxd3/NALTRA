"""Build the portable local Laya artifact from the pinned checkpoint in configs/models/laya.yaml.

Laya is scored zero-shot, so there is nothing to train: the artifact holds the pinned
checkpoint files, the 473 topic questions and their SHA-256 values, and `predict_all.py`
loads it like any trained member. Downloads the checkpoint once (about 680 MB).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from naltra.models.laya import LayaModel  # noqa: E402
from naltra.utils.config import load_yaml  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--output-dir", required=True, help="artifact root; Laya goes in <dir>/laya"
    )
    result.add_argument("--config", default="configs/models/laya.yaml")
    result.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    target = (REPO_ROOT / args.output_dir / "laya").resolve()
    if target.exists() and any(target.iterdir()):
        parser().error(f"{target} already holds an artifact; choose a new --output-dir.")
    try:
        config = load_yaml(REPO_ROOT / args.config) | {"device": args.device}
        LayaModel(config).save(target)
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), flush=True)
        return 1
    print(json.dumps({"status": "success", "artifact": str(target)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
