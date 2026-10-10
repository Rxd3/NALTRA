"""Score each member with and without the release's known translation exceptions.

The exception records (text/alignment mismatch and translation-QA failures) are read from
the training provenance that --prebuilt-release stores in an artifact, mapped to projects
by pair_id, and removed from every evaluated set (EN, TR, code-switch, noisy). Dumps are
loaded through the benchmark's validated loader and must be the ones the summary scored, and
the artifact must be the one that produced a scored family's dumps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.make_figures import scored_split  # noqa: E402
from scripts.predict_all import (  # noqa: E402
    artifact_hashes,
    parse_set,
    repeated_names,
    resolve_path,
)

from naltra.data.manifest import compute_file_sha256  # noqa: E402
from naltra.evaluation.matrix import binarize, flat_metrics  # noqa: E402
from naltra.utils.provenance import repo_path  # noqa: E402

ISSUES = ("alignment_text_mismatch", "qa_failed")
INVENTORY = ("artifact_files_sha256", "artifact_files")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--summary", required=True, help="evaluate_ensemble summary.json")
    result.add_argument("--predictions", required=True)
    result.add_argument("--sets", nargs="+", type=parse_set, required=True)
    result.add_argument(
        "--artifact", required=True, help="naltra.json of a --prebuilt-release model"
    )
    result.add_argument("--output", default="results/metrics/translation_exceptions.json")
    return result


def exception_pairs(payload: dict) -> set[str]:
    """pair_ids of records flagged by the prebuilt audit (record id is pair_id:language)."""
    issues = payload["metadata"]["translation_issues"]
    return {record.rsplit(":", 1)[0] for key in ISSUES for record in issues[key]}


def bound_family(
    artifact: Path, digest: str, files: dict, dumps: dict[str, dict], families: list[str]
) -> str:
    """The family whose every scored dump records this naltra.json and, if kept, its files."""
    if files["artifact_sha256"] != digest:
        raise ValueError(f"{artifact} changed while it was being read; re-run.")
    for family in families:
        origins = [origin for key, origin in dumps.items() if key.rpartition("/")[0] == family]
        if all(origin.get("artifact_sha256") == digest for origin in origins) and all(
            origin.get(f) in (None, files[f]) for origin in origins for f in INVENTORY
        ):
            return family
    raise ValueError(
        f"{artifact}: no scored family was produced by this artifact; pass the naltra.json "
        "whose model wrote the summary's dumps, or re-run predict_all and evaluate_ensemble."
    )


def scores(truth: np.ndarray, selected: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if not mask.any():
        return {"records": 0}
    metrics = flat_metrics(truth[mask], selected[mask])
    return {
        "records": int(mask.sum()),
        "micro_f1": metrics["micro_f1"],
        "macro_f1": metrics["macro_f1"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if repeated := repeated_names(args.sets):
        parser().error(f"Each set name must be unique; repeated: {', '.join(repeated)}")
    try:
        summary_path, artifact = resolve_path(args.summary), resolve_path(args.artifact)
        if artifact.name != "naltra.json":
            raise ValueError(f"--artifact must be a model's naltra.json, not {artifact.name}.")
        # Parse and hash the same bytes, so the exceptions belong to the recorded digest.
        raw = artifact.read_bytes()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        flagged = exception_pairs(json.loads(raw))
        families = summary["members"] + summary.get("baselines", [])
        report = {"exception_pairs": len(flagged), "inputs": {}, "sets": {}}
        scored = {}
        for name, path in args.sets:
            data = scored_split(summary, resolve_path(args.predictions), families, (name, path))
            scored |= {key: summary["dumps"][key] for key in data["dumps"]}
            report["inputs"][name] = {
                "input_sha256": data["dumps"][f"{families[0]}/{name}"]["input_sha256"],
                "npz_sha256": {f: data["dumps"][f"{f}/{name}"]["npz_sha256"] for f in families},
            }
            truth = binarize([r["labels_direct"] for r in data["rows"]], data["labels"])
            exception = np.array([r["pair_id"] in flagged for r in data["rows"]])
            report["sets"][name] = {
                family: {
                    kind: scores(truth, data["scores"][family] >= summary["thresholds"][family], m)
                    for kind, m in (
                        ("all", np.ones_like(exception)),
                        ("without_exceptions", ~exception),
                        ("exceptions_only", exception),
                    )
                }
                for family in families
            }
        digest, files = hashlib.sha256(raw).hexdigest(), artifact_hashes(artifact.parent)
        report = {
            "artifact": {
                "path": repo_path(artifact),
                "sha256": digest,
                "artifact_files_sha256": files["artifact_files_sha256"],
                "family": bound_family(artifact, digest, files, scored, families),
            },
            "summary": {
                "path": repo_path(summary_path),
                "sha256": compute_file_sha256(summary_path),
            },
        } | report
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), flush=True)
        return 1
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "success", "output": str(output)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
