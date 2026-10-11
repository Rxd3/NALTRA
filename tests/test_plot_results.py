"""Archived chart inputs must stay comparable and traceable without loading models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from scripts import plot_results

ROOT = Path(__file__).resolve().parents[1]


def read_sources() -> dict:
    return {
        key: json.loads((ROOT / path).read_text(encoding="utf-8"))
        for key, path in plot_results.DEFAULTS.items()
    }


@pytest.mark.parametrize(
    ("source", "keys", "replacement", "message"),
    [
        ("sample", ("taxonomy_sha256",), "other", "taxonomy"),
        ("sample", ("label_count",), 1, "label counts"),
        ("benchmark", ("eval_set_sha256",), "other", "test inputs"),
        ("benchmark", ("models", "svm", "artifact_files_sha256"), "other", "artifacts"),
        ("benchmark", ("models", "hybrid_knn", "device"), "cpu+encoder:cuda:0", "CPU"),
        ("full", ("metrics", "en_test", "svm", "micro_f1"), float("nan"), "finite"),
        ("ood", ("results", "svm", "en", "auroc_ci_high"), 0.1, "finite"),
        (
            "full",
            (
                "significance",
                "cross_condition",
                "svm",
                "en_test->tr_test",
                "micro_f1",
                "difference",
            ),
            0.5,
            "does not match",
        ),
    ],
)
def test_invalid_inputs_do_not_create_output(tmp_path, source, keys, replacement, message):
    summaries = read_sources()
    row = summaries[source]
    for key in keys[:-1]:
        row = row[key]
    row[keys[-1]] = replacement
    modified = tmp_path / "modified.json"
    modified.write_text(json.dumps(summaries[source]), encoding="utf-8")
    output = tmp_path / "charts"
    args = ["--output-dir", str(output)]
    for key, default in plot_results.DEFAULTS.items():
        args.extend([f"--{key}", str(modified if key == source else ROOT / default)])
    with pytest.raises(ValueError, match=message):
        plot_results.main(args)
    assert not output.exists()


def test_complete_exports_record_exact_source_checksums(tmp_path):
    args = ["--output-dir", str(tmp_path)]
    for key, default in plot_results.DEFAULTS.items():
        args.extend([f"--{key}", str(ROOT / default)])
    assert plot_results.main(args) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["figures"]) == 16
    assert set(manifest["figures"]) == {
        f"{name}.{ext}" for name in plot_results.FIGURES for ext in ("png", "svg")
    }
    for key, default in plot_results.DEFAULTS.items():
        raw = (ROOT / default).read_bytes()
        assert manifest["sources"][key]["sha256"] == hashlib.sha256(raw).hexdigest()
    for filename in manifest["figures"]:
        raw = (tmp_path / filename).read_bytes()
        if filename.endswith(".png"):
            assert raw.startswith(b"\x89PNG\r\n\x1a\n")
        else:
            assert b"<svg" in raw
