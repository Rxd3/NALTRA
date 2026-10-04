"""Regression checks for release auditing and preparation entry points."""

from __future__ import annotations

import copy
import shutil
import sys
from pathlib import Path

import pytest
from scripts import prepare_data
from scripts import validate_datasets as audit

from naltra.data.loader import load_jsonl, save_jsonl
from naltra.data.manifest import create_manifest
from naltra.data.ood import generate_near_ood_benchmarks
from naltra.data.preprocessing import load_canonical_label_ids

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOPICS = (
    "arts_culture_entertainment_media",
    "geography",
    "health",
    "politics",
    "science_technology",
    "sport",
    "travel",
)


def record(split: str, index: int = 0, language: str = "en", label: str = "politics") -> dict:
    return {
        "id": f"sib200:{split}:{index}:{language}",
        "pair_id": f"sib200:{split}:{index}",
        "source_id": f"{split}:{index}",
        "text": f"Example sentence {split} {index} in {language}.",
        "labels": [label],
        "language": language,
        "source": "sib200",
        "license": "CC BY-SA 4.0",
        "split": split,
    }


@pytest.mark.parametrize("mutation", ["duplicate", "split", "missing_field", "unknown_label"])
def test_audit_loader_rejects_bad_records(tmp_path: Path, mutation: str) -> None:
    records = [record("validation")]
    if mutation == "duplicate":
        records.append(copy.deepcopy(records[0]))
    elif mutation == "split":
        records[0]["split"] = "test"
    elif mutation == "missing_field":
        del records[0]["text"]
    else:
        records[0]["labels"] = ["unknown"]
    path = save_jsonl(records, tmp_path / "validation.jsonl")
    with pytest.raises(ValueError):
        audit.load_audit_records(path)


@pytest.mark.parametrize("key", ["id", "text", "pair_id"])
def test_audit_rejects_cross_partition_contamination(key: str) -> None:
    train, test = record("train"), record("test")
    test[key] = train[key]
    with pytest.raises(ValueError, match="contamination"):
        audit.check_disjoint_partitions({"train": [train], "test": [test]})


@pytest.fixture
def near_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    shutil.copytree(
        PROJECT_ROOT / "taxonomy",
        tmp_path / "taxonomy",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
    source = tmp_path / "data/processed/sib200"
    for split in ("train", "validation", "test"):
        save_jsonl(
            [
                record(split, i, lang, topic)
                for i, topic in enumerate(TOPICS)
                for lang in ("en", "tr")
            ],
            source / f"{split}.jsonl",
        )
    output = tmp_path / "data/ood/near/sib200"
    generate_near_ood_benchmarks(source, output)
    return output


def test_near_ood_audit_accepts_generated_release(near_release: Path) -> None:
    assert audit.validate_near_ood(load_canonical_label_ids())


@pytest.mark.parametrize(
    "mutation", ["duplicate", "split", "heldout_id", "changed_source", "cross_split_content"]
)
def test_near_ood_audit_rejects_corruption_even_with_fresh_manifest(
    near_release: Path, mutation: str
) -> None:
    fold = near_release / "politics"
    path = fold / "validation_id.jsonl"
    records = load_jsonl(path)
    if mutation == "duplicate":
        records.append(copy.deepcopy(records[0]))
    elif mutation == "split":
        records[0]["split"] = "test"
    elif mutation == "heldout_id":
        records[0]["labels"] = ["politics"]
    elif mutation == "changed_source":
        records[0]["text"] = "Changed source text."
    else:
        records[0]["text"] = load_jsonl(fold / "train_id.jsonl")[0]["text"]
    save_jsonl(records, path)
    source = audit.REPO_ROOT / "data/processed/sib200"
    create_manifest(
        "near_ood_sib200_politics",
        fold,
        {"heldout_topic": "politics"},
        {"sib200_dir": str(source)},
        input_files=[source / f"{s}.jsonl" for s in ("train", "validation", "test")],
    )
    with pytest.raises(ValueError):
        audit.validate_near_ood(load_canonical_label_ids())


def test_prepare_all_runs_code_switch_summary_then_ood(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The original list/dict crash occurred after generation, before the OOD step."""
    monkeypatch.chdir(tmp_path)
    save_jsonl(
        [record("validation", language=lang) for lang in ("en", "tr")],
        tmp_path / "data/processed/sib200/validation.jsonl",
    )
    steps = []
    for name in ("sib200", "multifin", "mn_ds", "noisy", "ood"):
        monkeypatch.setattr(prepare_data, f"prepare_{name}", lambda name=name: steps.append(name))
    monkeypatch.setattr(
        prepare_data,
        "generate_code_switch_benchmarks",
        lambda **kwargs: {"validation": 1, "test": 0},
    )
    monkeypatch.setattr(sys, "argv", ["prepare_data.py", "--dataset", "all"])
    prepare_data.main()
    assert steps == ["sib200", "multifin", "mn_ds", "noisy", "ood"]
    assert "Synthetic Mixed:" in capsys.readouterr().out


def test_audit_reports_missing_datasets_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    shutil.copytree(
        PROJECT_ROOT / "taxonomy",
        tmp_path / "taxonomy",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
    assert audit.main() == 1
    output = capsys.readouterr().out
    assert "AUDIT SUMMARY" in output
    assert "Traceback" not in output


def test_installed_console_command_runs_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    import naltra.data.code_switching as code_switching
    from naltra.cli import prepare_data as console

    calls = []
    monkeypatch.setattr(
        code_switching, "generate_code_switch_benchmarks", lambda: calls.append("code_switch")
    )
    monkeypatch.setattr(sys, "argv", ["naltra-prepare-data", "--dataset", "code_switch"])
    console()
    assert calls == ["code_switch"]
