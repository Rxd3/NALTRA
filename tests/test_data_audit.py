"""Regression checks for the CORDIS release and the retained integrity safeguards."""

from __future__ import annotations

import copy
import csv
import hashlib
import json

import pytest
from scripts import validate_datasets as audit

from naltra.data import preprocessing
from naltra.data.loader import load_jsonl, save_jsonl
from naltra.data.manifest import compute_file_sha256, create_manifest
from naltra.data.translation import MockTranslator, TranslationCache, translate_cordis_dataset
from naltra.data.validation import (
    check_disjoint_partitions,
    load_audit_records,
    validate_cordis_release,
)


@pytest.fixture
def cordis_release(tmp_path, monkeypatch):
    taxonomy = tmp_path / "taxonomy"
    taxonomy.mkdir()
    labels = [
        {
            "id": "natural_sciences",
            "name": "Natural sciences",
            "parent": None,
            "euroscivoc_code": "/23",
            "is_direct_supported": False,
            "depth": 1,
        },
        {
            "id": "acoustics",
            "name": "Acoustics",
            "parent": "natural_sciences",
            "euroscivoc_code": "/23/43/273",
            "is_direct_supported": True,
            "depth": 2,
        },
        {
            "id": "optics",
            "name": "Optics",
            "parent": "natural_sciences",
            "euroscivoc_code": "/23/43/277",
            "is_direct_supported": True,
            "depth": 2,
        },
    ]
    (taxonomy / "taxonomy.json").write_text(json.dumps({"version": "0.4.0", "labels": labels}))
    (taxonomy / "label_map.json").write_text(
        json.dumps({"aliases": {label["euroscivoc_code"]: label["id"] for label in labels}})
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    with (raw / "project.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow([f"field_{i}" for i in range(22)])
        for i in range(40):
            row = [""] * 22
            row[0], row[3], row[15], row[20] = (
                str(i),
                f"Project {i}",
                f"<p>Scientific research number {i}. Further observations about the project.</p>",
                "NA",
            )
            writer.writerow(row)
    with (raw / "euroSciVoc.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["code", "path", "title", "description", "project"])
        for i in range(40):
            writer.writerow([labels[1 + i % 2]["euroscivoc_code"], "", "topic", "", str(i)])
    lock = {
        "url": "https://example.test/cordis.zip",
        "archive_sha256": "0" * 64,
        "project_sha256": compute_file_sha256(raw / "project.csv"),
        "euroscivoc_sha256": compute_file_sha256(raw / "euroSciVoc.csv"),
    }
    monkeypatch.setattr(preprocessing, "get_source_config", lambda name: lock)
    import naltra.data.validation as validation

    original = validation.get_source_config
    monkeypatch.setattr(
        validation,
        "get_source_config",
        lambda name: lock if name == "cordis_h2020" else original(name),
    )
    base = tmp_path / "processed"
    preprocessing.process_cordis_h2020(
        raw,
        base / "en",
        tmp_path / "splits",
        taxonomy / "taxonomy.json",
        taxonomy / "label_map.json",
    )
    return base, taxonomy / "taxonomy.json"


def republish(base, taxonomy, language="en"):
    directory = base / language
    old = json.loads((directory / "manifest.json").read_text())
    create_manifest(
        old["benchmark_name"],
        directory,
        old["generation_parameters"],
        old["source_metadata"],
        taxonomy_dir=taxonomy.parent,
        input_files=[directory / path for path in old["input_files"]],
    )


def test_cordis_release_checks_sources_and_hierarchy(cordis_release):
    base, taxonomy = cordis_release
    release = validate_cordis_release(base, ["en"], taxonomy_path=taxonomy)
    assert release["label_universe"] == ["acoustics", "optics"]
    assert sum(len(records) for records in release["corpora"]["en"].values()) == 40
    assert all(
        set(r["labels_direct"]) < set(r["labels"])
        for records in release["corpora"]["en"].values()
        for r in records
    )


@pytest.mark.parametrize(
    "mutation", ["duplicate", "split", "unknown_label", "closure", "fingerprint", "project"]
)
def test_cordis_rejects_corruption_with_fresh_manifest(cordis_release, mutation):
    base, taxonomy = cordis_release
    path = base / "en/validation.jsonl"
    rows = load_jsonl(path)
    if mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "split":
        rows[0]["split"] = "test"
    elif mutation == "unknown_label":
        rows[0]["labels_direct"] = ["unknown"]
    elif mutation == "closure":
        rows[0]["labels"] = rows[0]["labels_direct"]
    elif mutation == "fingerprint":
        rows[0]["text"] = "Changed text while retaining old fingerprint."
    else:
        rows[0]["project_id"] = "unexpected"
    save_jsonl(rows, path)
    republish(base, taxonomy)
    with pytest.raises(ValueError):
        validate_cordis_release(base, ["en"], taxonomy_path=taxonomy)


@pytest.mark.parametrize("key", ["id", "text", "pair_id", "project_id"])
def test_project_and_content_contamination_is_rejected(key):
    train = {"id": "train", "pair_id": "p1", "project_id": "1", "text": "One unique objective"}
    test = {"id": "test", "pair_id": "p2", "project_id": "2", "text": "Another unique objective"}
    test[key] = train[key]
    with pytest.raises(ValueError, match="contamination"):
        check_disjoint_partitions({"train": [train], "test": [test]})


def test_missing_source_inventory_is_rejected(cordis_release):
    base, taxonomy = cordis_release
    directory = base / "en"
    manifest = json.loads((directory / "manifest.json").read_text())
    source = next(path for path in manifest["input_files"] if path.endswith("project.csv"))
    del manifest["input_files"][source]
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source inventory"):
        validate_cordis_release(base, ["en"], taxonomy_path=taxonomy)


def test_changed_locked_raw_source_is_rejected(cordis_release):
    base, taxonomy = cordis_release
    raw = base.parent / "raw/project.csv"
    raw.write_text(raw.read_text() + "\n")
    with pytest.raises(ValueError):
        validate_cordis_release(base, ["en"], taxonomy_path=taxonomy)


def test_translation_requires_explicit_mock_and_complete_alignment(cordis_release):
    base, taxonomy = cordis_release
    kwargs = dict(
        en_processed_dir=base / "en",
        output_dir=base / "tr",
        cache_dir=base.parent / "cache",
        taxonomy_path=taxonomy,
        translator=MockTranslator(),
        review_sample_path=base.parent / "review.csv",
    )
    with pytest.raises(ValueError, match="Mock"):
        translate_cordis_dataset(**kwargs)
    translate_cordis_dataset(**kwargs, allow_mock=True)
    release = validate_cordis_release(base, ["en", "tr"], taxonomy_path=taxonomy, allow_mock=True)
    assert len(release["corpora"]["tr"]["train"]) == len(release["corpora"]["en"]["train"])
    with pytest.raises(ValueError, match="mock"):
        validate_cordis_release(base, ["en", "tr"], taxonomy_path=taxonomy)
    rows = load_jsonl(base / "tr/validation.jsonl")
    rows[0]["translation_source_hash"] = hashlib.sha256(b"wrong source").hexdigest()
    save_jsonl(rows, base / "tr/validation.jsonl")
    republish(base, taxonomy, "tr")
    with pytest.raises(ValueError, match="source hash"):
        validate_cordis_release(base, ["en", "tr"], taxonomy_path=taxonomy, allow_mock=True)


def test_cache_identity_changes_with_pipeline_revision(tmp_path):
    before = TranslationCache(tmp_path, namespace="revision-a")
    after = TranslationCache(tmp_path, namespace="revision-b")
    assert before.compute_key("1", "hash", "backend", "model") != after.compute_key(
        "1", "hash", "backend", "model"
    )


def test_missing_cordis_release_fails_without_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
    assert audit.main(["--languages", "en"]) == 1
    output = capsys.readouterr().out
    assert "AUDIT SUMMARY: FAILED" in output
    assert "Traceback" not in output


def test_installed_console_command_dispatches_cordis(monkeypatch):
    from naltra.cli import prepare_data

    calls = []
    monkeypatch.setattr(preprocessing, "process_cordis_h2020", lambda *args: calls.append("cordis"))
    prepare_data(["--dataset", "cordis_h2020"])
    assert calls == ["cordis"]


def test_audit_loader_rejects_empty_partition(cordis_release):
    base, taxonomy = cordis_release
    save_jsonl([], base / "en/test.jsonl")
    with pytest.raises(ValueError, match="Empty"):
        load_audit_records(base / "en/test.jsonl", taxonomy)


def test_long_sentence_chunking_preserves_all_words():
    from naltra.data.translation import chunk_sentence_if_needed

    class Tokenizer:
        def tokenize(self, text):
            return text.split()

    sentence = " ".join(f"word{i}" for i in range(25))
    chunks = chunk_sentence_if_needed(sentence, Tokenizer(), max_tokens=7)
    assert " ".join(chunks) == sentence
    assert all(len(chunk.split()) <= 7 for chunk in chunks)
