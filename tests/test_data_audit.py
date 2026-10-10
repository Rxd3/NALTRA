"""Regression checks for the CORDIS release and the retained integrity safeguards."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest
from scripts import validate_datasets as audit

from naltra.data import preprocessing
from naltra.data.code_switching import generate_code_switch_benchmarks
from naltra.data.loader import load_jsonl, save_jsonl
from naltra.data.manifest import compute_file_sha256, create_manifest
from naltra.data.noise import generate_noisy_benchmarks
from naltra.data.translation import MockTranslator, TranslationCache, translate_cordis_dataset
from naltra.data.validation import (
    SPLITS,
    check_disjoint_partitions,
    load_audit_records,
    validate_code_switch_release,
    validate_cordis_release,
    validate_prebuilt_cordis_release,
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


def split_hashes(base, language="en"):
    return {language: {s: compute_file_sha256(base / language / f"{s}.jsonl") for s in SPLITS}}


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


@pytest.fixture
def mixed_release(cordis_release):
    base, taxonomy = cordis_release
    translate_cordis_dataset(
        en_processed_dir=base / "en",
        output_dir=base / "tr",
        cache_dir=base.parent / "cache",
        taxonomy_path=taxonomy,
        translator=MockTranslator(),
        review_sample_path=base.parent / "review.csv",
        allow_mock=True,
    )
    release = validate_cordis_release(base, ["en", "tr"], taxonomy_path=taxonomy, allow_mock=True)
    for strategy in ("sentence_mix", "chunk_mix"):
        generate_code_switch_benchmarks(
            en_dir=base / "en",
            tr_dir=base / "tr",
            output_base_dir=base / "code_switch",
            strategy=strategy,
            taxonomy_path=taxonomy,
        )
    return base, taxonomy, release


def test_code_switch_audit_accepts_exact_sources_without_noise(mixed_release):
    base, taxonomy, release = mixed_release
    result = validate_code_switch_release(base, release, taxonomy_path=taxonomy)
    for strategy, partitions in result["corpora"].items():
        assert set(partitions) == {"validation", "test"}
        for split, records in partitions.items():
            assert len(records) == len(release["corpora"]["en"][split])
            assert all(r["code_switch_strategy"] == strategy for r in records)
    assert not (base / "code_switch/sentence_mix/balanced/train.jsonl").exists()


def test_code_switch_cli_audit_does_not_require_noise(mixed_release, monkeypatch):
    base, taxonomy, release = mixed_release
    original = validate_code_switch_release
    monkeypatch.setattr(audit, "validate_cordis_release", lambda *args, **kwargs: release)
    monkeypatch.setattr(
        audit,
        "validate_code_switch_release",
        lambda *args: original(base, release, taxonomy_path=taxonomy),
    )
    monkeypatch.setattr(audit, "validate_derived", lambda *args: pytest.fail("noise requested"))
    assert audit.main(["--code-switch"]) == 0


def test_derived_manifests_record_source_dirs_relative_to_themselves(mixed_release):
    base, _, _ = mixed_release

    def sources(directory):
        return json.loads((base / directory / "manifest.json").read_text())["source_metadata"]

    assert sources("tr") == {"source_en_dir": "../en", "cache_dir": "../../cache"}
    for strategy in ("sentence_mix", "chunk_mix"):
        assert sources(f"code_switch/{strategy}/balanced") == {
            "source_en_dir": "../../../en",
            "source_tr_dir": "../../../tr",
        }


CROSS_DRIVE = [
    ("english", "raw"),
    ("english", "label_map"),
    ("english", "splits"),
    ("english", "output"),
    ("turkish", "en"),
    ("turkish", "cache"),
    ("turkish", "output"),
    ("code_switch", "en"),
    ("code_switch", "tr"),
    ("code_switch", "output"),
    ("noise", "processed"),
    ("noise", "output"),
]


@pytest.mark.parametrize("writer, moved", CROSS_DRIVE, ids=[f"{w}-{m}" for w, m in CROSS_DRIVE])
def test_writers_refuse_any_source_on_another_drive_before_writing(
    cordis_release, monkeypatch, writer, moved
):
    """os.path.relpath raises when a source, cache or output sits on another drive or a UNC
    share. A manifest could not then record that path relative to itself, so each writer must
    stop before any split file exists, whichever single location is on the other drive.
    """
    base, taxonomy = cordis_release
    root, far = base.parent, base.parent / "far"
    if writer == "code_switch":
        mock_translated_release(base, taxonomy)

    def place(name, path, target=None):
        """The path itself, or a copy on the other drive when it is the moved location."""
        if name != moved:
            return path
        target = target or far / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)
        return target

    real_relpath = os.path.relpath

    def relpath(path, start=None):
        if ("far" in Path(path).parts) != ("far" in Path(start).parts):
            raise ValueError("path is on mount 'D:', start on mount 'C:'")
        return real_relpath(path, start)

    monkeypatch.setattr(os.path, "relpath", relpath)
    out = far / "out" if moved == "output" else root / "out"
    splits = far / "splits" if moved == "splits" else out / "splits"
    cache = far / "cache" if moved == "cache" else out / "cache"
    with pytest.raises(ValueError, match="one drive"):
        if writer == "english":
            label_map = place("label_map", taxonomy.parent / "label_map.json")
            preprocessing.process_cordis_h2020(
                place("raw", root / "raw"), out, splits, taxonomy, label_map
            )
        elif writer == "turkish":
            translate_cordis_dataset(
                en_processed_dir=place("en", base / "en"),
                output_dir=out,
                cache_dir=cache,
                taxonomy_path=taxonomy,
                translator=MockTranslator(),
                review_sample_path=out / "review.csv",
                allow_mock=True,
            )
        elif writer == "code_switch":
            generate_code_switch_benchmarks(
                en_dir=place("en", base / "en"),
                tr_dir=place("tr", base / "tr"),
                output_base_dir=out,
                taxonomy_path=taxonomy,
            )
        else:
            processed = place("processed", base, far / "base" / base.name)
            generate_noisy_benchmarks(processed.parent, out, datasets=(f"{base.name}/en",))
    for written in (out, splits, cache):
        assert not written.exists() or not any(written.rglob("*"))


@pytest.mark.parametrize("mutation", ["text", "pair", "duplicate", "params", "input"])
def test_code_switch_audit_rejects_corruption_with_fresh_hashes(mixed_release, mutation):
    base, taxonomy, release = mixed_release
    directory = base / "code_switch/sentence_mix/balanced"
    path = directory / "validation.jsonl"
    rows = load_jsonl(path)
    if mutation == "text":
        rows[0]["text"] += " A stale translation."
    elif mutation == "pair":
        rows[0]["pair_id"] = rows[1]["pair_id"]
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    save_jsonl(rows, path)
    republish(base, taxonomy, "code_switch/sentence_mix/balanced")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "params":
        manifest["generation_parameters"]["base_seed"] = 99
    elif mutation == "input":
        del manifest["input_files"][next(iter(manifest["input_files"]))]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        validate_code_switch_release(base, release, taxonomy_path=taxonomy)


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
    assert release["split_sha256"] == split_hashes(base)


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


@pytest.mark.parametrize("missing", [None, ""])
def test_unpaired_rows_without_group_ids_are_not_contamination(missing):
    train = {"id": "train", "pair_id": missing, "project_id": missing, "text": "One objective"}
    test = {"id": "test", "pair_id": missing, "project_id": missing, "text": "Another objective"}
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


@pytest.mark.parametrize(
    "mutation", ["grouped", "missing_source", "reordered_source", "empty_target", "changed_target"]
)
def test_imported_alignment_preserves_complete_source_and_output(cordis_release, mutation):
    base, taxonomy = cordis_release
    translate_cordis_dataset(
        en_processed_dir=base / "en",
        output_dir=base / "tr",
        cache_dir=base.parent / "cache",
        taxonomy_path=taxonomy,
        translator=MockTranslator(),
        review_sample_path=base.parent / "review.csv",
        allow_mock=True,
    )
    rows = load_jsonl(base / "tr/validation.jsonl")
    record = rows[0]
    source = record["english_source_text"]
    if mutation == "reordered_source":
        record["sentence_alignment"].reverse()
        for index, pair in enumerate(record["sentence_alignment"]):
            pair["index"] = index
        record["text"] = " ".join(p["tr"] for p in record["sentence_alignment"])
    else:
        record["sentence_alignment"] = [{"index": 0, "en": source, "tr": record["text"]}]
        if mutation == "missing_source":
            record["sentence_alignment"][0]["en"] = source.split(". ")[0]
        elif mutation == "empty_target":
            record["sentence_alignment"][0]["tr"] = ""
            record["text"] = ""
        elif mutation == "changed_target":
            record["text"] += " Changed output."
    save_jsonl(rows, base / "tr/validation.jsonl")
    republish(base, taxonomy, "tr")
    if mutation == "grouped":
        validate_cordis_release(base, ["en", "tr"], taxonomy_path=taxonomy, allow_mock=True)
    else:
        with pytest.raises(ValueError):
            validate_cordis_release(base, ["en", "tr"], taxonomy_path=taxonomy, allow_mock=True)


def test_missing_cordis_release_fails_without_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
    assert audit.main(["--languages", "en"]) == 1
    output = capsys.readouterr().out
    assert "AUDIT SUMMARY: FAILED" in output
    assert "Traceback" not in output


def test_missing_raw_sources_name_the_download_and_prebuilt_remedies(
    cordis_release, monkeypatch, capsys
):
    base, taxonomy = cordis_release
    (base.parent / "raw/euroSciVoc.csv").unlink()
    original = validate_cordis_release
    monkeypatch.setattr(
        audit,
        "validate_cordis_release",
        lambda _, languages, **kwargs: original(base, languages, taxonomy_path=taxonomy),
    )
    assert audit.main(["--languages", "en"]) == 1
    output = capsys.readouterr().out
    assert "python scripts/prepare_data.py --dataset cordis_h2020 --download" in output
    assert "--prebuilt-release" in output
    assert "Errno" not in output
    assert "AUDIT SUMMARY: FAILED" in output


def test_prebuilt_release_cli_audits_records_without_raw_sources(
    cordis_release, monkeypatch, capsys
):
    base, taxonomy = cordis_release
    mock_translated_release(base, taxonomy)
    (base.parent / "raw/euroSciVoc.csv").unlink()
    original = validate_prebuilt_cordis_release
    monkeypatch.setattr(
        audit,
        "validate_prebuilt_cordis_release",
        lambda _, languages, **kwargs: original(
            base, languages, taxonomy_path=taxonomy, allow_mock=True
        ),
    )
    monkeypatch.setattr(
        audit, "validate_cordis_release", lambda *args, **kwargs: pytest.fail("full audit")
    )
    issues = original(base, ["en", "tr"], taxonomy_path=taxonomy, allow_mock=True)[
        "translation_issues"
    ]
    assert audit.main(["--prebuilt-release"]) == 0
    output = capsys.readouterr().out
    assert "[PASS] CORDIS en" in output and "[PASS] CORDIS tr" in output
    assert (
        f"segmenter drift {issues['segmenter_drift']}, "
        f"alignment/text mismatches {len(issues['alignment_text_mismatch'])}, "
        f"QA failures {len(issues['qa_failed'])}"
    ) in output
    assert "reported, not enforced:" in output
    summary = output.strip().splitlines()[-1]
    assert summary.startswith("AUDIT SUMMARY: PASSED") and "reported, not enforced" in summary

    rows = load_jsonl(base / "tr/test.jsonl")
    rows[0]["labels_direct"] = []
    save_jsonl(rows, base / "tr/test.jsonl")
    assert audit.main(["--prebuilt-release"]) == 1
    assert "AUDIT SUMMARY: FAILED" in capsys.readouterr().out


@pytest.mark.parametrize("derived", ["--code-switch", "--derived"])
def test_prebuilt_release_refuses_derived_audits(derived, capsys):
    with pytest.raises(SystemExit) as stopped:
        audit.main(["--prebuilt-release", derived])
    assert stopped.value.code == 2
    assert "EN and TR records only" in capsys.readouterr().err


def test_changed_generation_code_names_the_prebuilt_remedy(cordis_release, monkeypatch, capsys):
    base, taxonomy = cordis_release
    from naltra.data import manifest

    monkeypatch.setattr(manifest, "get_generation_code_hashes", lambda: {"edited.py": "0" * 64})
    original = validate_cordis_release
    monkeypatch.setattr(
        audit,
        "validate_cordis_release",
        lambda _, languages, **kwargs: original(base, languages, taxonomy_path=taxonomy),
    )
    assert audit.main(["--languages", "en"]) == 1
    output = capsys.readouterr().out
    assert "generation code" in output and "--prebuilt-release" in output
    assert "AUDIT SUMMARY: FAILED" in output


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


def downgrade_manifest(base, language="en"):
    """Mimic the distributed 1.0.0 release bundle that lacks raw sources and a split map."""
    path = base / language / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["manifest_version"] = "1.0.0"
    path.write_text(json.dumps(manifest))


def test_prebuilt_release_audits_records_without_raw_sources(cordis_release):
    base, taxonomy = cordis_release
    downgrade_manifest(base)
    (base.parent / "raw/project.csv").unlink()
    with pytest.raises(ValueError):
        validate_cordis_release(base, ["en"], taxonomy_path=taxonomy)
    release = validate_prebuilt_cordis_release(base, ["en"], taxonomy_path=taxonomy)
    assert release["audit"] == "prebuilt_records_only"
    assert release["label_universe"] == ["acoustics", "optics"]
    assert sum(len(records) for records in release["corpora"]["en"].values()) == 40
    assert release["manifests"]["en"]["manifest_version"] == "1.0.0"
    assert release["split_sha256"] == split_hashes(base)


@pytest.mark.parametrize("mutation", ["text", "record_count"])
def test_prebuilt_release_refuses_splits_that_differ_from_their_manifest(cordis_release, mutation):
    base, taxonomy = cordis_release
    downgrade_manifest(base)
    if mutation == "text":
        path = base / "en/train.jsonl"
        rows = load_jsonl(path)
        rows[0]["text"] = "A consistently rewritten objective that the manifest never hashed."
        rows[0]["content_fingerprint"] = preprocessing.compute_content_fingerprint(rows[0]["text"])
        save_jsonl(rows, path)
    else:
        path = base / "en/manifest.json"
        manifest = json.loads(path.read_text())
        manifest["output_files"]["train.jsonl"]["record_count"] += 1
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        validate_prebuilt_cordis_release(base, ["en"], taxonomy_path=taxonomy)


@pytest.mark.parametrize("language", ["en", "tr"])
@pytest.mark.parametrize("legacy", ["no_manifest", "no_output_hashes", "no_test_hash"])
def test_prebuilt_release_requires_each_manifest_and_its_split_hashes(
    cordis_release, legacy, language
):
    base, taxonomy = cordis_release
    mock_translated_release(base, taxonomy)
    path = base / language / "manifest.json"
    if legacy == "no_manifest":
        path.unlink()
    else:
        manifest = json.loads(path.read_text())
        if legacy == "no_output_hashes":
            manifest["output_files"] = {name: {} for name in manifest["output_files"]}
        else:
            del manifest["output_files"]["test.jsonl"]["sha256"]
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=f"CORDIS {language} .*manifest"):
        validate_prebuilt_cordis_release(
            base, ["en", "tr"], taxonomy_path=taxonomy, allow_mock=True
        )


@pytest.mark.parametrize("mutation", ["duplicate", "split", "closure", "fingerprint", "identity"])
def test_prebuilt_release_still_rejects_corrupt_records(cordis_release, mutation):
    base, taxonomy = cordis_release
    path = base / "en/validation.jsonl"
    rows = load_jsonl(path)
    if mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "split":
        rows[0]["split"] = "test"
    elif mutation == "closure":
        rows[0]["labels"] = rows[0]["labels_direct"]
    elif mutation == "fingerprint":
        rows[0]["text"] = "Changed text while retaining old fingerprint."
    else:
        rows[0]["id"] = "cordis:other:en"
    save_jsonl(rows, path)
    republish(base, taxonomy)
    with pytest.raises(ValueError):
        validate_prebuilt_cordis_release(base, ["en"], taxonomy_path=taxonomy)


def test_prebuilt_release_rejects_cross_split_project_leakage(cordis_release):
    base, taxonomy = cordis_release
    train, validation = load_jsonl(base / "en/train.jsonl"), load_jsonl(
        base / "en/validation.jsonl"
    )
    leaked = {**copy.deepcopy(train[0]), "split": "validation", "id": "cordis:leak:en"}
    save_jsonl([*validation, leaked], base / "en/validation.jsonl")
    republish(base, taxonomy)
    with pytest.raises(ValueError):
        validate_prebuilt_cordis_release(base, ["en"], taxonomy_path=taxonomy)


def test_prebuilt_translation_reports_observed_provenance(cordis_release):
    base, taxonomy = cordis_release
    translate_cordis_dataset(
        en_processed_dir=base / "en",
        output_dir=base / "tr",
        cache_dir=base.parent / "cache",
        taxonomy_path=taxonomy,
        translator=MockTranslator(),
        review_sample_path=base.parent / "review.csv",
        allow_mock=True,
    )
    with pytest.raises(ValueError, match="mock"):
        validate_prebuilt_cordis_release(base, ["en", "tr"], taxonomy_path=taxonomy)
    release = validate_prebuilt_cordis_release(
        base, ["en", "tr"], taxonomy_path=taxonomy, allow_mock=True
    )
    assert len(release["corpora"]["tr"]["test"]) == len(release["corpora"]["en"]["test"])
    observed = release["translation_provenance"]
    assert set(observed) == {
        "translation_backend",
        "translation_model",
        "translation_model_revision",
    }
    rows = load_jsonl(base / "tr/train.jsonl")
    rows[0]["translation_model"] = "another/model"
    save_jsonl(rows, base / "tr/train.jsonl")
    with pytest.raises(ValueError, match="provenance"):
        validate_prebuilt_cordis_release(
            base, ["en", "tr"], taxonomy_path=taxonomy, allow_mock=True
        )


def mock_translated_release(base, taxonomy):
    translate_cordis_dataset(
        en_processed_dir=base / "en",
        output_dir=base / "tr",
        cache_dir=base.parent / "cache",
        taxonomy_path=taxonomy,
        translator=MockTranslator(),
        review_sample_path=base.parent / "review.csv",
        allow_mock=True,
    )


def test_prebuilt_tolerates_segmenter_drift_but_reports_it(cordis_release):
    base, taxonomy = cordis_release
    mock_translated_release(base, taxonomy)
    rows = load_jsonl(base / "tr/train.jsonl")
    first = rows[0]["sentence_alignment"]
    merged = {
        "index": 0,
        "en": " ".join(p["en"] for p in first),
        "tr": " ".join(p["tr"] for p in first),
    }
    rows[0]["sentence_alignment"] = [merged]
    rows[1]["text"] = rows[1]["text"] + " ek"
    save_jsonl(rows, base / "tr/train.jsonl")
    republish(base, taxonomy, "tr")
    with pytest.raises(ValueError, match="alignment"):
        validate_cordis_release(base, ["en", "tr"], taxonomy_path=taxonomy, allow_mock=True)
    release = validate_prebuilt_cordis_release(
        base, ["en", "tr"], taxonomy_path=taxonomy, allow_mock=True
    )
    issues = release["translation_issues"]
    assert issues["segmenter_drift"] == 1
    assert issues["alignment_text_mismatch"] == [rows[1]["id"]]


def test_prebuilt_rejects_alignment_that_drops_source_text(cordis_release):
    base, taxonomy = cordis_release
    mock_translated_release(base, taxonomy)
    rows = load_jsonl(base / "tr/train.jsonl")
    rows[0]["sentence_alignment"][0]["en"] = "Something the source never said."
    save_jsonl(rows, base / "tr/train.jsonl")
    with pytest.raises(ValueError, match="alignment"):
        validate_prebuilt_cordis_release(
            base, ["en", "tr"], taxonomy_path=taxonomy, allow_mock=True
        )


def test_derived_audit_reads_the_noisy_layout_prepare_data_writes(
    cordis_release, tmp_path, monkeypatch, capsys
):
    base, taxonomy = cordis_release
    mock_translated_release(base, taxonomy)
    release = validate_cordis_release(base, ["en", "tr"], taxonomy_path=taxonomy, allow_mock=True)
    processed = tmp_path / "data/processed/cordis_h2020"
    shutil.copytree(base, processed)
    monkeypatch.chdir(tmp_path)  # noise manifests hash ./taxonomy, as prepare_data runs from root
    generate_noisy_benchmarks(tmp_path / "data/processed", tmp_path / "data/noisy")
    for strategy in ("sentence_mix", "chunk_mix"):
        generate_code_switch_benchmarks(
            en_dir=processed / "en",
            tr_dir=processed / "tr",
            output_base_dir=processed / "code_switch",
            strategy=strategy,
            taxonomy_path=taxonomy,
        )
    monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(audit, "validate_cordis_release", lambda *args, **kwargs: release)
    assert audit.main(["--derived"]) == 0
    assert "[PASS] CORDIS noise" in capsys.readouterr().out

    legacy = tmp_path / "data/noisy/cordis_h2020/combined"
    legacy.parent.mkdir()
    (tmp_path / "data/noisy/combined").rename(legacy)
    assert audit.main(["--derived"]) == 1
    output = capsys.readouterr().out
    assert "[FAIL]" in output and "manifest.json" in output


@pytest.mark.parametrize("strict", [True, False])
def test_alignment_matching_the_segmenter_is_accepted_when_it_normalises_list_marks(strict):
    """Repaired 1.1.0 records follow segment_sentences, which rewrites bullet/list marks."""
    from naltra.data.translation import segment_sentences
    from naltra.data.validation import _check_translated_record

    text = "Partners are SMEs and petrol companies • High innovation and - Piloting of tools."
    parts = segment_sentences(text)
    assert "".join(parts).replace(" ", "") != text.replace(" ", "")
    shared = {"labels": ["optics"], "labels_direct": ["optics"], "pair_id": "cordis:p1"}
    shared |= {"split": "train", "source": "cordis_h2020", "license": "CC BY 4.0"}
    source = {"id": "cordis:p1:en", "project_id": "p1", "language": "en", "text": text, **shared}
    alignment = [{"index": i, "en": p, "tr": f"tr {i}"} for i, p in enumerate(parts)]
    translated = {
        "id": "cordis:p1:tr",
        "project_id": "p1",
        "language": "tr",
        "variant_of": "cordis:p1:en",
        "translation_source_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "sentence_alignment": alignment,
        "text": " ".join(p["tr"] for p in alignment),
        **shared,
    }
    assert _check_translated_record(translated, source, strict=strict) == set()
