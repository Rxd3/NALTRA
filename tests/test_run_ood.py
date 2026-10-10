"""Held-out-root near-OOD protocol: train without a root, test on its exclusive projects."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scripts import run_ood
from tests.test_provenance import absolute_paths

from naltra.data.loader import save_jsonl
from naltra.data.manifest import compute_file_sha256, get_environment_metadata
from naltra.data.preprocessing import hierarchy_closure
from naltra.models.naive_bayes import NaiveBayesModel
from naltra.utils.provenance import commit

PARENTS = {
    "natural_sciences": None,
    "physical_sciences": "natural_sciences",
    "acoustics": "physical_sciences",
    "optics": "physical_sciences",
    "humanities": None,
    "arts": "humanities",
}
SPLITS = ("train", "validation", "test")
TAXONOMY = run_ood.REPO_ROOT / "taxonomy/taxonomy.json"
CONTAMINATIONS = {
    "record id": ("en", {"id": "cordis:train0:en"}),
    "pair id": ("en", {"pair_id": "cordis:train0"}),
    "project id": ("en", {"project_id": "train0"}),
    "identical text": ("en", {"text": "sound waves acoustic noise vibration train 0"}),
    "declared train split": ("en", {"split": "train"}),
    "translated training project": ("tr", {"pair_id": "cordis:train0", "project_id": "train0"}),
}


def test_partition_separates_clean_heldout_and_mixed_projects() -> None:
    records = [
        {"labels_direct": ["acoustics"]},
        {"labels_direct": ["arts"]},
        {"labels_direct": ["optics", "arts"]},
    ]
    clean, heldout, mixed = run_ood.partition(records, "humanities", PARENTS)
    assert [r["labels_direct"] for r in clean] == [["acoustics"]]
    assert [r["labels_direct"] for r in heldout] == [["arts"]]
    assert [r["labels_direct"] for r in mixed] == [["optics", "arts"]]


def rows(split: str, lang: str = "en") -> list[dict]:
    texts = {
        "acoustics": "sound waves acoustic noise vibration",
        "optics": "laser light lens photon beam",
        "arts": "painting sculpture museum artist gallery",
    }
    result = []
    for index in range(12):
        label = ["acoustics", "optics", "arts"][index % 3]
        english = f"{texts[label]} {split} {index}"
        record = {
            "id": f"cordis:{split}{index}:{lang}",
            "project_id": f"{split}{index}",
            "pair_id": f"cordis:{split}{index}",
            "text": f"ceviri {english}" if lang == "tr" else english,
            "labels_direct": [label],
            "labels": [label],
            "language": lang,
            "source": "fixture",
            "license": "MIT",
            "split": split,
        }
        if lang == "tr":
            record |= {
                "variant_of": f"cordis:{split}{index}:en",
                "translation_source_hash": hashlib.sha256(english.encode("utf-8")).hexdigest(),
                "sentence_alignment": [{"index": 0, "en": english, "tr": record["text"]}],
            }
        result.append(record)
    return result


def write_release(base: Path, languages: tuple[str, ...] = ("en",)) -> dict[tuple, list[dict]]:
    release = {(lang, split): rows(split, lang) for lang in languages for split in SPLITS}
    for (lang, split), records in release.items():
        save_jsonl(records, base / "data" / lang / f"{split}.jsonl")
    return release


def run(base: Path, languages: tuple[str, ...] = ("en",), *extra: str) -> int:
    return run_ood.main(
        [
            "--data-dir",
            str(base / "data"),
            "--languages",
            *languages,
            "--models",
            "naive_bayes",
            "--heldout-root",
            "humanities",
            "--output-dir",
            str(base / "ood"),
            *extra,
        ]
    )


def read_summary(base: Path) -> dict[str, Any]:
    return json.loads((base / "ood/summary.json").read_text(encoding="utf-8"))


def test_heldout_projects_score_as_more_out_of_distribution(tmp_path) -> None:
    write_release(tmp_path)
    assert run(tmp_path) == 0
    summary = read_summary(tmp_path)
    assert summary["heldout_root"] == "humanities"
    assert summary["counts"]["train"] == 8
    result = summary["results"]["naive_bayes"]["en"]
    assert result["id_records"] == 8 and result["ood_records"] == 4
    assert result["auroc"] > 0.5
    assert "arts" not in summary["results"]["naive_bayes"]["trained_labels"]


@pytest.mark.parametrize("lang, change", CONTAMINATIONS.values(), ids=list(CONTAMINATIONS))
def test_contaminated_or_misassigned_test_inputs_are_refused_before_fitting(
    tmp_path, monkeypatch, capsys, lang: str, change: dict[str, str]
) -> None:
    test = write_release(tmp_path, ("en", "tr"))[lang, "test"]
    save_jsonl([{**test[0], **change}, *test[1:]], tmp_path / "data" / lang / "test.jsonl")

    def fit(*_: Any) -> None:
        raise AssertionError("A model was fitted on an unaudited release.")

    monkeypatch.setattr(NaiveBayesModel, "train", fit)
    assert run(tmp_path, ("en", "tr")) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["status"] == "failed"
    assert not (tmp_path / "ood").exists()


def custom_taxonomy(base: Path) -> Path:
    """Moves arts under natural sciences and optics under humanities."""
    moved = {"arts": "natural_sciences", "optics": "humanities"}
    taxonomy = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    taxonomy["labels"] = [
        {**item, "parent": moved.get(item["id"], item["parent"])} for item in taxonomy["labels"]
    ]
    path = base / "taxonomy/taxonomy.json"
    path.parent.mkdir()
    path.write_text(json.dumps(taxonomy), encoding="utf-8")
    shutil.copy(TAXONOMY.parent / "label_map.json", path.parent)
    return path


def test_selected_taxonomy_drives_both_the_heldout_split_and_the_model(
    tmp_path, monkeypatch
) -> None:
    write_release(tmp_path)
    taxonomy = custom_taxonomy(tmp_path)
    built: list[NaiveBayesModel] = []

    class Recorded(NaiveBayesModel):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            built.append(self)

    monkeypatch.setitem(run_ood.MODEL_TYPES, "naive_bayes", Recorded)
    assert run(tmp_path, ("en",), "--taxonomy", str(taxonomy)) == 0
    trained = read_summary(tmp_path)["results"]["naive_bayes"]["trained_labels"]
    assert "arts" in trained and "optics" not in trained
    assert built[0].parents["optics"] == "humanities"
    assert built[0].taxonomy_identity["taxonomy_sha256"] == compute_file_sha256(taxonomy)


def test_taxonomy_must_be_a_taxonomy_json_beside_its_label_map(tmp_path) -> None:
    write_release(tmp_path)
    renamed = tmp_path / "renamed.json"
    shutil.copy(TAXONOMY, renamed)
    with pytest.raises(SystemExit):
        run(tmp_path, ("en",), "--taxonomy", str(renamed))
    assert not (tmp_path / "ood").exists()


def test_summary_identifies_taxonomy_models_code_and_auroc_uncertainty(tmp_path) -> None:
    write_release(tmp_path, ("en", "tr"))
    assert run(tmp_path, ("en", "tr")) == 0
    summary = read_summary(tmp_path)
    assert summary["taxonomy_sha256"] == compute_file_sha256(TAXONOMY)
    saved = tmp_path / "ood/models/naive_bayes"
    payload = json.loads((saved / "naltra.json").read_text(encoding="utf-8"))
    model = summary["models"]["naive_bayes"]
    assert model["naltra_json_sha256"] == compute_file_sha256(saved / "naltra.json")
    assert model["estimator_sha256"] == compute_file_sha256(saved / "estimator.joblib")
    assert model["config"] == payload["config"] and model["metadata"] == payload["metadata"]
    assert "environment" in model["metadata"]
    assert model["taxonomy"]["taxonomy_sha256"] == summary["taxonomy_sha256"]
    assert isinstance(model["commit"], str) and model["commit"]
    assert absolute_paths(summary, tmp_path) == []
    results = summary["results"]["naive_bayes"]
    assert set(results) == {"trained_labels", "en", "tr"}
    for lang in ("en", "tr"):
        assert 0 <= results[lang]["auroc_ci_low"] <= results[lang]["auroc"]
        assert results[lang]["auroc"] <= results[lang]["auroc_ci_high"] <= 1


def test_summary_records_the_code_commit_environment_and_bootstrap_seeds(
    tmp_path, monkeypatch
) -> None:
    write_release(tmp_path)
    seeds, resample = [], run_ood.resample_counts

    def recorded(rows: int, resamples: int, seed: int) -> np.ndarray:
        seeds.append(seed)
        return resample(rows, resamples, seed)

    monkeypatch.setattr(run_ood, "resample_counts", recorded)
    assert run(tmp_path) == 0
    summary = read_summary(tmp_path)
    assert summary["code_commit"] == commit() == summary["models"]["naive_bayes"]["commit"]
    assert summary["environment"] == get_environment_metadata()
    assert summary["bootstrap_seeds"] == {"id": seeds[0], "ood": seeds[1]}


def test_auroc_interval_brackets_chance_for_indistinguishable_scores() -> None:
    scores = np.linspace(0.0, 1.0, 40)
    low, high = run_ood.auroc_interval(scores, scores)
    assert low < 0.5 < high


def test_auroc_interval_collapses_at_one_for_perfectly_separated_scores() -> None:
    id_scores, ood_scores = np.linspace(0.0, 0.4, 30), np.linspace(0.6, 1.0, 10)
    assert run_ood.auroc_interval(id_scores, ood_scores) == (1.0, 1.0)


def test_languages_must_cover_the_same_projects_per_split(tmp_path, monkeypatch, capsys) -> None:
    test = write_release(tmp_path, ("en", "tr"))["tr", "test"]
    extra = {
        **test[0],
        "id": "cordis:extra0:tr",
        "project_id": "extra0",
        "pair_id": "cordis:extra0",
    }
    save_jsonl(
        [{**extra, "text": "ceviri unrelated extra project"}, *test[1:]],
        tmp_path / "data/tr/test.jsonl",
    )
    monkeypatch.setattr(NaiveBayesModel, "train", lambda *_: pytest.fail("fitted"))
    assert run(tmp_path, ("en", "tr")) == 1
    assert "project" in json.loads(capsys.readouterr().out.splitlines()[-1])["error"]
    assert not (tmp_path / "ood").exists()


def test_missing_heldout_projects_fail_cleanly_without_output(tmp_path, capsys) -> None:
    release = write_release(tmp_path)
    kept = [r for r in release["en", "test"] if r["labels_direct"] != ["arts"]]
    save_jsonl(kept, tmp_path / "data/en/test.jsonl")
    assert run(tmp_path) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["status"] == "failed"
    assert not (tmp_path / "ood").exists()


def test_translated_siblings_with_different_labels_are_refused_before_fitting(
    tmp_path, monkeypatch, capsys
) -> None:
    """English train0 moves to arts with its closure; its Turkish sibling stays acoustics."""
    taxonomy = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    parents = {item["id"]: item["parent"] for item in taxonomy["labels"]}
    for (lang, split), records in write_release(tmp_path, ("en", "tr")).items():
        cordis = []
        for r in records:
            direct = ["arts"] if r["id"] == "cordis:train0:en" else r["labels_direct"]
            cordis.append(
                {
                    **r,
                    "source": "cordis_h2020",
                    "taxonomy_version": taxonomy["version"],
                    "labels_direct": direct,
                    "labels": sorted(hierarchy_closure(direct, parents)),
                }
            )
        save_jsonl(cordis, tmp_path / "data" / lang / f"{split}.jsonl")
    monkeypatch.setattr(NaiveBayesModel, "train", lambda *_: pytest.fail("fitted"))
    assert run(tmp_path, ("en", "tr")) == 1
    error = json.loads(capsys.readouterr().out.splitlines()[-1])["error"]
    assert "train0" in error and "labels" in error
    assert not (tmp_path / "ood").exists()


def test_every_language_variant_of_a_heldout_project_leaves_training() -> None:
    english = {"project_id": "p0", "labels_direct": ["arts"]}
    turkish = {"project_id": "p0", "labels_direct": ["acoustics"]}
    kept = {"project_id": "p1", "labels_direct": ["optics"]}
    parts = [[english, kept], [turkish]]
    assert run_ood.drop_heldout_projects(parts, "humanities", PARENTS) == [kept]


@pytest.mark.parametrize("languages", [("en",), ("en", "tr")])
def test_a_project_with_two_rows_in_one_language_is_refused_before_fitting(
    tmp_path, monkeypatch, capsys, languages
) -> None:
    """A second English test0 row carrying arts would score test0 as both ID and OOD."""
    taxonomy = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    parents = {item["id"]: item["parent"] for item in taxonomy["labels"]}
    for (lang, split), records in write_release(tmp_path, languages).items():
        cordis = [
            {
                **r,
                "source": "cordis_h2020",
                "taxonomy_version": taxonomy["version"],
                "labels": sorted(hierarchy_closure(r["labels_direct"], parents)),
            }
            for r in records
        ]
        if (lang, split) == ("en", "test"):
            first = cordis[0]
            duplicate = {
                **first,
                "id": "cordis:test0dup:en",
                "text": first["text"] + " Second copy.",
                "labels_direct": ["arts"],
                "labels": sorted(hierarchy_closure(["arts"], parents)),
            }
            cordis = [duplicate, *cordis]
        save_jsonl(cordis, tmp_path / "data" / lang / f"{split}.jsonl")
    monkeypatch.setattr(NaiveBayesModel, "train", lambda *_: pytest.fail("fitted"))
    assert run(tmp_path, languages) == 1
    error = json.loads(capsys.readouterr().out.splitlines()[-1])["error"]
    assert "test0" in error and "en" in error
    assert not (tmp_path / "ood").exists()


def test_ood_scores_normalise_text_like_training_and_prediction() -> None:
    seen: list[str] = []

    class Recorder:
        def _probabilities(self, texts: list[str]) -> np.ndarray:
            seen.extend(texts)
            return np.full((len(texts), 2), 0.25)

    scores = run_ood.ood_scores(Recorder(), [{"text": "  sound\n\twaves  "}])
    assert seen == ["sound waves"] and scores.tolist() == [0.75]


def contents(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_a_failed_run_leaves_no_partial_output_and_keeps_the_previous_one(
    tmp_path, monkeypatch
) -> None:
    write_release(tmp_path)
    save = NaiveBayesModel.save

    def save_then_fail(self: NaiveBayesModel, path: Path) -> None:
        save(self, path)
        raise OSError("disk full")

    monkeypatch.setattr(NaiveBayesModel, "save", save_then_fail)
    assert run(tmp_path) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["data"]
    monkeypatch.undo()
    assert run(tmp_path) == 0
    previous = contents(tmp_path / "ood")
    monkeypatch.setattr(NaiveBayesModel, "save", save_then_fail)
    assert run(tmp_path) == 1
    assert contents(tmp_path / "ood") == previous
    assert sorted(p.name for p in tmp_path.iterdir()) == ["data", "ood"]
    monkeypatch.undo()
    assert run(tmp_path) == 0
    summary, saved = read_summary(tmp_path), tmp_path / "ood/models/naive_bayes"
    assert summary["models"]["naive_bayes"]["naltra_json_sha256"] == compute_file_sha256(
        saved / "naltra.json"
    )


def test_a_failed_swap_keeps_the_complete_previous_output(tmp_path, monkeypatch) -> None:
    write_release(tmp_path)
    both = ("--models", "naive_bayes", "svm")
    assert run(tmp_path, ("en",), *both) == 0
    previous, rename, calls = contents(tmp_path / "ood"), Path.rename, []

    def locked_on_second_call(self: Path, target: Any) -> Path:
        calls.append(self)
        if len(calls) == 2:
            raise PermissionError("locked")
        return rename(self, target)

    monkeypatch.setattr(Path, "rename", locked_on_second_call)
    assert run(tmp_path, ("en",), *both) == 1
    assert contents(tmp_path / "ood") == previous
    assert sorted(p.name for p in tmp_path.iterdir()) == ["data", "ood"]


def test_a_failed_swap_and_restore_leave_the_previous_output_recoverable(
    tmp_path, monkeypatch, capsys
) -> None:
    write_release(tmp_path)
    assert run(tmp_path) == 0
    previous, rename = contents(tmp_path / "ood"), Path.rename

    def concurrent_destination_after_backup(self: Path, target: Any) -> Path:
        moved = rename(self, target)
        if self.name == "ood":  # another writer recreates the output: both renames into it fail
            (tmp_path / "ood").mkdir()
            (tmp_path / "ood/concurrent.txt").write_text("other run", encoding="utf-8")
        return moved

    monkeypatch.setattr(Path, "rename", concurrent_destination_after_backup)
    assert run(tmp_path) == 1
    failure = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert contents(tmp_path / "ood") == {"concurrent.txt": b"other run"}
    assert contents(Path(failure["previous_output"])) == previous


def test_a_rerun_replaces_the_whole_previous_output(tmp_path) -> None:
    write_release(tmp_path)
    assert run(tmp_path, ("en",), "--models", "naive_bayes", "svm") == 0
    assert run(tmp_path, ("en",), "--models", "svm") == 0
    assert sorted(p.name for p in (tmp_path / "ood/models").iterdir()) == ["svm"]
    assert list(read_summary(tmp_path)["models"]) == ["svm"]


@pytest.fixture
def unfitted(monkeypatch) -> None:
    monkeypatch.setattr(NaiveBayesModel, "train", lambda *_: pytest.fail("fitted"))


def refused(capsys, tmp_path: Path, output: Path, reason: str, *extra: str) -> None:
    """The run stops with a usage error before fitting and every file stays as it was."""
    before = contents(tmp_path)
    with pytest.raises(SystemExit) as stopped:
        run(tmp_path, ("en",), "--output-dir", str(output), *extra)
    assert stopped.value.code == 2 and reason in capsys.readouterr().err
    assert contents(tmp_path) == before


def test_an_existing_output_that_is_not_a_previous_run_is_refused(
    tmp_path, capsys, unfitted
) -> None:
    write_release(tmp_path)
    (tmp_path / "results/predictions").mkdir(parents=True)
    (tmp_path / "results/predictions/svm_en_test.npz").write_bytes(b"scores")
    refused(capsys, tmp_path, tmp_path / "results", "pass --overwrite")


@pytest.mark.parametrize("output", ["data", "data/en", "data/ood", "."])
def test_an_output_overlapping_the_data_dir_is_refused_even_with_overwrite(
    tmp_path, capsys, unfitted, output: str
) -> None:
    write_release(tmp_path)
    refused(capsys, tmp_path, tmp_path / output, "overlaps the input", "--overwrite")


@pytest.mark.parametrize("output", ["checkout/repo", "checkout"])
def test_the_repository_or_a_directory_holding_it_is_refused_even_with_overwrite(
    tmp_path, monkeypatch, capsys, unfitted, output: str
) -> None:
    write_release(tmp_path)
    (tmp_path / "checkout/repo/src").mkdir(parents=True)
    (tmp_path / "checkout/repo/src/code.py").write_text("kept", encoding="utf-8")
    monkeypatch.setattr(run_ood, "REPO_ROOT", tmp_path / "checkout/repo")
    taxonomy = ("--taxonomy", str(TAXONOMY))
    refused(capsys, tmp_path, tmp_path / output, "repository", "--overwrite", *taxonomy)


@pytest.mark.parametrize("output", [".git", "src", "data/raw", "docs/results", "results", "notes"])
def test_overwrite_inside_the_repository_is_refused_outside_results(
    tmp_path, monkeypatch, capsys, unfitted, output: str
) -> None:
    write_release(tmp_path)
    repo = tmp_path / "checkout/repo"
    (repo / output).mkdir(parents=True)
    (repo / output / "kept.txt").write_text("kept", encoding="utf-8")
    monkeypatch.setattr(run_ood, "REPO_ROOT", repo)
    taxonomy = ("--taxonomy", str(TAXONOMY))
    refused(capsys, tmp_path, repo / output, "under results/", "--overwrite", *taxonomy)


def test_overwrite_under_results_inside_the_repository_is_allowed(tmp_path, monkeypatch) -> None:
    write_release(tmp_path)
    output = tmp_path / "checkout/repo/results/ood"
    output.mkdir(parents=True)
    (output / "notes.txt").write_text("unrelated", encoding="utf-8")
    monkeypatch.setattr(run_ood, "REPO_ROOT", tmp_path / "checkout/repo")
    extra = ("--output-dir", str(output), "--overwrite", "--taxonomy", str(TAXONOMY))
    assert run(tmp_path, ("en",), *extra) == 0
    assert not (output / "notes.txt").exists() and (output / "summary.json").is_file()


def marker(**change: Any) -> dict[str, Any]:
    """A summary.json shaped like the ones run_ood writes, with ``change`` applied."""
    return {
        "heldout_root": "humanities",
        "protocol": "held out",
        "taxonomy_sha256": "0" * 64,
        "inputs": {"en/test": "0" * 64},
        "counts": {"train": 8},
        "results": {"svm": {"trained_labels": ["acoustics"], "en": {"auroc": 0.9}}},
        "models": {"svm": {"commit": None}},
    } | change


SPOOFED = {
    "top-level keys only": dict.fromkeys(("heldout_root", "protocol", "results", "models")),
    "protocol not text": marker(protocol={"held": "out"}),
    "inputs not an object": marker(inputs=["en/test"]),
    "result without a language": marker(results={"svm": {"trained_labels": []}}),
    "language result not an object": marker(results={"svm": {"trained_labels": [], "en": 0.9}}),
    "models of other families": marker(models={"naive_bayes": {}}),
}


@pytest.mark.parametrize("summary", SPOOFED.values(), ids=list(SPOOFED))
def test_a_summary_not_shaped_like_a_run_ood_output_is_not_a_previous_run(
    tmp_path, capsys, unfitted, summary: dict[str, Any]
) -> None:
    write_release(tmp_path)
    (tmp_path / "ood").mkdir()
    (tmp_path / "ood/summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (tmp_path / "ood/report.docx").write_bytes(b"user file")
    refused(capsys, tmp_path, tmp_path / "ood", "pass --overwrite")


def test_run_ood_outputs_count_as_previous_runs(tmp_path) -> None:
    (tmp_path / "summary.json").write_text(json.dumps(marker()), encoding="utf-8")
    assert run_ood.previous_run(tmp_path)
    assert run_ood.previous_run(run_ood.REPO_ROOT / "results/ood/cordis_v1.1.0")


def test_an_output_below_a_file_fails_cleanly_before_fitting(tmp_path, capsys, unfitted) -> None:
    write_release(tmp_path)
    (tmp_path / "blocker.txt").write_text("notes", encoding="utf-8")
    assert run(tmp_path, ("en",), "--output-dir", str(tmp_path / "blocker.txt/sub")) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["status"] == "failed"
    assert (tmp_path / "blocker.txt").read_text(encoding="utf-8") == "notes"


def test_an_existing_file_is_refused_as_output_even_with_overwrite(
    tmp_path, capsys, unfitted
) -> None:
    write_release(tmp_path)
    (tmp_path / "ood").write_text("notes", encoding="utf-8")
    refused(capsys, tmp_path, tmp_path / "ood", "existing file", "--overwrite")


def test_overwrite_replaces_unrelated_content_and_a_rerun_replaces_its_own_output(
    tmp_path,
) -> None:
    write_release(tmp_path)
    output = tmp_path / "ood"
    output.mkdir()
    (output / "notes.txt").write_text("unrelated", encoding="utf-8")
    assert run(tmp_path, ("en",), "--output-dir", str(output), "--overwrite") == 0
    assert not (output / "notes.txt").exists()
    first = contents(output)
    assert run(tmp_path, ("en",), "--output-dir", str(output), "--models", "svm") == 0
    assert contents(output) != first and list(read_summary(tmp_path)["models"]) == ["svm"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["data", "ood"]


def test_a_previous_output_that_cannot_be_deleted_is_reported(
    tmp_path, monkeypatch, capsys
) -> None:
    write_release(tmp_path)
    assert run(tmp_path) == 0
    previous = contents(tmp_path / "ood")
    monkeypatch.setattr(run_ood.shutil, "rmtree", lambda *_, **__: None)
    assert run(tmp_path, ("en",), "--models", "svm") == 0
    done = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert done["status"] == "success"
    assert contents(Path(done["previous_output"])) == previous
