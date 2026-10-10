from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pytest

from naltra.data.loader import load_jsonl, save_jsonl
from naltra.data.noise import (
    VALID_STRATEGIES,
    create_noisy_record,
    generate_noisy_benchmarks,
    get_deterministic_seed,
    perturb_text,
)

# Add repo root and src/ to sys.path so tests can be run directly with python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_deterministic_seeding_across_calls() -> None:
    """Verify SHA-256 seeding produces identical seeds and outputs across separate calls."""
    record_id = "cordis:101:en"
    strategy = "combined"
    severity = "medium"
    base_seed = 42

    seed1 = get_deterministic_seed(record_id, strategy, severity, base_seed)
    seed2 = get_deterministic_seed(record_id, strategy, severity, base_seed)
    assert seed1 == seed2
    assert isinstance(seed1, int)

    # Cross-check against manual hashlib computation
    expected_digest = hashlib.sha256(
        f"{record_id}:{strategy}:{severity}:{base_seed}".encode()
    ).digest()
    expected_seed = int.from_bytes(expected_digest[:8], byteorder="big")
    assert seed1 == expected_seed

    # Verify perturbation text is 100% deterministic across separate invocations
    text = "The quick brown fox jumps over the lazy dog."
    noisy1 = perturb_text(
        text, strategy=strategy, severity=severity, seed=base_seed, record_id=record_id
    )
    noisy2 = perturb_text(
        text, strategy=strategy, severity=severity, seed=base_seed, record_id=record_id
    )
    assert noisy1 == noisy2

    # Different record ID or seed must produce distinct seed
    seed_diff = get_deterministic_seed("cordis:102:en", strategy, severity, base_seed)
    assert seed1 != seed_diff


def test_deep_copy_isolation() -> None:
    """Verify create_noisy_record uses deepcopy and does not mutate or share references."""
    original = {
        "id": "sample:001",
        "text": "This is a clean sentence for evaluation.",
        "labels": ["technology", "business"],
        "language": "en",
        "source": "unit_test",
        "source_id": "orig_001",
        "split": "test",
        "source_labels": ["tech", "biz"],
        "license": "CC-BY-4.0",
        "metadata": {"nested_key": [1, 2, 3]},
    }
    original_copy = copy.deepcopy(original)

    noisy = create_noisy_record(original, strategy="combined", severity="medium", base_seed=42)

    # Mutate the noisy record's mutable structures
    noisy["labels"].append("finance")
    noisy["source_labels"].append("extra")
    noisy["metadata"]["nested_key"].append(999)

    # Verify the original record was completely untouched
    assert original["labels"] == original_copy["labels"]
    assert original["source_labels"] == original_copy["source_labels"]
    assert original["metadata"]["nested_key"] == original_copy["metadata"]["nested_key"]
    assert original["text"] == original_copy["text"]
    assert original["id"] == original_copy["id"]


def test_text_actually_changes_and_never_empty() -> None:
    """Verify that every strategy changes text and never produces an empty string."""
    short_text = "Natural language processing benchmarks."
    for strat in sorted(VALID_STRATEGIES):
        if strat == "turkish_diacritics":
            # Test with Turkish text
            tr_text = "İstanbul ve Çanakkale boğazları çok güzeldir."
            noisy = perturb_text(
                tr_text, strategy=strat, severity="light", seed=42, language="tr", record_id="tr1"
            )
            assert noisy != tr_text
            assert len(noisy.strip()) > 0
        else:
            noisy = perturb_text(
                short_text,
                strategy=strat,
                severity="light",
                seed=42,
                language="en",
                record_id="en1",
            )
            assert noisy != short_text
            assert len(noisy.strip()) > 0

    # Test edge case: short word with light severity fallback
    single_word = "benchmark"
    noisy_word = perturb_text(
        single_word, strategy="char_swap", severity="light", seed=123, record_id="w1"
    )
    assert noisy_word != single_word
    assert len(noisy_word.strip()) > 0


def test_turkish_diacritics_behavior_and_language_guard() -> None:
    """Verify Turkish diacritic folding and strict error when called on non-tr language."""
    tr_text = "Türkçe metinlerde ç, ş, ğ, ı, ö, ü harfleri bulunur."
    noisy_tr = perturb_text(
        tr_text, strategy="turkish_diacritics", severity="medium", language="tr", record_id="tr2"
    )

    assert noisy_tr != tr_text
    # Should replace Turkish diacritics with ASCII equivalents
    assert any(c in noisy_tr for c in ["c", "s", "g", "i", "o", "u"])

    # Must raise ValueError when language != 'tr'
    try:
        perturb_text(
            "English text here.", strategy="turkish_diacritics", language="en", record_id="en2"
        )
    except ValueError as exc:
        assert "only valid for language 'tr'" in str(exc)
    else:
        raise AssertionError("Expected ValueError when applying turkish_diacritics to English text")

    # In create_noisy_record with en record
    en_record = {"id": "en:1", "text": "Some text", "language": "en"}
    try:
        create_noisy_record(en_record, strategy="turkish_diacritics")
    except ValueError as exc:
        assert "only valid for language 'tr'" in str(exc)
    else:
        raise AssertionError("Expected ValueError in create_noisy_record for non-tr language")


def test_labels_and_metadata_preserved() -> None:
    """Verify all labels, language, source, license, and split remain intact."""
    clean_record = {
        "id": "cordis:42:tr",
        "text": "Merkez Bankası enflasyon tahminini açıkladı.",
        "labels": ["economy"],
        "language": "tr",
        "source": "cordis_h2020",
        "source_id": "42",
        "split": "validation",
        "license": "CC BY 4.0",
        "pair_id": "cordis:42",
    }

    noisy_record = create_noisy_record(
        clean_record,
        strategy="combined",
        severity="medium",
        base_seed=100,
    )

    # Core metadata preservation
    assert noisy_record["labels"] == clean_record["labels"]
    assert noisy_record["language"] == clean_record["language"]
    assert noisy_record["source"] == clean_record["source"]
    assert noisy_record["source_id"] == clean_record["source_id"]
    assert noisy_record["split"] == clean_record["split"]
    assert noisy_record["license"] == clean_record["license"]
    assert noisy_record["pair_id"] == clean_record["pair_id"]

    # Traceability attributes
    assert noisy_record["original_id"] == "cordis:42:tr"
    assert noisy_record["id"] == "cordis:42:tr:noise:combined:medium"
    assert noisy_record["noise_strategy"] == "combined"
    assert noisy_record["noise_severity"] == "medium"
    assert noisy_record["text"] != clean_record["text"]


def test_individual_perturbation_strategies() -> None:
    """Test character swap, deletion, duplication, whitespace, capitalization, punctuation."""
    text = "Machine learning models classify documents accurately."

    swap_out = perturb_text(text, strategy="char_swap", severity="medium", seed=1, record_id="r1")
    assert swap_out != text

    del_out = perturb_text(text, strategy="char_delete", severity="medium", seed=2, record_id="r2")
    assert del_out != text
    assert len(del_out) <= len(text)

    dup_out = perturb_text(
        text, strategy="char_duplicate", severity="medium", seed=3, record_id="r3"
    )
    assert dup_out != text
    assert len(dup_out) >= len(text)

    ws_out = perturb_text(text, strategy="whitespace", severity="medium", seed=4, record_id="r4")
    assert ws_out != text

    cap_out = perturb_text(
        text, strategy="capitalization", severity="medium", seed=5, record_id="r5"
    )
    assert cap_out != text

    punct_text = "Wait, is this ready? Yes, absolutely!"
    punct_out = perturb_text(
        punct_text, strategy="punctuation", severity="medium", seed=6, record_id="r6"
    )
    assert punct_out != punct_text


def test_invalid_parameters_raise_appropriate_errors() -> None:
    """Verify input validation on invalid strategies, severities, and record formats."""
    # Invalid strategy
    try:
        perturb_text("Valid text.", strategy="invalid_strategy_xyz")
    except ValueError as exc:
        assert "Unknown noise strategy" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown strategy")

    # Invalid severity string
    try:
        perturb_text("Valid text.", severity="super_extreme")
    except ValueError as exc:
        assert "Unknown severity level" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown severity")

    # Invalid numeric severity out of range
    try:
        perturb_text("Valid text.", severity=1.5)
    except ValueError as exc:
        assert "Numeric severity must be in range" in str(exc)
    else:
        raise AssertionError("Expected ValueError for out-of-range severity")

    # Empty text
    try:
        perturb_text("   ")
    except ValueError as exc:
        assert "Cannot perturb empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty text")

    # Invalid record dict
    try:
        create_noisy_record({"no_id": True, "text": "Hello"})
    except ValueError as exc:
        assert "Record must be a dictionary" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing id in record")


def test_generate_noisy_benchmarks_directory_structure_and_counts() -> None:
    """Verify noisy benchmark generation and split filtering."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_proc = Path(tmp_dir) / "processed"
        tmp_noisy = Path(tmp_dir) / "noisy"

        ds = "cordis_h2020"
        ds_dir = tmp_proc / ds
        ds_dir.mkdir(parents=True, exist_ok=True)

        for split in ("train", "validation", "test"):
            records = [
                {
                    "id": f"{ds}:{split}:{i}",
                    "text": f"This is mock record {i} for {ds} {split}.",
                    "labels": ["general"],
                    "language": "en",
                    "source": ds,
                    "source_id": f"s_{i}",
                    "split": split,
                }
                for i in range(5)
            ]
            save_jsonl(records, ds_dir / f"{split}.jsonl")

        counts = generate_noisy_benchmarks(
            processed_base_dir=tmp_proc,
            output_base_dir=tmp_noisy,
            datasets=["cordis_h2020"],
            splits=["validation", "test"],
            strategy="combined",
            severity="medium",
            seed=42,
        )

        assert counts == {
            "cordis_h2020": {"validation": 5, "test": 5},
        }

        for split in ("validation", "test"):
            expected_path = tmp_noisy / "combined" / "medium" / "cordis_h2020" / f"{split}.jsonl"
            assert expected_path.exists()

            loaded = load_jsonl(expected_path)
            assert len(loaded) == 5

            for rec in loaded:
                assert ":noise:combined:medium" in rec["id"]
                assert rec["noise_strategy"] == "combined"
                assert rec["noise_severity"] == "medium"
                assert rec["labels"] == ["general"]

        train_path = tmp_noisy / "combined" / "medium" / "cordis_h2020" / "train.jsonl"
        assert not train_path.exists()


def test_noisy_manifests_record_the_clean_source_relative_to_themselves(tmp_path: Path) -> None:
    record = {"id": "cordis:1:en", "text": "Optical crop sensing.", "labels": ["optics"]}
    save_jsonl([{**record, "language": "en", "split": "test"}], tmp_path / "clean/en/test.jsonl")
    generate_noisy_benchmarks(tmp_path / "clean", tmp_path / "noisy", datasets=["en"])
    out = tmp_path / "noisy/combined/medium"

    def recorded(directory: Path) -> str:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        return manifest["source_metadata"]["processed_base_dir"]

    assert recorded(out / "en") == "../../../../clean"
    assert recorded(out) == "../../../clean"


def test_single_character_inputs_across_strategies() -> None:
    """Verify single-character inputs change text, stay non-empty, and are reproducible."""
    single_char = "a"

    # Specifically test whitespace strategy on "a"
    noisy_ws = perturb_text(single_char, strategy="whitespace", severity="medium", seed=42)
    assert noisy_ws != single_char, "Whitespace strategy on 'a' must not return unchanged 'a'"
    assert len(noisy_ws.strip()) > 0
    # Reproducibility
    assert perturb_text(single_char, strategy="whitespace", severity="medium", seed=42) == noisy_ws

    # Test all valid strategies on "a"
    for strat in sorted(VALID_STRATEGIES):
        lang = "tr" if strat == "turkish_diacritics" else "en"
        noisy = perturb_text(
            single_char, strategy=strat, severity="medium", seed=100, language=lang
        )
        assert (
            noisy != single_char
        ), f"Strategy '{strat}' returned unchanged single character '{single_char}'"
        assert len(noisy.strip()) > 0, f"Strategy '{strat}' returned empty string"
        # Reproducibility check
        noisy_repeat = perturb_text(
            single_char, strategy=strat, severity="medium", seed=100, language=lang
        )
        assert noisy == noisy_repeat, f"Strategy '{strat}' is not reproducible with same seed"

    # Turkish diacritic single character
    noisy_tr = perturb_text(
        "ç", strategy="turkish_diacritics", severity="medium", seed=42, language="tr"
    )
    assert noisy_tr == "c"
    assert noisy_tr != "ç"

    # Single punctuation mark
    noisy_punct = perturb_text(".", strategy="punctuation", severity="medium", seed=42)
    assert noisy_punct != "."
    assert len(noisy_punct.strip()) > 0


@pytest.mark.parametrize("prebuilt", [False, True])
def test_prepare_data_noisy_writes_en_and_tr_test_sets(tmp_path, monkeypatch, prebuilt) -> None:
    """The CLI noisy step writes the EN and TR test sets every reported result reads."""
    from naltra import cli
    from naltra.data import validation

    audited = []
    validator = "validate_prebuilt_cordis_release" if prebuilt else "validate_cordis_release"
    monkeypatch.setattr(validation, validator, lambda base, languages: audited.append(languages))
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    for language in ("en", "tr"):
        for split in ("validation", "test"):
            record = {"id": f"cordis:{language}:{split}", "text": "Optical crop sensing."}
            save_jsonl(
                [record], tmp_path / "data/processed/cordis_h2020" / language / f"{split}.jsonl"
            )
    noise_dir = tmp_path / ("elsewhere" if prebuilt else "data/noisy")
    flags = ["--prebuilt-release", "--noise-dir", str(noise_dir)] if prebuilt else []

    cli.prepare_data(["--dataset", "noisy", *flags])

    assert audited == [["en", "tr"]]
    for language in ("en", "tr"):
        directory = noise_dir / "combined/medium/cordis_h2020" / language
        assert [r["original_id"] for r in load_jsonl(directory / "test.jsonl")] == [
            f"cordis:{language}:test"
        ]
        assert not (directory / "validation.jsonl").exists()


def test_missing_input_preflight_check(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Cannot generate noisy benchmarks"):
        generate_noisy_benchmarks(
            processed_base_dir=tmp_path / "non_existent",
            output_base_dir=tmp_path / "noisy_out",
        )


# (clean source, noisy output) sha256 of the shipped release 1.1.0 test splits.
SHIPPED_NOISY_TEST_SHA256 = {
    "en": (
        "4f8564416013dff1c07e51f2c1f2064c2013c144908a26beec8122f455614052",
        "09f9b6af9b690981e47620653d0be2819652363c79c3650a49a1036b15b4f1e7",
    ),
    "tr": (
        "d269933ab6d24dd26cbbda44f9bbe3c8c8fa71c60ed1b109d2e907a227ea0141",
        "724c800cd9af4889e22c78590fd88c2aabffb51126e77a926fdb46fdde99e26c",
    ),
}


def test_default_noise_spec_reproduces_shipped_test_sets(tmp_path) -> None:
    """The default spec regenerates the shipped noisy EN/TR test sets byte for byte."""
    from naltra.data.manifest import compute_file_sha256

    processed = PROJECT_ROOT / "data/processed"
    for language, (source, _) in SHIPPED_NOISY_TEST_SHA256.items():
        path = processed / "cordis_h2020" / language / "test.jsonl"
        if not path.exists() or compute_file_sha256(path) != source:
            pytest.skip("Shipped CORDIS 1.1.0 test splits are not unpacked.")

    generate_noisy_benchmarks(processed, tmp_path)

    for language, (_, noisy) in SHIPPED_NOISY_TEST_SHA256.items():
        path = tmp_path / "combined/medium/cordis_h2020" / language / "test.jsonl"
        assert compute_file_sha256(path) == noisy


if __name__ == "__main__":
    print("Running test_noise suite...")
    test_deterministic_seeding_across_calls()
    test_deep_copy_isolation()
    test_text_actually_changes_and_never_empty()
    test_turkish_diacritics_behavior_and_language_guard()
    test_labels_and_metadata_preserved()
    test_individual_perturbation_strategies()
    test_invalid_parameters_raise_appropriate_errors()
    test_generate_noisy_benchmarks_directory_structure_and_counts()
    test_single_character_inputs_across_strategies()
    print("All 9 noise benchmark tests passed successfully!")
