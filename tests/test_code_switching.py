from __future__ import annotations

import copy
import hashlib
import sys
import tempfile
from pathlib import Path

from naltra.data.code_switching import (
    create_code_switched_record,
    generate_code_switch_benchmarks,
    get_code_switch_seed,
    mix_code_switched_text,
    pair_aligned_records,
)
from naltra.data.loader import load_jsonl, save_jsonl
from naltra.data.preprocessing import validate_record

# Add repo root and src/ to sys.path so tests can be run directly with python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_deterministic_seeding_and_repeatability() -> None:
    """Verify SHA-256 seeding produces identical seeds and outputs across separate calls."""
    pair_id = "cordis:548"
    strategy = "chunk_mix"
    strength = "balanced"
    seed = 42

    seed1 = get_code_switch_seed(pair_id, strategy, strength, seed)
    seed2 = get_code_switch_seed(pair_id, strategy, strength, seed)
    assert seed1 == seed2
    assert isinstance(seed1, int)

    expected_digest = hashlib.sha256(f"{pair_id}:{strategy}:{strength}:{seed}".encode()).digest()
    assert seed1 == int.from_bytes(expected_digest[:8], byteorder="big")

    en_text = "With the change from the quarter to the half mile run, speed becomes less important."
    tr_text = (
        "Çeyrek mil koşusunun yarım mil koşusuna dönüşmesiyle birlikte, "
        "hız daha az önemli hale gelir."
    )

    out1, lang1 = mix_code_switched_text(
        en_text, tr_text, strategy=strategy, strength=strength, seed=seed, pair_id=pair_id
    )
    out2, lang2 = mix_code_switched_text(
        en_text, tr_text, strategy=strategy, strength=strength, seed=seed, pair_id=pair_id
    )
    assert out1 == out2
    assert lang1 == lang2

    # Different pair_id produces a different seed
    seed_diff = get_code_switch_seed("cordis:999", strategy, strength, seed)
    assert seed1 != seed_diff


def test_original_records_unmodified() -> None:
    """Verify that creating code-switched records leaves original records completely unchanged."""
    en_rec = {
        "id": "cordis:548:en",
        "text": "Speed becomes of much less importance.",
        "labels": ["sport"],
        "language": "en",
        "source": "cordis_h2020",
        "license": "CC BY 4.0",
        "split": "validation",
        "pair_id": "cordis:548",
        "source_id": 548,
        "source_labels": ["sports"],
    }
    tr_rec = {
        "id": "cordis:548:tr",
        "text": "Hız çok daha az önemli hale geliyor.",
        "labels": ["sport"],
        "language": "tr",
        "source": "cordis_h2020",
        "license": "CC BY 4.0",
        "split": "validation",
        "pair_id": "cordis:548",
        "source_id": 548,
        "source_labels": ["sports"],
    }

    en_backup = copy.deepcopy(en_rec)
    tr_backup = copy.deepcopy(tr_rec)

    cs_rec = create_code_switched_record(en_rec, tr_rec, strategy="chunk_mix", strength="balanced")

    # Mutate the output record
    cs_rec["labels"].append("mutated")
    cs_rec["text"] = "Altered text"

    # Verify sources remain unchanged
    assert en_rec == en_backup
    assert tr_rec == tr_backup


def test_record_structure_and_traceability() -> None:
    """Verify generated record fields, metadata, ID format, and schema compliance."""
    en_rec = {
        "id": "cordis:548:en",
        "text": "Speed becomes of much less importance and endurance becomes a necessity.",
        "labels": ["sport"],
        "language": "en",
        "source": "cordis_h2020",
        "license": "CC BY 4.0",
        "split": "validation",
        "pair_id": "cordis:548",
    }
    tr_rec = {
        "id": "cordis:548:tr",
        "text": (
            "Hız çok daha az önemli hale geliyor ve dayanıklılık bir gereklilik haline geliyor."
        ),
        "labels": ["sport"],
        "language": "tr",
        "source": "cordis_h2020",
        "license": "CC BY 4.0",
        "split": "validation",
        "pair_id": "cordis:548",
    }

    record = create_code_switched_record(
        en_rec, tr_rec, strategy="chunk_mix", strength="balanced", base_seed=42
    )

    assert record["id"] == "cordis:548:codeswitch:chunk_mix:balanced"
    assert record["pair_id"] == "cordis:548"
    assert record["original_pair_id"] == "cordis:548"
    assert record["en_id"] == "cordis:548:en"
    assert record["tr_id"] == "cordis:548:tr"
    assert record["language"] == "en-tr"
    assert record["labels"] == ["sport"]
    assert record["source"] == "cordis_h2020"
    assert record["split"] == "validation"
    assert record["license"] == "CC BY 4.0"
    assert record["code_switch_strategy"] == "chunk_mix"
    assert record["code_switch_strength"] == "balanced"
    assert record["primary_language"] in ("en", "tr")
    assert isinstance(record["text"], str) and len(record["text"].strip()) > 0

    # Ensure schema validator accepts the generated record
    validate_record(record)


def test_both_languages_contribute_tokens() -> None:
    """Verify that both source languages contribute tokens in balanced mode."""
    en_text = "The national economic growth expanded rapidly over the recent fiscal quarter."
    tr_text = "Ulusal ekonomik büyüme son mali çeyrekte hızla genişleme kaydetti."

    mixed, primary = mix_code_switched_text(
        en_text, tr_text, strategy="chunk_mix", strength="balanced", seed=42, pair_id="pair_check"
    )

    en_token_set = set(en_text.split())
    tr_token_set = set(tr_text.split())
    mixed_tokens = mixed.split()

    has_en = any(t in en_token_set for t in mixed_tokens)
    has_tr = any(t in tr_token_set for t in mixed_tokens)

    assert has_en, "Mixed text should contain English tokens"
    assert has_tr, "Mixed text should contain Turkish tokens"


def test_pair_aligned_records_validation() -> None:
    """Verify strict alignment validation for pair matching."""
    # Valid matching pair
    records = [
        {
            "id": "en:1",
            "language": "en",
            "pair_id": "p1",
            "split": "test",
            "labels": ["tech"],
            "text": "Hi",
        },
        {
            "id": "tr:1",
            "language": "tr",
            "pair_id": "p1",
            "split": "test",
            "labels": ["tech"],
            "text": "Selam",
        },
    ]
    pairs = pair_aligned_records(records)
    assert len(pairs) == 1
    assert pairs[0][0]["id"] == "en:1"
    assert pairs[0][1]["id"] == "tr:1"

    # Mismatched labels
    bad_labels = [
        {
            "id": "en:1",
            "language": "en",
            "pair_id": "p1",
            "split": "test",
            "labels": ["tech"],
            "text": "Hi",
        },
        {
            "id": "tr:1",
            "language": "tr",
            "pair_id": "p1",
            "split": "test",
            "labels": ["sports"],
            "text": "Selam",
        },
    ]
    try:
        pair_aligned_records(bad_labels)
    except ValueError as exc:
        assert "Label mismatch" in str(exc)
    else:
        raise AssertionError("Expected ValueError for label mismatch")

    # Mismatched split
    bad_split = [
        {
            "id": "en:1",
            "language": "en",
            "pair_id": "p1",
            "split": "validation",
            "labels": ["tech"],
            "text": "Hi",
        },
        {
            "id": "tr:1",
            "language": "tr",
            "pair_id": "p1",
            "split": "test",
            "labels": ["tech"],
            "text": "Selam",
        },
    ]
    try:
        pair_aligned_records(bad_split)
    except ValueError as exc:
        assert "Split mismatch" in str(exc)
    else:
        raise AssertionError("Expected ValueError for split mismatch")

    # Missing language
    missing_tr = [
        {
            "id": "en:1",
            "language": "en",
            "pair_id": "p1",
            "split": "test",
            "labels": ["tech"],
            "text": "Hi",
        },
    ]
    try:
        pair_aligned_records(missing_tr)
    except ValueError as exc:
        assert "Incomplete pair" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing language in pair")


def test_input_validation_and_empty_text() -> None:
    """Verify error handling on invalid strategy, strength, or empty text."""
    try:
        mix_code_switched_text("Hello world", "Merhaba dunya", strategy="nonexistent_strategy")
    except ValueError as exc:
        assert "Unknown code-switch strategy" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid strategy")

    try:
        mix_code_switched_text("Hello world", "Merhaba dunya", strength="extreme")
    except ValueError as exc:
        assert "Unknown code-switch strength" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid strength")

    try:
        mix_code_switched_text("   ", "Merhaba dunya")
    except ValueError as exc:
        assert "empty or whitespace-only" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty text")


def test_mock_benchmark_generation_and_split_safety() -> None:
    """Verify generate_code_switch_benchmarks respects directory hierarchy and ignores train."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_in = Path(tmp_dir) / "cordis_h2020"
        tmp_out = Path(tmp_dir) / "code_switch"
        tmp_in.mkdir(parents=True, exist_ok=True)

        for split in ("train", "validation", "test"):
            recs = [
                {
                    "id": f"cordis:{i}:en",
                    "text": f"English text record {i} for {split}.",
                    "labels": ["science_technology"],
                    "language": "en",
                    "source": "cordis_h2020",
                    "license": "CC BY 4.0",
                    "split": split,
                    "pair_id": f"cordis:{i}",
                }
                for i in range(10)
            ] + [
                {
                    "id": f"cordis:{i}:tr",
                    "text": f"Türkçe metin kaydı {i} split {split}.",
                    "labels": ["science_technology"],
                    "language": "tr",
                    "source": "cordis_h2020",
                    "license": "CC BY 4.0",
                    "split": split,
                    "pair_id": f"cordis:{i}",
                }
                for i in range(10)
            ]
            save_jsonl(recs, tmp_in / f"{split}.jsonl")

        results = generate_code_switch_benchmarks(
            processed_base_dir=tmp_in,
            output_base_dir=tmp_out,
            splits=("validation", "test"),
            strategy="chunk_mix",
            strength="balanced",
            seed=42,
        )

        assert results == {"validation": 10, "test": 10}

        val_file = tmp_out / "chunk_mix" / "balanced" / "validation.jsonl"
        test_file = tmp_out / "chunk_mix" / "balanced" / "test.jsonl"
        train_file = tmp_out / "chunk_mix" / "balanced" / "train.jsonl"

        assert val_file.exists()
        assert test_file.exists()
        assert not train_file.exists(), "Train split must not be created by default"

        val_records = load_jsonl(val_file)
        assert len(val_records) == 10
        for r in val_records:
            assert r["language"] == "en-tr"
            assert r["code_switch_strategy"] == "chunk_mix"
            assert r["code_switch_strength"] == "balanced"


def test_missing_input_preflight_check() -> None:
    """Verify generation fails clearly before writing when the input directory is missing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            generate_code_switch_benchmarks(
                processed_base_dir=Path(tmp_dir) / "non_existent",
                output_base_dir=Path(tmp_dir) / "cs_out",
            )
        except FileNotFoundError as exc:
            assert "Cannot generate code-switch benchmarks" in str(exc)
        else:
            raise AssertionError("Expected FileNotFoundError for a missing input directory")


if __name__ == "__main__":
    print("Running test_code_switching suite...")
    test_deterministic_seeding_and_repeatability()
    test_original_records_unmodified()
    test_record_structure_and_traceability()
    test_both_languages_contribute_tokens()
    test_pair_aligned_records_validation()
    test_input_validation_and_empty_text()
    test_mock_benchmark_generation_and_split_safety()
    test_missing_input_preflight_check()
    print("All 8 code-switching benchmark tests passed successfully!")
