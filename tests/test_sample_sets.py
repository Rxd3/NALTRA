"""Fixed pair-hash samples keep the same projects in every language."""

from __future__ import annotations

import json

import pytest
from scripts import sample_sets

from naltra.data.loader import load_jsonl, save_jsonl


def write(tmp_path, language: str, pairs: range) -> str:
    path = tmp_path / language / "test.jsonl"
    save_jsonl(
        [{"id": f"p{i}:{language}", "pair_id": f"p{i}", "text": f"t{i}"} for i in pairs], path
    )
    return f"{language}_test={path}"


def run(tmp_path, *sets: str, pairs: int = 3) -> int:
    return sample_sets.main(
        ["--sets", *sets, "--pairs", str(pairs), "--output-dir", str(tmp_path / "out")]
    )


def test_every_language_keeps_the_same_smallest_hash_pairs(tmp_path, capsys) -> None:
    assert run(tmp_path, write(tmp_path, "en", range(20)), write(tmp_path, "tr", range(20))) == 0
    en = load_jsonl(tmp_path / "out/en_test.jsonl")
    tr = load_jsonl(tmp_path / "out/tr_test.jsonl")
    assert len(en) == 3
    assert [r["pair_id"] for r in en] == [r["pair_id"] for r in tr]
    expected = sorted((f"p{i}" for i in range(20)), key=sample_sets.pair_hash)[:3]
    assert {r["pair_id"] for r in en} == set(expected)
    status = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert status["status"] == "success" and status["sets"]["en_test"]["records"] == 3


def test_sample_keeps_file_order_and_is_deterministic(tmp_path) -> None:
    spec = write(tmp_path, "en", range(20))
    assert run(tmp_path, spec) == 0
    first = (tmp_path / "out/en_test.jsonl").read_bytes()
    order = [int(r["pair_id"][1:]) for r in load_jsonl(tmp_path / "out/en_test.jsonl")]
    assert order == sorted(order)
    assert run(tmp_path, spec) == 0
    assert (tmp_path / "out/en_test.jsonl").read_bytes() == first


def test_records_without_pair_id_are_refused(tmp_path, capsys) -> None:
    path = tmp_path / "bad.jsonl"
    save_jsonl([{"id": "x", "text": "t"}], path)
    assert run(tmp_path, f"bad={path}") == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["status"] == "failed"
    assert not (tmp_path / "out/bad.jsonl").exists()


def test_a_sample_needs_at_least_one_pair(tmp_path) -> None:
    spec = write(tmp_path, "en", range(5))
    with pytest.raises(SystemExit) as result:
        run(tmp_path, spec, pairs=0)
    assert result.value.code == 2
