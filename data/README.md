# Dataset card: the NALTRA CORDIS EN-TR benchmark, release 1.1.0

The NALTRA CORDIS EN-TR benchmark, release 1.1.0, pairs every usable Horizon 2020 project
objective in CORDIS with a Turkish machine translation and two synthetic EN-TR code-switched
versions, all with the same EuroSciVoc labels and the same project split; seeded noisy test
copies are generated locally. It is built from one public source; no other corpus is added.
This directory holds the source archive and the prepared tracks. Dataset files are CC BY 4.0;
see [LICENSE](LICENSE) for the attribution and the changes made. Preparation, record fields,
and audits are documented in the [dataset guide](../docs/dataset.md).

## At a glance

| Track | Splits | Records | Words per objective, mean / median | Notes |
| --- | --- | ---: | ---: | --- |
| EN | train, validation, test | 31,407 | 266.1 / 276 | cleaned CORDIS objectives |
| TR | train, validation, test | 31,407 | 216.9 / 227 | `Helsinki-NLP/opus-mt-tc-big-en-tr`; 283,002 aligned sentence pairs |
| Code-switch `sentence_mix` | validation, test | 9,422 | 241.3 / 253 | sentences alternate; English is the primary language in 50.1% |
| Code-switch `chunk_mix` | validation, test | 9,422 | 241.3 / 253 | first half / second half; English primary in 49.8% |
| Noisy EN, noisy TR | test | 4,711 each | | generated locally, not shipped ([parameters](../docs/dataset.md#noisy-test-copies)) |

- **Projects**: 31,407 of the archive's 35,389 records (those with an objective and at least
  one supported label), split by project into 21,985 / 4,711 / 4,711 train / validation /
  test, identical in every track.
- **Labels**: 473 direct labels in a 586-node EuroSciVoc tree (6 roots, depth 1-7); 3.21
  direct labels per project (10.24 with ancestors); 84.3% of projects have more than one, and
  59.8% span more than one root.
- **Length**: words are whitespace tokens; in characters, EN objectives average 1,843
  (median 1,960) and TR 1,787 (median 1,884).
- **Size**: 100.9 MB of compressed splits, plus the 55.2 MB raw archive.

The track figures and the Turkish screen in [Known issues](#known-issues) (item 6) come from
the committed splits; run from the repository root:

```python
import glob, gzip, json, statistics

def records(track):
    for path in sorted(glob.glob(f"data/processed/cordis_h2020/{track}/*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as lines:
            yield from map(json.loads, lines)

for track in ("en", "tr", "code_switch/sentence_mix/balanced", "code_switch/chunk_mix/balanced"):
    words = [len(r["text"].split()) for r in records(track)]
    print(track, len(words), round(statistics.mean(words), 1), statistics.median(words))
tr = list(records("tr"))
print(sum(len(r["sentence_alignment"]) for r in tr),  # aligned sentence pairs
      sum(any(c in "çğıöşüÇĞİÖŞÜ" for c in r["text"]) for r in tr),  # Turkish letters
      sum("�" in r["text"] for r in tr),  # U+FFFD
      sum(r["text"] == r["english_source_text"] for r in tr))  # untranslated
```

## Intended use

- Benchmarking hierarchical multi-label classification of research text into EuroSciVoc,
  with direct-label and ancestor-closed (hierarchical) metrics.
- Paired robustness measurement: every track holds the same projects with the same labels,
  so EN -> TR, code-switch and noise drops are paired differences.
- Near-OOD rejection with one EuroSciVoc root held out of training (`scripts/run_ood.py`).

## Out-of-scope use

- Judging Turkish translation quality or training translation systems: TR is unreviewed
  machine output.
- Claims about natural Turkish or natural EN-TR code-switching: TR is translated from
  English, and the code-switch tracks mix whole sentences synthetically.
- Assessing, ranking or funding projects, people or organisations: the labels are silver.

## Known issues

1. **Silver labels.** The archive's field description (`information.zip`,
   `DET_fields_description.pdf`) does not say how CORDIS assigned EuroSciVoc categories to
   projects, so we treat them as silver labels, not human gold annotation.
2. **Rare codes dropped, not rolled up.** `labels_direct` keeps only codes with >= 50
   supporting projects (so does `source_label_ids`); sub-threshold codes are dropped, not
   rolled up to their nearest supported ancestor. 8,128 of the 31,407 retained projects lose
   at least one directly assigned code (9,909 codes in total). Within the active 586-node
   taxonomy, 1,039 of the 4,711 test projects lack 2,375 CORDIS-implied nodes, and 327 test
   projects miss a whole root. Restoring those active nodes changes hierarchical F1 by at most
   0.007 (SVM 0.668 -> 0.662) and no model ranking; full unpruned-tree performance was not
   evaluated. Counting the code paths the taxonomy pruned as well, 1,231 test projects lack
   3,744 (counted, not scored; `results/metrics/label_coverage.json`, from
   `scripts/extra_analyses.py`).
3. **Label universe uses test projects.** The >= 50 threshold counts every project in the
   archive, test projects included ([results limitations](../docs/results.md#limitations)).
4. **Near-OOD and dropped codes.** The held-out-root split looks at supported labels only, so
   165 train and 19 validation projects with a sub-threshold CORDIS humanities code stay in
   the near-OOD training data, and 34 of the 4,351 in-domain test projects carry one.
5. **No human translation review.** No human has reviewed the Turkish text (the
   human-review columns in `data/review/` are still `pending`). Those columns of
   `review/translation_review_sample.csv` (`review_status`, `meaning_preserved`,
   `terminology_quality`, `fluency`) read `pending` in all 100 rows, which are validation
   projects. What exists is an AI-assisted review of the same 100 projects
   (`review/translation_ai_review.csv`: meaning preserved `yes` 83, `partial` 17, `no` 0);
   its model, prompt and date were not recorded. Both files predate TR 1.1.0: 82 rows match
   the shipped text, while the other 18 refer to repaired projects and match neither archived
   release. `top_level_domain` holds each project's first direct label, not its root
   ([translation review](../docs/dataset.md#translation-review)).
6. **Automatic quality checks only.** The pipeline checks length ratios, identity and
   alignment. A one-off screen (snippet above) finds Turkish-specific letters in all 31,407 TR
   records, U+FFFD in none, and no record equal to its English source. Language ID reads
   31,400 TR records (99.98%) as Turkish and 7 as mixed EN-TR, none as English. Translated back
   to English, 300 seeded TR test records reach a mean round-trip chrF of 77.1 against the
   English source (minimum 40.6); the 37 test records flagged in item 7 average 57.3
   (`results/metrics/translation_quality.json`;
   [translation quality](../docs/dataset.md#translation-quality)). These are proxies, not
   measures of fluency or faithfulness. TR titles stay English; the models read only `text`.
7. **Possible truncations and repetitions in TR.** Two automatic screens that the
   per-document length ratio misses flag 180 of 31,407 TR records (0.57%) as possible
   truncations, e.g. test project 640771, and 48 (0.15%) as possible repetitions
   (`results/metrics/translation_checks.json`; definitions and per-split counts in
   [translation screens](../docs/dataset.md#translation-screens)). Flags are candidates, not
   confirmed errors; the released text is unchanged, and their effect on scores is not
   measured.
8. **TR 1.1.0 cannot be rebuilt.** It is the 1.0.0 import with 598 records re-translated in
   128-token chunks by a repair script that is not in the repository
   ([provenance handoff](../docs/reproducibility.md#training-and-data-provenance)); the
   committed scripts build a different TR release. The prebuilt audit tolerates 15,434
   segmenter-drift records ([details](../docs/dataset.md#shipped-turkish-release-110)).
9. **Cleaning loss.** The HTML-tag pattern also deletes `<…>` comparison spans in 62 retained
   projects (17,290 characters); labels are unaffected
   ([details](../docs/dataset.md#known-limitation-angle-bracket-comparisons)).
10. **Noisy track is generated, not shipped.** `scripts/prepare_data.py --dataset noisy`
    (`combined`, severity medium = 0.10, seed 42) gives each document one main edit type, so
    corruption is bimodal: a mean of 32.5% (EN) and 37.3% (TR) of space-separated tokens
    change, but about 31% of documents have under 15% changed
    ([parameters](../docs/dataset.md#noisy-test-copies)).
11. **Synthetic, inter-sentential code-switching.** Both tracks switch at sentence
    boundaries or document halves; switching inside a sentence occurs only in the 11
    single-sentence documents per track, so it is not covered
    ([derived tracks](../docs/dataset.md#derived-evaluation-tracks)).
12. **Near-duplicates.** Split grouping catches only identical objectives; near-identical
    ones can cross splits ([results limitations](../docs/results.md#limitations)).
13. **Domain skew.** 67.8% of projects carry a natural-sciences label and 41.2% an
    engineering one, against 7.6% humanities and 6.6% agricultural sciences.
14. **Historical figures.** The 1.0.0 translation runtime, sentence statistics and
    before-hardening audit come from runs whose logs were not kept; they cannot be recomputed.

## Directory layout

| Path | Contents | In Git |
| --- | --- | --- |
| `raw/cordis_h2020/cordis-h2020projects-csv.zip` | Original source archive, SHA-256 locked in `configs/data.yaml` | Yes |
| `processed/cordis_h2020/en/` | English train, validation, and test splits | Compressed files and manifest |
| `processed/cordis_h2020/tr/` | Turkish versions of the same projects | Compressed files and manifest |
| `processed/cordis_h2020/code_switch/{sentence_mix,chunk_mix}/balanced/` | Synthetic mixed-language validation/test splits | Compressed files and manifests |
| `splits/cordis_h2020/project_splits.json` | Shared project-to-split assignment | Yes |
| `review/` | Translation review sample and AI-assisted review (known issue 5) | Yes |
| `noisy/` | Generated noisy test copies | No |
| `samples/{val80,test120}/` | Fixed EN/TR evaluation samples for Kev, Jev and Laya | No |
| `cache/` | Translation caches | No |

Each manifest lists the SHA-256, size, and record count of its splits. Extracted CSVs
and decompressed `.jsonl` files also stay local.

## Source

- **CORDIS**: EU research projects under Horizon 2020 (2014-2020), European Commission /
  Publications Office of the European Union,
  `https://cordis.europa.eu/data/cordis-h2020projects-csv.zip`, CC BY 4.0 (attribution
  "European Union, CORDIS - https://cordis.europa.eu - CC BY 4.0"). Input text: the
  `objective` field. Labels: the EuroSciVoc category paths in `euroSciVoc.csv` (taxonomy
  0.4.0: 6 roots, 586 active categories).
- **Turkish**: machine translation of the identical English records by
  `Helsinki-NLP/opus-mt-tc-big-en-tr` @ `e539fc16a8a1a0ea5950eb339b595bfcce990e90`
  ([translation](../docs/dataset.md#translation)).
- **Code-switch**: aligned English and Turkish sentences of the same project, mixed
  deterministically with a per-pair seed.

## Restore the committed release

Run from the repository root:

```bash
python scripts/unpack_data.py
python scripts/validate_datasets.py --prebuilt-release
```

Unpacking verifies each split's checksum and size before replacing its local
`.jsonl`. Matching files are skipped. The full validator refuses the shipped release
([why](../docs/dataset.md#release-validation)); training on it takes the same
`--prebuilt-release` flag ([runbook](../docs/runbook.md#2-train)).

## Generate variants or rebuild data

Commands for noisy copies, samples, and a new dataset release are in the
[runbook](../docs/runbook.md#0-data). A rebuild changes tracked manifests and may
change the split map or review sample; use a separate checkout and a new release
version.

The shipped Turkish 1.1.0 release includes a targeted repair of 598 records.
A standard translation run does not recreate that repair. The repair implementation
is part of the [outstanding provenance handoff](../docs/reproducibility.md#training-and-data-provenance).

## Versioning and maintenance

- Release 1.1.0 is identified by the per-split SHA-256s in the four committed manifests
  (EN, TR, both code-switch tracks). `manifest_version` is the manifest schema version
  (`MANIFEST_SCHEMA_VERSION` in `src/naltra/data/manifest.py`), which any rebuild also
  writes, so a rebuilt release does not identify itself as a new version.
- The 1.1.0 files are hash-locked and not changed in place: a change to text or labels, such
  as a fix for a known issue above, needs a new release version and a re-evaluation of every
  member.
- Maintainer: the data and taxonomy owner in [team ownership](../docs/team_ownership.md);
  report problems through the repository's GitHub issues.

## How to cite

```bibtex
@misc{naltra_cordis_entr_2026,
  title        = {The {NALTRA} {CORDIS} {EN-TR} Benchmark, Release 1.1.0},
  author       = {Alkabi, Abdalraouf and Gamal, Abdullah and Alzyod, Omar Ibrahim Fahmi and
                  Alakili, Almutesim and Kamaleldin, Mohamed and Isaoglu, Rasit},
  year         = {2026},
  howpublished = {\url{https://github.com/Rxd3/NALTRA}},
  note         = {Derived from CORDIS H2020 projects, European Union, CC BY 4.0}
}
```

Any reuse must also carry the source attribution: "European Union, CORDIS -
https://cordis.europa.eu - CC BY 4.0" ([LICENSE](LICENSE)).
