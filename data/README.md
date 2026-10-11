# Dataset card: the NALTRA CORDIS EN-TR benchmark, release 1.1.0

The NALTRA CORDIS EN-TR benchmark, release 1.1.0, pairs every usable Horizon 2020 project
objective in CORDIS with a Turkish machine translation and two synthetic EN-TR code-switched
versions, all with the same EuroSciVoc labels and the same project split; seeded noisy test
copies are generated locally. It is built from one public source; no other corpus is added.
Dataset files are CC BY 4.0; see [LICENSE](LICENSE) for the attribution and the changes
made. The full specification (schema, label policy, translation pipeline, audits) is in
[docs/dataset.md](../docs/dataset.md), and the preprocessing steps are in the
[README](../README.md#preprocessing-pipeline).

## At a glance

| Track | Splits | Records | Words per objective, mean / median | Notes |
| --- | --- | ---: | ---: | --- |
| EN | train, validation, test | 31,407 | 266.1 / 276 | cleaned CORDIS objectives |
| TR | train, validation, test | 31,407 | 216.9 / 227 | `Helsinki-NLP/opus-mt-tc-big-en-tr`; 283,002 aligned sentence pairs |
| Code-switch `sentence_mix` | validation, test | 9,422 | 241.3 / 253 | sentences alternate; English is the primary language in 50.1% |
| Code-switch `chunk_mix` | validation, test | 9,422 | 241.3 / 253 | first half / second half; English primary in 49.8% |
| Noisy EN, noisy TR | test | 4,711 each | | generated locally, not shipped ([parameters](../docs/dataset.md#10-noisy-test-track-generated-not-shipped)) |

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
      sum("\ufffd" in r["text"] for r in tr),  # U+FFFD
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
   archive, test projects included ([README](../README.md#limitations)).
4. **Near-OOD and dropped codes.** The held-out-root split looks at supported labels only, so
   165 train and 19 validation projects with a sub-threshold CORDIS humanities code stay in
   the near-OOD training data, and 34 of the 4,351 in-domain test projects carry one.
5. **No human translation review.** The human-review columns of
   `review/translation_review_sample.csv` (`review_status`, `meaning_preserved`,
   `terminology_quality`, `fluency`) read `pending` in all 100 rows. What exists is an
   AI-assisted audit of the same 100 validation projects (`review/translation_ai_review.csv`:
   meaning preserved `yes` 83, `partial` 17, `no` 0); its model, prompt and date were not
   recorded. Both files predate TR 1.1.0: 82 rows match the shipped text, the other 18 show
   Turkish that no release ships, and `top_level_domain` holds each project's first direct
   label, not its root ([details](../docs/dataset.md#8-translation-pipeline--turkish-derivation)).
6. **Structural QA only.** The pipeline checks length ratios, identity and alignment, not
   whether the output is Turkish. A one-off screen (snippet above) finds Turkish-specific
   letters in all 31,407 TR records, U+FFFD in none, and no record equal to its English
   source. TR titles stay English; the models read only `text`.
7. **Possible truncations and repetitions in TR.** Two automatic screens of the TR splits
   (`python scripts/extra_analyses.py translation`, `results/metrics/translation_checks.json`)
   flag candidates the structural QA (item 6; its length ratio is per document) does not
   catch. *Possible truncation*: at least one
   `sentence_alignment` pair whose English sentence is longer than 80 characters and whose
   Turkish sentence has fewer than 40% of its characters; 180 of 31,407 records (0.57%: 117
   train, 32 validation, 31 test, e.g. test project 640771). *Possible repetition*: the
   lowercased `\w+` words of the TR text hold a word n-gram (n >= 4) immediately followed by
   itself, and that doubled sequence does not occur in the paired English text; 48 records
   (0.15%: 37 train, 5 validation, 6 test). Flags are candidates, not confirmed errors; the
   released text is unchanged, and their effect on scores is not measured.
8. **TR 1.1.0 cannot be rebuilt.** It is the 1.0.0 import with 598 records re-translated in
   128-token chunks by a repair script that is not in the repository; the committed scripts
   build a different TR release. The prebuilt audit tolerates 15,434 segmenter-drift records
   ([details](../docs/dataset.md#shipped-turkish-release-110)).
9. **Cleaning loss.** The HTML-tag pattern also deletes `<…>` comparison spans in 62 retained
   projects (17,290 characters); labels are unaffected
   ([details](../docs/dataset.md#known-limitation-angle-bracket-comparisons)).
10. **Noisy track is generated, not shipped.** `scripts/prepare_data.py --dataset noisy`
    (`combined`, severity medium = 0.10, seed 42) gives each document one main edit type, so
    corruption is bimodal: a mean of 32.5% (EN) and 37.3% (TR) of space-separated tokens
    change, but about 31% of documents have under 15% changed
    ([parameters](../docs/dataset.md#10-noisy-test-track-generated-not-shipped)).
11. **Synthetic, inter-sentential code-switching.** Both tracks switch at sentence
    boundaries or document halves; switching inside a sentence occurs only in the 11
    single-sentence documents per track, so it is not covered.
12. **Near-duplicates.** Split grouping catches only identical objectives; near-identical
    ones can cross splits ([README](../README.md#limitations)).
13. **Domain skew.** 67.8% of projects carry a natural-sciences label and 41.2% an
    engineering one, against 7.6% humanities and 6.6% agricultural sciences.
14. **Historical figures.** The 1.0.0 translation runtime, sentence statistics and
    before-hardening audit come from runs whose logs were not kept; they cannot be recomputed.

## What Git tracks

The raw archive and the compressed processed splits are tracked, so no download is needed.
Everything that can be regenerated from them stays ignored through `.gitignore`.

| Path | Tracked | Contents |
| --- | --- | --- |
| `raw/cordis_h2020/cordis-h2020projects-csv.zip` | yes | Unmodified official archive, SHA-256 locked in `configs/data.yaml` |
| `processed/cordis_h2020/**/*.jsonl.gz`, `manifest.json` | yes | Cleaned splits; each manifest lists SHA-256, size and record count per split |
| `review/` | yes | Translation review sample and its AI-assisted audit, both drawn before TR 1.1.0 (Known issues, item 5) |
| `splits/cordis_h2020/project_splits.json` | yes | Project-level split map |
| `processed/**/*.jsonl` | no | Restored by `python scripts/unpack_data.py` (or `make data`) |
| `raw/cordis_h2020/*.csv` | no | Extracted by `scripts/prepare_data.py --download` |
| `cache/`, `noisy/`, `samples/` | no | Translation cache, noisy test copies, Kev, Jev and Laya evaluation samples |

`scripts/unpack_data.py` decompresses each `.jsonl.gz` next to itself and replaces the
`.jsonl` only after its SHA-256 and size match the manifest; files that already match are
skipped, and any mismatch fails without leaving a partial file.

```
data/
├── LICENSE                      # CC BY 4.0, CORDIS attribution, list of changes
├── raw/cordis_h2020/
│   ├── cordis-h2020projects-csv.zip
│   ├── project.csv              # extracted: 35,389 project records
│   └── euroSciVoc.csv           # extracted: EuroSciVoc category paths per project
├── processed/cordis_h2020/
│   ├── en/                      # train 21,985 / validation 4,711 / test 4,711
│   ├── tr/                      # the same projects, machine-translated
│   └── code_switch/{sentence_mix,chunk_mix}/balanced/   # validation and test only
├── review/                      # translation_review_sample.csv, translation_ai_review.csv
├── splits/cordis_h2020/project_splits.json            # project-level split map
├── cache/translations/cordis_h2020/                   # resumable translation cache
├── noisy/combined/medium/cordis_h2020/{en,tr}/test.jsonl   # prepare_data.py --dataset noisy
└── samples/{val80,test120}/                           # scripts/sample_sets.py, for Kev, Jev and Laya
```

## Source and derivation

- **Source**: CORDIS - EU research projects under Horizon 2020 (2014-2020), European
  Commission / Publications Office of the European Union,
  `https://cordis.europa.eu/data/cordis-h2020projects-csv.zip`, CC BY 4.0 (attribution
  "European Union, CORDIS - https://cordis.europa.eu - CC BY 4.0"). Input text: the
  `objective` field. Labels: the EuroSciVoc category paths in `euroSciVoc.csv` (taxonomy
  0.4.0: 6 roots, 586 active categories).
- **Turkish**: machine translation of the identical English records by
  `Helsinki-NLP/opus-mt-tc-big-en-tr` @ `e539fc16a8a1a0ea5950eb339b595bfcce990e90`. 30,809
  imported records keep the legacy alias `Helsinki-NLP/opus-mt-en-tr` in
  `translation_requested_model`, which `src/naltra/data/translation.py` normalises to that
  checkpoint.
- **Code-switch**: aligned English and Turkish sentences of the same project, mixed
  deterministically with a per-pair seed.
- **Traceability**: every record keeps `project_id`, `pair_id = cordis:<project_id>`,
  `split`, the direct labels `labels_direct` and their hierarchy closure `labels`.
- **Integrity**: splits are assigned per project before any language derivation; projects
  with identical normalised objectives share a SHA-256 content fingerprint and therefore a
  split; translation and mixing change only the text, never the labels.

## Restore, rebuild, and train

To use the committed release, restore it, audit it record by record and train with the
same audit. The plain validator refuses the shipped release
([why](../docs/dataset.md#shipped-turkish-release-110)):

```bash
python scripts/unpack_data.py
python scripts/validate_datasets.py --prebuilt-release
python scripts/train_all.py --datasets cordis_h2020 --prebuilt-release --models naive_bayes svm --device cpu --output-dir models/cordis_v1.1.0
```

To rebuild from the committed archive instead (the archive is reused, not downloaded):

```bash
python scripts/build_taxonomy.py
python scripts/prepare_data.py --dataset cordis_h2020 --download
python scripts/validate_datasets.py --languages en
python scripts/prepare_data.py --dataset translation --device cuda
python scripts/validate_datasets.py --languages en tr
python scripts/train_all.py --datasets cordis_h2020 --languages en tr --models naive_bayes svm hybrid_knn bilstm transformer --device auto --output-dir models/cordis_rebuild
```

Name `--models` explicitly: the default in `configs/training.yaml` is only `bilstm transformer`.
A rebuild rewrites the tracked `manifest.json` files, `splits/cordis_h2020/project_splits.json`
and `review/translation_review_sample.csv`, so check `git diff` before committing. `--dataset all --download` additionally produces the
noisy test copies and both code-switch tracks; `validate_datasets.py --derived` audits them.
The installed `naltra-prepare-data` command has the same options. Rebuilt manifests use
schema 1.1.0 and verify source, output, generation-code/configuration, taxonomy, and
split-map hashes; relevant code changes require genuine regeneration. Translation caches
include checkpoint revision and implementation identity, so legacy entries are not reused
silently.

A rebuild reproduces EN from the raw archive with the committed scripts, and the derived
tracks from the EN/TR release they are built on; it does not reproduce TR 1.1.0 (Known
issues, item 8), so a rebuilt TR needs a new release version. Do not commit decompressed
`.jsonl` files, extracted CSVs, translation caches, or model weights.

## Versioning and maintenance

- Release 1.1.0 is identified by the per-split SHA-256s in the four committed manifests
  (EN, TR, both code-switch tracks). `manifest_version` is the manifest schema version
  (`src/naltra/data/manifest.py`), which any rebuild also writes.
- The 1.1.0 files are hash-locked and not changed in place: a change to text or labels, such
  as a fix for a known issue above, needs a new release version and a re-evaluation of every
  member.
- Maintainer: the Data & Taxonomy area owner in [team ownership](../docs/team_ownership.md);
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
