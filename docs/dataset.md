# Dataset

NALTRA uses one source: **CORDIS EU research projects under Horizon 2020
(2014–2020)**. English project objectives are the original text; Turkish and
code-switch tracks are derived variants. No external parallel dataset is added.

The prepared data form **the NALTRA CORDIS EN-TR benchmark, release 1.1.0**. Its
[dataset card](../data/README.md) summarises the tracks and gives intended use,
known issues, and citation.

## Source and provenance

The source is published by the European Commission / Publications Office.
The [dataset landing page](https://data.europa.eu/data/datasets/cordish2020projects)
identifies the distribution. Attribution and changes are recorded in
[data/LICENSE](../data/LICENSE).

| Item | Recorded value |
| --- | --- |
| Source archive | `data/raw/cordis_h2020/cordis-h2020projects-csv.zip` |
| Distribution URL | `https://cordis.europa.eu/data/cordis-h2020projects-csv.zip` |
| Recorded retrieval | October 2026 |
| Format | UTF-8, semicolon-delimited CSV |
| Archive size | 55,219,250 bytes |
| Source projects | 35,389 |
| EuroSciVoc assignment rows | 111,829 |
| Projects with assignments | 32,210 |
| Distinct directly assigned category codes | 1,053 |
| Retained projects | 31,407 |

Archive SHA-256:
`d4a8e3645d0b1896b967618e8e7a7f7928738b469737363a30816c217cbd48f2`.

The archive contains `project.csv` and `euroSciVoc.csv`. Their sizes and checksums
are locked in `configs/data.yaml`; dataset manifests also record source identity,
split identity, taxonomy hashes, generation settings, and code/configuration hashes.

## Labels and splits

The active taxonomy has 473 supported direct labels and 586 nodes including
ancestors. Support means at least 50 distinct source projects per directly assigned
category. The filter is computed **before splitting, over the full archive**,
including projects later assigned to test. See [taxonomy](taxonomy.md).
Codes below the threshold are dropped from `labels_direct` and `source_label_ids`,
not rolled up to their nearest supported ancestor (`src/naltra/data/preprocessing.py`);
[known issue 2](../data/README.md#known-issues) counts the lost codes and their
effect on hierarchical scores.

| Split | Projects per language | Share |
| --- | --- | --- |
| Train | 21,985 | 70% |
| Validation | 4,711 | 15% |
| Test | 4,711 | 15% |

Splitting uses seed 42 and groups identical normalized objectives by SHA-256
fingerprint before multi-label stratification. A project and its language variants
remain in the same split. This prevents exact project/text leakage; similar texts
that are not identical can still cross splits.

Of the retained projects, 84.3% have multiple direct labels and 59.8% span multiple
roots. Mean direct-label cardinality is 3.21; mean ancestor-expanded cardinality
is 10.24. Models train on direct labels, while hierarchical evaluation expands
ancestors. Record counts and objective lengths per track are in the
[dataset card](../data/README.md#at-a-glance).

## Record schema

Each UTF-8 JSONL line is one project variant. Key fields are:

| Field | Meaning |
| --- | --- |
| `id` | Variant-specific identifier, such as `cordis:641585:en` |
| `project_id`, `source_id` | CORDIS project identity |
| `pair_id` | Shared identity, such as `cordis:641585` |
| `title`, `text` | Project title and objective; classification uses `text` |
| `labels_direct` | Direct canonical label IDs that pass the support threshold |
| `labels` | Direct labels plus their ancestors |
| `language`, `split` | Variant language and source project partition |
| `source`, `license`, `taxonomy_version` | Source and taxonomy identity |
| `synthetic_language_variant` | Whether the text is a derived language variant |

English records also carry `content_fingerprint`, `human_validated`,
`source_label_ids`, and `source_labels`. The CORDIS `human_validated` value is
a string (`false`, `true`, or `NA`), rather than a translation-quality flag.

Turkish records retain the English title, project/pair IDs, split, and label sets.
They omit the English-specific fingerprint/annotation fields and add:

- `variant_of` and `english_source_text`;
- `translation_backend`, `translation_model`, and `translation_model_revision`;
- `translation_source_hash` and indexed EN/TR `sentence_alignment` pairs;
- import/repair metadata where applicable.

The record validator is `validate_record()` in
`src/naltra/data/preprocessing.py`. Every direct label must occur in `labels`,
and every node in `labels` must include its ancestors.

## Preparation

1. Verify and extract the locked archive.
2. Parse project objectives and EuroSciVoc assignments, repairing rows with
   unescaped semicolons and dropping invalid/incomplete records.
3. Strip HTML, unescape entities, collapse whitespace, and preserve casing.
4. Map supported labels and expand ancestors; drop projects without a retained label.
5. Group duplicate objectives, stratify projects, and write splits/manifests.
6. Translate English while preserving project identity, labels, and split.
7. Build synthetic mixed/noisy variants and audit the release.

Commands for restoration and regeneration are in the [runbook](runbook.md#0-data).

### Known limitation: angle-bracket comparisons

The archived cleaner uses `<[^>]+>` to remove HTML. It also removes scientific
comparison spans: `after (z<3) and before (z>3)` becomes `after (z 3)`.
Across 62 retained projects (48 train, 5 validation, 9 test), 17,290 characters
were lost. Labels are unchanged.

Dropping the nine affected test projects changes full-set EN micro-F1 by at most
0.0002 per member; the effect on training is unmeasured. Changing the cleaner
requires a new dataset release and evaluation, because the current text is
checksum-locked.

## Translation

The checkpoint is `Helsinki-NLP/opus-mt-tc-big-en-tr`, revision
`e539fc16a8a1a0ea5950eb339b595bfcce990e90`. Translation is local, cached, and
sentence-aligned. QA checks empty/untranslated text, control characters, and
length-ratio outliers. Automatic QA is not semantic validation, and the pipeline QA
itself does not check the output language; see [translation screens](#translation-screens)
and [translation quality](#translation-quality).

The standard translator uses chunks up to 500 source tokens. Cache identity
includes checkpoint and implementation identity. Reusing translations requires
matching English text, labels, taxonomy, split, and model provenance. Aligned
chunks must preserve source and translated text in order.

### Shipped Turkish release (1.1.0)

All four manifests (EN, TR, and two code-switch tracks) identify release 1.1.0.
English split bytes are unchanged from 1.0.0.

TR 1.1.0 repairs 598 records flagged by the 1.0.0 alignment audit, using 128-token
source chunks. The repaired records carry `import_archive_sha256`: 416 train,
99 validation, and 83 test. All 598 alignments were rewritten; 300 translated texts
changed (214 train, 49 validation, 37 test). The 300 is a historical figure: the
1.0.0 import is not shipped, so it cannot be recomputed. Labels stayed identical.
The remaining 30,809 imported records retain the legacy requested-model alias
`Helsinki-NLP/opus-mt-en-tr`, normalized to the pinned checkpoint by the code.

The `environment` block of all four manifests records the import and repair software:
Python 3.13.5, PyTorch `2.6.0+cu124`, and Transformers `5.18.0`. The 1.0.0 bulk
run's software is not recorded; `requirements.txt` pins the later evaluation
environment instead.

This is a frozen imported release. The targeted repair implementation is pending
handoff, so a standard translation run produces a different TR release.
See [training/data provenance](reproducibility.md#training-and-data-provenance).

NB, SVM, and hybrid kNN were trained on this release. BiLSTM/Transformer
`cordis_v0.5.0` heads were refit on it using frozen `cordis_v0.4.0` encoders.
Original XLM-R encoder fine-tuning data is not recorded. Zero-shot Kev, Jev, and
Laya were evaluated on this release using the fixed 80/120-pair samples.

### Release validation

The archived manifests lock generation code from commit `ea00e6b`. Current data
code differs, and extracted raw CSVs initially exist only inside the ZIP. The
full validator therefore refuses the archived release.

Use:

```bash
python scripts/unpack_data.py
python scripts/validate_datasets.py --prebuilt-release
```

This checks declared split checksums/counts and every EN/TR record: identity,
hierarchy, fingerprints, leakage, and alignment. It reports alignment-text mismatches
and QA failures rather than enforcing them; both are zero on 1.1.0. It tolerates
15,434 segmenter-drift records, which training records in provenance. Of the repaired
records, 204 rely on the accepted segmentation normalization of bullet/list marks.

The prebuilt audit does not audit code-switch tracks. Combining
`--prebuilt-release` with `--code-switch` or `--derived` is a usage error.
Decompression checks their manifest-declared bytes; the code audit separately
confirmed their contents match current generation when aligned by ID.

Freshly regenerated releases use the full validator instead. Do not edit archived
code hashes to bypass a failed audit.

### Translation screens

The length ratio is checked per document, so a truncated sentence inside a long
document passes QA. Two screens of the shipped TR splits flag candidates, not
confirmed errors (`python scripts/extra_analyses.py translation`, which refuses a
split whose SHA-256 differs from its manifest; `results/metrics/translation_checks.json`):

| Screen | A TR record is flagged when | Flagged |
| --- | --- | --- |
| Possible truncation | a `sentence_alignment` pair has an English sentence longer than 80 characters and a Turkish sentence with fewer than 40% as many characters | 180 of 31,407 (0.57%): 117 train, 32 validation, 31 test |
| Possible repetition | its lowercased `\w+` words hold an n-gram (n ≥ 4) immediately followed by itself, and that doubled sequence does not occur in the paired English text | 48 (0.15%): 37 train, 5 validation, 6 test |

A one-off screen ([snippet](../data/README.md#at-a-glance)) finds Turkish-specific
letters (`çğıöşüÇĞİÖŞÜ`) in all 31,407 TR records, U+FFFD in none, and no record
equal to its English source. The released text is hash-locked and unchanged; the
effect of flagged records on scores is not measured.

### Translation quality

No human has reviewed the Turkish text (the human-review columns in `data/review/` are
still `pending`). Beyond the structural QA and screens above, the evidence is two
automatic proxies (`python scripts/extra_analyses.py mtquality`, run only when named,
with `--device` `auto`, `cuda` or `cpu` and `--batch-size`; it reads only manifest-locked
splits; `results/metrics/translation_quality.json`):

| Check | Records | Result |
| --- | --- | --- |
| Language ID: the repository's `LinguaLanguageDetector` (EN and TR only, minimum confidence 0.6) | every record of every split | TR: 31,400 of 31,407 (99.98%) Turkish, 7 mixed `en-tr` (3 train, 1 validation, 3 test), none English or undetermined; EN: all 31,407 English |
| Round-trip chrF: each aligned TR sentence translated back to English with `Helsinki-NLP/opus-mt-tc-big-tr-en` @ `2261c8fc` (fp16, 4 beams), then document chrF (character 1- to 6-grams, beta 2) against the English source | 300 TR test records drawn with seed 42, plus every test record a screen flags (334 scored) | sample mean 77.1, median 78.5, interquartile range 75.4-80.9, range 40.6-86.6; none below 40 |

The 31 test records flagged as possible truncations average 53.5 and the 6 possible
repetitions 77.1 (57.3 for all 37 flagged), against 77.4 for the 297 unflagged sampled
records. The lowest sampled records are `cordis:730846:tr` (40.6), `cordis:744457:tr`
(42.8), `cordis:687757:tr` (46.9), `cordis:641615:tr` (48.4, also one of the 7 mixed
records) and `cordis:768583:tr` (52.7). Both checks are proxies, not human judgements:
language ID shows that a text reads as Turkish, not that it is faithful; chrF measures the
back-translation's character overlap with the English source, so errors of the two MT
models can cancel out or be added, and Turkish fluency is not measured.

### Translation review

Human review remains pending. The 100-row review sample and AI-assisted review
predate 1.1.0. Eighty-two sample translations match the current release; 18 refer
to repaired projects and match neither archived release. These are multi-label
validation projects at depths 4–7. The `top_level_domain` field stores the first
direct label, rather than its root. The AI-assisted review does not record its
model, prompt, or date.

Historical 1.0.0 translation logs report 234.52 minutes on an RTX 3060, 12 GB,
for 31,407 projects and 280,039 sentences, with peak allocation logged as 2,837.2 MB.
These timing/allocation figures are not stored in the release manifests and their
logging implementation is absent.

## Derived evaluation tracks

| Track | Construction | Splits |
| --- | --- | --- |
| Sentence mix | Alternating aligned EN/TR sentences from one project | Validation and test |
| Chunk mix | First/second halves of aligned EN/TR sentences | Validation and test |
| Noisy EN/TR | Character swap/duplicate/delete, casing, and punctuation noise; combined, medium, seed 42 | Test; generated locally, not shipped |
| Fixed samples | Projects with smallest pair-ID SHA-256 values, shared by EN/TR | 80 validation and 120 test pairs; generated locally |

Each code-switch split contains 4,711 projects. Derived tracks preserve pair IDs,
splits, and labels. They are synthetic evaluation inputs; models are not trained
on them. Code-switching is inter-sentential: only the 11 single-sentence documents
of each track's 9,422 switch inside a sentence, by joining word halves.

### Noisy test copies

`python scripts/prepare_data.py --dataset noisy --prebuilt-release` writes the copies
to `data/noisy/combined/medium/cordis_h2020/{en,tr}/test.jsonl`
(`generate_noisy_benchmarks` in `src/naltra/data/noise.py`). Each record's seed is a
SHA-256 of its ID, strategy, severity, and base seed 42, so regeneration is exact.
`id` gains the suffix `:noise:combined:medium`, and `original_id` names the clean record.

Severity medium is 0.10. Each document gets one primary edit type: an adjacent-character
swap per space-separated word (35% of documents), or duplication (35%) or deletion
(30%) per alphanumeric character, each applied with probability 0.10. With probability
0.10 each, casing then flips on 5% of letters and punctuation swaps on 10% of
`. , ! ? ; :`. At least one edit is guaranteed. Among documents whose space-separated
token count is unchanged (4,707 EN, 4,708 TR), a mean of 32.5% (EN) and 37.3% (TR) of
tokens change. The rate is bimodal: 31.4% (EN) and 31.2% (TR) of these documents
have under 15% of tokens changed, and 50.9% and 64.8% have over 40%. `noise.py`
also implements a `turkish_diacritics` strategy that no command produces.

## Limitations

CORDIS labels are silver labels; CORDIS does not document how they are assigned
([known issue 1](../data/README.md#known-issues)). Translation quality has only
automatic checks, no human review, and synthetic mixed/noisy text does not represent
all real bilingual writing. Exact-text grouping does not remove near-duplicates. The
support filter uses the complete source archive, rather than training-only counts.

Every known issue of release 1.1.0 is listed once in the
[dataset card](../data/README.md#known-issues). See
[results limitations](results.md#limitations) for model-selection reuse,
training budgets, sample coverage, and OOD scope.
