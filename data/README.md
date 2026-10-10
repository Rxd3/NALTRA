# Data Workspace

This directory holds the CORDIS Horizon 2020 source data and the derived multilingual
benchmark tracks used by NALTRA. The dataset files are CC BY 4.0; see [LICENSE](LICENSE) for
the attribution and the changes made. The preprocessing steps are documented in the
[README](../README.md#preprocessing-pipeline) and the [dataset card](../docs/dataset.md).

## What Git tracks

The raw archive and the compressed processed splits are tracked, so no download is needed.
Everything that can be regenerated from them stays ignored through `.gitignore`.

| Path | Tracked | Contents |
| --- | --- | --- |
| `raw/cordis_h2020/cordis-h2020projects-csv.zip` | yes | Unmodified official archive, SHA-256 locked in `configs/data.yaml` |
| `processed/cordis_h2020/**/*.jsonl.gz`, `manifest.json` | yes | Cleaned splits; each manifest lists SHA-256, size and record count per split |
| `review/` | yes | Translation review sample and its AI-assisted audit, both drawn before TR 1.1.0 (see below) |
| `splits/cordis_h2020/project_splits.json` | yes | Project-level split map |
| `processed/**/*.jsonl` | no | Restored by `python scripts/unpack_data.py` (or `make data`) |
| `raw/cordis_h2020/*.csv` | no | Extracted by `scripts/prepare_data.py --download` |
| `cache/`, `noisy/`, `samples/` | no | Translation cache, noisy test copies, Kev, Jev and Laya evaluation samples |

`scripts/unpack_data.py` decompresses each `.jsonl.gz` next to itself and replaces the
`.jsonl` only after its SHA-256 and size match the manifest; files that already match are
skipped, and any mismatch fails without leaving a partial file.

## Directory Layout

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

## Source Dataset

NALTRA uses **one source dataset**:

- **Title**: CORDIS - EU research projects under Horizon 2020 (2014-2020)
- **Publisher**: European Commission / Publications Office of the European Union
- **Canonical URL**: `https://cordis.europa.eu/data/cordis-h2020projects-csv.zip`
- **License**: CC BY 4.0, attribution "European Union, CORDIS - https://cordis.europa.eu - CC BY 4.0"
- **Primary Text Field**: `objective` (project summary text)
- **Ontology**: European Science Vocabulary (EuroSciVoc), 6 root fields, 586 active categories (v0.4.0)

## Derived Multilingual Variants

1. **English (`en`)**: Cleaned CORDIS project objectives.
2. **Turkish (`tr`)**: machine translation of the identical English records by
   `Helsinki-NLP/opus-mt-tc-big-en-tr` @ `e539fc16a8a1a0ea5950eb339b595bfcce990e90`. The
   shipped release is TR 1.1.0: the 1.0.0 import with the 598 records its audit flagged
   re-translated in 128-token chunks. 30,809 imported records keep the legacy alias
   `Helsinki-NLP/opus-mt-en-tr` in `translation_requested_model`, which
   `src/naltra/data/translation.py` normalises to that checkpoint.
3. **Code-Switching (`en-tr`)**: Aligned English and Turkish sentences of the same project, mixed deterministically.

All derived records preserve:
- Source project ID (`project_id`) and pair ID (`pair_id = cordis:<project_id>`)
- Split assignment (`train`, `validation`, `test`)
- Direct EuroSciVoc labels (`labels_direct`) and hierarchy closure (`labels`)

## Data Integrity Guarantees

- **Zero Project Leakage**: Splits are assigned per source project *before* any language derivation.
- **Zero Content Leakage**: Projects with identical normalized objectives share a SHA-256 content fingerprint and therefore a split.
- **Zero Label Mutation**: Translation and mixing change only the text; labels are copied from the English record.

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
tracks from the EN/TR release they are built on; it does not reproduce TR 1.1.0. That is a
frozen imported release: its targeted repair of 598 records in 128-token source chunks
(recorded in `processed/cordis_h2020/tr/manifest.json` `generation_parameters`) was run by
a teammate, and the repair script is not in the repository. The standard translation step
above therefore builds a different TR release, which needs a new release version.

Do not commit decompressed `.jsonl` files, extracted CSVs, translation caches, or model
weights. Human translation review remains pending until reviewers complete the sample:
the sample's `review_status`, `meaning_preserved`, `terminology_quality` and `fluency`
columns read `pending` in all 100 rows, and `notes` says the review is pending.

The review sample (`review/translation_review_sample.csv`) and its AI-assisted audit
(`review/translation_ai_review.csv`) predate TR 1.1.0. 82 of the 100 rows match the shipped
Turkish text exactly (unchanged since 1.0.0). The other 18 are among the 598 records repaired
in 1.1.0, and their sample text matches neither the 1.0.0 nor the 1.1.0 text. All 100 are
multi-label validation projects at hierarchy depths 4–7, and the `top_level_domain` column
holds each project's first direct label rather than its root; the
[dataset card](../docs/dataset.md#8-translation-pipeline--turkish-derivation) has the details.
