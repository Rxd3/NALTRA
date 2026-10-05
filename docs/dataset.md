# Dataset and Benchmark Specification

NALTRA uses **SIB-200** as its single source dataset for cross-lingual English (`en`), Turkish (`tr`), and synthetic English–Turkish (`en-tr`) topic classification and model robustness benchmarking.

---

## 1. Principles and Repository Policy

1. **Single Source Dataset**: SIB-200 (`Davlan/sib200`) is the only source dataset for the project. All benchmark tracks (clean, noisy, code-switched, and Near-OOD) derive entirely from SIB-200.
2. **Unified Evaluation**: Within each benchmark track, every evaluated model receives the exact same records, splits, label space, text representations, noise variants, and OOD folds.
3. **Zero Split Leakage**: Split assignments strictly respect official SIB-200 upstream partitions (`train`, `validation`, `test`). All partitions are audited for record ID overlap and SHA-256 normalized content-fingerprint leakage (both strictly zero).
4. **Local Data Storage Policy**: Large raw dataset caches, clean processed files, noisy variants, and OOD benchmark folds remain local and are ignored by Git (`data/raw/*`, `data/processed/*`, `data/noisy/*`, `data/ood/*`). Only code, test suites, documentation, and `.gitkeep` placeholders are tracked in version control.

---

## 2. Unified Record Schema

All clean in-distribution benchmark records share a strongly typed, common JSONL record schema validated by `validate_record()` in `src/naltra/data/preprocessing.py`:

```json
{
  "id": "sib200:en:548",
  "text": "With the change from the quarter to the half mile run, speed becomes of much less importance and endurance becomes an absolute necessity.",
  "labels": ["sport"],
  "language": "en",
  "source": "sib200",
  "license": "CC BY-SA 4.0",
  "split": "validation",
  "pair_id": "sib200:548",
  "source_id": 548,
  "source_labels": ["sports"]
}
```

### Schema Fields
- `id` (str): Unique machine-readable identifier (`sib200:{lang}:{index_id}`).
- `text` (str): Normalized UTF-8 text string (whitespace collapsed, line-breaks normalized via `preprocess_record_text`).
- `labels` (list[str]): Canonical taxonomy identifiers from `taxonomy/taxonomy.json`. Note that while the internal schema stores labels as a list (`["sport"]`), SIB-200 is strictly a single-label topic classification benchmark.
- `language` (str): Language code (`en`, `tr`, or `en-tr`).
- `source` (str): Source dataset name (`sib200`).
- `license` (str): Dataset distribution license (`CC BY-SA 4.0`).
- `split` (str): Upstream partition name (`train`, `validation`, `test`).
- Traceability metadata:
  - `pair_id` (str): Cross-lingual parallel alignment identifier (`sib200:{index_id}`).
  - `source_id` (int): Upstream row identifier (`index_id`).
  - `source_labels` (list[str]): Raw upstream category labels (e.g. `["sports"]`).

---

## 3. SIB-200 Source Dataset

### 3.1 Overview
- **Dataset**: SIB-200 (Simple, Inclusive, and Big evaluation dataset for topic classification)
- **Hugging Face ID**: `Davlan/sib200`
- **Revision**: `38977a667f6fc264d5c26ec57a01e16db040b358`
- **Paper**: Adelani et al., "SIB-200: A Simple, Inclusive, and Big Evaluation Dataset for Topic Classification in 200+ Languages and Dialects"
- **License**: CC BY-SA 4.0
- **Configurations Used**: `eng_Latn` (English) and `tur_Latn` (Turkish)

### 3.2 Record Counts and Split Distribution
SIB-200 provides official, parallel train/validation/test partitions. The dataset contains 1,004 parallel English/Turkish pairs, yielding 2,008 total records:

| Split | English (`en`) | Turkish (`tr`) | Total Records | Aligned Pairs (`pair_id`) |
| :--- | :---: | :---: | :---: | :---: |
| `train` | 701 | 701 | 1,402 | 701 |
| `validation` | 99 | 99 | 198 | 99 |
| `test` | 204 | 204 | 408 | 204 |
| **Total** | **1,004** | **1,004** | **2,008** | **1,004** |

### 3.3 Cross-Lingual Alignment (`pair_id`)
English and Turkish records sharing the same underlying sentence share an identical `pair_id` (`sib200:<index_id>`) and identical canonical topic labels. This enables matched paired evaluation across languages.

### 3.4 Canonical Taxonomy Mapping
The active taxonomy (`taxonomy/taxonomy.json`, version `0.3.0`) defines seven flat canonical topics (`parent: null`). Upstream SIB-200 categories are mapped through `taxonomy/label_map.json`:

| SIB-200 Source Category | Canonical Topic ID | Display Name | Total Records |
| :--- | :--- | :--- | :---: |
| `science/technology` | `science_technology` | Science and Technology | 504 |
| `travel` | `travel` | Travel | 396 |
| `politics` | `politics` | Politics | 292 |
| `sports` | `sport` | Sport | 244 |
| `health` | `health` | Health | 220 |
| `entertainment` | `arts_culture_entertainment_media` | Arts, Culture, Entertainment and Media | 186 |
| `geography` | `geography` | Geography | 166 |
| **Total** | | | **2,008** |

Output location: `data/processed/sib200/` (`train.jsonl`, `validation.jsonl`, `test.jsonl`).

---

## 4. Noisy Robustness Benchmark

The noisy benchmark evaluates model resilience against typographical slips, capitalization changes, punctuation mutations, and Turkish orthographic shifts.

- **Perturbation Target**: Generated exclusively for `validation` (198 records) and `test` (408 records). Clean training data is **never** perturbed.
- **Seeding and Determinism**: Uses platform-independent SHA-256 seeding (`f"{record_id}:{strategy}:{severity}:{base_seed}"`).
- **Metadata Invariance**: Canonical `labels`, `language`, `source`, `license`, `split`, and `pair_id` are preserved with 100% fidelity.
- **Default Robustness Setting**:
  - **Strategy**: `combined` (composite of character swap, duplication, deletion, casing shift, punctuation change, and Turkish diacritic mutation).
  - **Severity**: `medium` (10% perturbation probability).
  - **Validation Count**: 198 records
  - **Test Count**: 408 records
  - **Total**: 606 records
- **Storage Location**: `data/noisy/combined/medium/sib200/` (`validation.jsonl`, `test.jsonl`, `manifest.json`).

---

## 5. Synthetic EN/TR Code-Switch Benchmark

The code-switch benchmark evaluates model performance on mixed-language English-Turkish inputs.

- **Source Material**: Generated exclusively from aligned SIB-200 English and Turkish pairs sharing `pair_id`.
- **Synthetic Chunk-Mixing**:
  > [!NOTE]
  > This is a **controlled synthetic chunk-mixing robustness benchmark**. It preserves token order within source-language chunks, but does **not** claim to represent linguistically natural, word-aligned, or grammatical code-switching.
- **Language Code**: `language="en-tr"`.
- **Default Setting**:
  - **Strategy**: `chunk_mix` (contiguous phrases sampled alternately from English and Turkish parallel sentences).
  - **Strength**: `balanced` (~50/50 token mix from aligned pairs, alternating start language deterministically per pair seed).
- **Benchmark Record Counts**:
  - `validation`: 99 mixed records
  - `test`: 204 mixed records
  - `train`: 0 (clean training data remains untouched)
  - **Total**: 303 unique mixed records
- **Storage Location**: `data/processed/code_switch/chunk_mix/balanced/` (`validation.jsonl`, `test.jsonl`, `manifest.json`).
- **Record Identifier**: `f"{pair_id}:codeswitch:{strategy}:{strength}"` (e.g. `sib200:548:codeswitch:chunk_mix:balanced`).

---

## 6. SIB-200 Near-OOD Benchmark (Leave-One-Topic-Out, 7 Folds)

Out-of-Distribution evaluation determines whether models can detect inputs that do not belong to the in-distribution topic set. NALTRA implements a **purely SIB-200-derived Near-OOD benchmark** using leave-one-topic-out cross-validation across all seven canonical topics. No external OOD dataset is used.

### 6.1 Dedicated OOD Record Schema
OOD evaluation records use a dedicated schema validated by `validate_ood_record()` in `src/naltra/data/ood.py`:

```json
{
  "id": "sib200:ood:politics:validation:sib200:en:546",
  "text": "General James Ewing would take 700 militia across the river at Trenton Ferry...",
  "language": "en",
  "source": "sib200",
  "license": "CC BY-SA 4.0",
  "split": "validation",
  "is_ood": true,
  "ood_type": "near_ood",
  "ood_source": "sib200_heldout",
  "ood_reason": "heldout_topic:politics",
  "source_id": 546,
  "source_label": "politics",
  "pair_id": "sib200:546"
}
```

> [!IMPORTANT]
> **No Fake Canonical Labels**: OOD records are **never** assigned an in-distribution canonical label from the taxonomy, nor is an artificial "OOD" taxonomy class created. The `source_label` field preserves original topic ground truth strictly for diagnostic logging and AUROC evaluation.

### 6.2 Leave-One-Topic-Out Folds
In each fold $k \in \{1..7\}$, one SIB-200 canonical topic is held out:
- **In-Distribution Training (`train_id.jsonl`)**: Contains all training records belonging to the other 6 topics.
- **In-Distribution Evaluation (`validation_id.jsonl`, `test_id.jsonl`)**: Validation and test records belonging to the other 6 topics.
- **Near-OOD Evaluation (`validation_ood.jsonl`, `test_ood.jsonl`)**: Validation and test records belonging exclusively to the held-out topic.
- **Storage Location**: `data/ood/near/sib200/<heldout_topic>/` (each fold directory contains `train_id.jsonl`, `validation_id.jsonl`, `test_id.jsonl`, `validation_ood.jsonl`, `test_ood.jsonl`, and `manifest.json`).

### 6.3 Verified Fold Counts

| Held-Out Topic | Train ID Recs | Val OOD Recs (EN / TR) | Test OOD Recs (EN / TR) | Total Near-OOD Recs |
| :--- | :---: | :---: | :---: | :---: |
| `arts_culture_entertainment_media` | 1,272 | 18 (9 / 9) | 38 (19 / 19) | **56** |
| `geography` | 1,286 | 16 (8 / 8) | 34 (17 / 17) | **50** |
| `health` | 1,248 | 22 (11 / 11) | 44 (22 / 22) | **66** |
| `politics` | 1,198 | 28 (14 / 14) | 60 (30 / 30) | **88** |
| `science_technology` | 1,050 | 50 (25 / 25) | 102 (51 / 51) | **152** |
| `sport` | 1,232 | 24 (12 / 12) | 50 (25 / 25) | **74** |
| `travel` | 1,126 | 40 (20 / 20) | 80 (40 / 40) | **120** |

---

## 7. Reproduction and Audit Commands

All clean, perturbed, code-switched, and Near-OOD benchmarks can be reproduced deterministically with the following commands:

```bash
# 1. Clean in-distribution SIB-200 dataset
python scripts/prepare_data.py --dataset sib200

# 2. Noisy robustness benchmark (combined/medium)
python scripts/prepare_data.py --dataset noisy

# 3. Synthetic EN/TR code-switched benchmark (chunk_mix/balanced)
python scripts/prepare_data.py --dataset code_switch

# 4. Near-OOD SIB-200 leave-one-topic-out folds (7 folds)
python scripts/prepare_data.py --dataset ood

# Alternatively, prepare all benchmark tracks in one pass:
python scripts/prepare_data.py --dataset all

# 5. Run full dataset-wide audit and verification
python scripts/validate_datasets.py
```
