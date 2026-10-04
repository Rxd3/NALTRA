# Dataset and Benchmark Specification

NALTRA maintains reproducible, cross-lingual English (`en`), Turkish (`tr`), and synthetic English-Turkish (`en-tr`) benchmark tracks for hierarchical topic classification and model robustness.

---

## 1. Fairness and Evaluation Principles

1. **Unified Evaluation**: Within each benchmark track, every evaluated model receives the exact same records, splits, label spaces, text representations, noise variants, and OOD sets.
2. **Leakage Policy**: Evaluation tracks are audited for record ID and normalized-content overlap across all splits. MN-DS content groups stay in one partition; MultiFin evaluation is filtered without changing official training. The official MultiFin track and its noisy copies retain upstream overlap for reference comparisons and must not be used to claim independent final evaluation.
3. **Repository Storage Policy**: Large raw archives, clean processed datasets, noisy variants, and OOD sets remain local and are ignored by Git (`data/raw/*`, `data/processed/*`, `data/noisy/*`, `data/ood/*`). Only code, test suites, documentation, and `.gitkeep` placeholders are tracked in the repository.

---

## 2. Unified Schema

All in-distribution benchmark records share a consistent, strongly typed schema validated by `validate_record()` in `src/naltra/data/preprocessing.py`:

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
- `id` (str): Unique machine-readable identifier.
- `text` (str): Normalized UTF-8 text string (whitespace collapsed, line-breaks normalized).
- `labels` (list[str]): Canonical taxonomy identifiers from `taxonomy/taxonomy.json`.
- `language` (str): Language code (`en`, `tr`, or `en-tr`).
- `source` (str): Source dataset name (`sib200`, `multifin`, `mn_ds`).
- `license` (str): Dataset distribution license.
- `split` (str): Partition name (`train`, `validation`, `test`).
- Traceability metadata: `pair_id` (cross-lingual alignment), `source_id` (original dataset key), `source_labels` (raw source labels).

---

## 3. Core Benchmark Datasets

### 3.1 SIB-200 (Multilingual Topic Benchmark)

SIB-200 provides parallel, cross-lingual sentence-level topic classification across English and Turkish.

- **Source**: `Davlan/sib200`
- **Paper**: Adelani et al., "SIB-200: A Simple, Inclusive, and Big Evaluation Dataset for Topic Classification in 200+ Languages and Dialects"
- **License**: CC BY-SA 4.0
- **Configurations Used**: `eng_Latn` (English) and `tur_Latn` (Turkish)
- **Pair Alignment**: 1,004 perfectly aligned English/Turkish pairs (2,008 total records) with matching `pair_id` and category labels across all splits.
- **Split Distribution**:
  - `train`: 701 EN / 701 TR (1,402 records)
  - `validation`: 99 EN / 99 TR (198 records)
  - `test`: 204 EN / 204 TR (408 records)
  - **Total**: 1,004 EN / 1,004 TR (2,008 records)
- **Topic Classes (7 canonical labels)**: `science_technology`, `travel`, `politics`, `sport`, `health`, `entertainment` (`arts_culture_entertainment_media`), `geography`.

### 3.2 MultiFin (Multilingual Financial Multi-Label Benchmark)

MultiFin provides real-world financial headline multi-label classification across English and Turkish.

- **Source**: `awinml/MultiFin`
- **Paper**: Jørgensen et al., "MultiFin: A Dataset for Multilingual Financial NLP" (Findings of EACL 2023)
- **License**: CC BY-NC 4.0
- **Configuration Used**: `all_languages_lowlevel`
- **Filtered Subset**: English (`en`) and Turkish (`tr`)

#### Tracks: Official vs. Leakage-Free Evaluation

1. **Official Track** (`data/processed/multifin/`):
   - Preserves official upstream train/validation/test partitions for baseline comparison.
   - `train`: 1,747 EN / 1,436 TR (3,183 records)
   - `validation`: 437 EN / 359 TR (796 records)
   - `test`: 546 EN / 449 TR (995 records)
   - **Total**: 2,730 EN / 2,244 TR (4,974 records)
   - **Multi-Label Preservation**: 1,591 records (31.99%) have $>1$ distinct canonical label.
   - **Documented Content Leakage**: Automated SHA-256 normalized content fingerprint analysis reveals that **120 validation records** and **157 test records** duplicate normalized texts present in the training set. Additionally, **45 content fingerprints** overlap between validation and test. Zero content leakage is **not** claimed for this official track.

2. **Leakage-Free Evaluation Track** (`data/splits/multifin/leakage_free/`):
   - Preserves the official training baseline while removing evaluation records contaminated by training content.
   - Uses the official training partition as the reference training set (3,183 records).
   - Excludes validation and test records whose normalized content fingerprint appears in training.
   - Preserves the retained test set and excludes validation records whose normalized content fingerprint appears in that test set. This policy is applied before tuning or model selection.
   - Preserves full traceability to original records and splits (`original_split`, original `id`).
   - `train`: 3,183 records (reference)
   - `validation`: 651 records (120 training leaks and 25 records covering 24 test-overlap fingerprints excluded)
   - `test`: 838 records (157 leaking records excluded)
   - **Three-Way Content Disjointness**: Train/validation, train/test and validation/test normalized-content overlaps are all **0**. Distinct source records with repeated content inside one split are preserved.
   - Abdullah and the technical lead should review and use this documented policy consistently before reporting benchmark results. Existing models tuned on the earlier overlapping validation set need fresh selection on this track.

### 3.3 MN-DS (Hierarchical News Benchmark)

MN-DS provides broad-coverage, fine-grained hierarchical English news classification.

- **Source**: Zenodo record 7394851 (`MN-DS-news-classification.csv`)
- **License**: CC BY 4.0
- **Dataset Composition**: 10,917 raw annotation rows merged into 10,491 unique article records.
- **Text Representation**:
  - Headline and body are combined according to standard sentence punctuation and spacing:
    - If headline ends in punctuation (`.`, `!`, `?`): `f"{title} {content}"`.
    - Otherwise: `f"{title}. {content}"`.
    - If only headline or only body is present, that non-empty string is used directly.
  - The combined text is then normalized via `preprocess_record_text()` (whitespace collapsed, line-breaks normalized).
- **Label Hierarchy**: 109 fine-grained level-2 categories mapped into 17 level-1 root categories.
- **Multi-Label Statistics**: 392 articles (3.74%) have $>1$ fine-grained label; 137 articles span multiple root categories.
- **Content-Grouped Stratification**:
  - Articles sharing the same normalized content fingerprint (SHA-256) are grouped as an atomic splitting unit *before* partitioning (`grouped_multilabel_stratified_split`, `seed=42`).
  - Completely prevents duplicate content with different source IDs from crossing split boundaries.
  - Preserves all 10,491 source records and maintains 100% (109/109) category coverage across all three splits:
    - `train`: 7,344 records (covers 109/109 categories)
    - `validation`: 1,574 records (covers 109/109 categories)
    - `test`: 1,573 records (covers 109/109 categories)
    - **Total**: 10,491 unique articles
  - **Zero ID Leakage and Zero Content Leakage**: Both ID overlap and content-fingerprint overlap across splits are strictly **0**.

---

## 4. Noisy Robustness Benchmark

The noisy benchmark evaluates model resilience to typographical slips, casing changes, punctuation mutations, and language-specific orthography shifts.

- **Target Splits**: Generated only for `validation` and `test` splits by default. Clean training data is **never** perturbed.
- **Seeding and Determinism**: Uses platform-independent SHA-256 seeding (`f"{record_id}:{strategy}:{severity}:{base_seed}"`). Python's non-deterministic built-in `hash()` is never used.
- **Metadata Invariance**: Canonical `labels`, `language`, `source`, `license`, `split`, and `pair_id` are preserved with 100% fidelity using `copy.deepcopy()`.
- **Minimum-Edit Guarantee**: Low-probability edge cases on short texts employ a deterministic fallback to guarantee at least one valid edit without returning empty text.
- **Storage Location**: `data/noisy/{strategy}/{severity}/{dataset}/{split}.jsonl` (ignored by Git).
- **Default Robustness Setting**:
  - **Strategy**: `combined` (realistic composite of 35% character swap, 35% duplicate, 30% deletion, casing variation, and punctuation substitution).
  - **Severity**: `medium` (10% perturbation probability).
  - **Total Generated**: 5,544 records (SIB-200: 606; MultiFin: 1,791; MN-DS: 3,147).

---

## 5. Synthetic EN/TR Code-Switch Benchmark

The code-switch benchmark evaluates model performance on mixed-language English-Turkish inputs.

- **Source Material**: Generated exclusively from aligned SIB-200 English and Turkish pairs matching on `pair_id`.
- **Synthetic Chunk-Mixing**:
  > [!NOTE]
  > This is explicitly a **controlled synthetic chunk-mixing robustness benchmark**. It preserves token order within source-language chunks, but does **not** claim to represent linguistically natural, word-aligned, or grammatical code-switching.
- **Language Code**: `language="en-tr"`.
- **Default Setting**:
  - **Strategy**: `chunk_mix` (deterministic contiguous spans from English and Turkish).
  - **Strength**: `balanced` (~50/50 token mix from aligned pairs, alternating primary language deterministically per pair seed to avoid systematic English dominance).
- **Benchmark Record Counts**:
  - `validation`: 99 mixed records
  - `test`: 204 mixed records
  - **Total**: 303 mixed records (Train split remains 0 / untouched).
- **Storage Location**: `data/processed/code_switch/{strategy}/{strength}/{split}.jsonl` (ignored by Git).
- **Record Identifier**: `f"{pair_id}:codeswitch:{strategy}:{strength}"` (e.g. `sib200:548:codeswitch:chunk_mix:balanced`).

---

## 6. Out-of-Distribution (OOD) Benchmark

OOD evaluation evaluates whether models can detect inputs that do **not** belong to the in-distribution label space. NALTRA provides two unmerged OOD regimes.

### 6.1 Dedicated OOD Record Schema

OOD records use a dedicated schema validated by `validate_ood_record()` in `src/naltra/data/ood.py`:

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
> **No Fake Canonical Labels**: OOD records are **never** assigned an in-distribution canonical label from the 151 taxonomy classes, nor is an artificial "OOD" taxonomy class created. The `source_label` field preserves original dataset ground truth strictly for diagnostic logging and AUROC evaluation.

### 6.2 Near-OOD: SIB-200 Leave-One-Topic-Out (7 Folds)

Evaluates semantic topic boundary detection between news/article categories. In each fold $k \in \{1..7\}$, one SIB-200 topic is completely excluded from training:

- **ID Training (`train_id.jsonl`)**: Contains only records from the remaining 6 topics.
- **ID Evaluation (`validation_id.jsonl`, `test_id.jsonl`)**: In-distribution validation/test records from the remaining 6 topics.
- **Near-OOD Evaluation (`validation_ood.jsonl`, `test_ood.jsonl`)**: Validation and test records belonging exclusively to the held-out topic.
- **Leakage Prevention**: Verified 0 occurrences of held-out topic in `train_id.jsonl`, and zero pair ID overlap between ID training and OOD evaluation.
- **Storage Location**: `data/ood/near/sib200/<heldout_topic>/` (ignored by Git).

#### Held-Out Evaluation Counts per Fold

| Held-out Topic | Train ID Recs | Val OOD Recs (EN / TR) | Test OOD Recs (EN / TR) | Total Near-OOD Recs |
| :--- | :---: | :---: | :---: | :---: |
| `arts_culture_entertainment_media` | 1,272 | 18 (9 / 9) | 38 (19 / 19) | **56** |
| `geography` | 1,286 | 16 (8 / 8) | 34 (17 / 17) | **50** |
| `health` | 1,248 | 22 (11 / 11) | 44 (22 / 22) | **66** |
| `politics` | 1,198 | 28 (14 / 14) | 60 (30 / 30) | **88** |
| `science_technology` | 1,050 | 50 (25 / 25) | 102 (51 / 51) | **152** |
| `sport` | 1,232 | 24 (12 / 12) | 50 (25 / 25) | **74** |
| `travel` | 1,126 | 40 (20 / 20) | 80 (40 / 40) | **120** |

### 6.3 Far-OOD: Amazon MASSIVE (English & Turkish)

Evaluates cross-domain shift using task-oriented voice-assistant commands.

- **Source**: Amazon MASSIVE Dataset v1.1 (`AmazonScience/massive`)
- **License**: CC BY 4.0
- **Locales**: `en-US` and `tr-TR`
- **Matched Pairing**: Every English utterance is matched with its exact parallel Turkish translation sharing the same source `id` (`pair_id: massive:<id>`).
- **Validation/Test Independence**: Sampling preserves the seeded test selection and excludes dev pairs sharing normalized English or Turkish text with that test selection before selecting validation pairs. Pair alignment and requested counts are preserved.
- **Explicit Scenario Allowlist**: Excludes all topics with ambiguous overlap with news/article topics (e.g., weather, news, music, games, QA, transportation, recommendations). Only unambiguous smart-home and assistant commands are allowed:
  - `alarm` (set, query, remove)
  - `datetime` (query, convert)
  - `calendar` (set, query, remove)
  - `lists` (create, query, remove)
  - `iot` (smart lighting, appliances, coffee maker)
  - `audio` (volume control, mute)
  - `takeaway` (food ordering)
- **Balanced Benchmark Counts**:
  - `validation`: 250 matched pairs = **500 records** (250 EN, 250 TR)
  - `test`: 500 matched pairs = **1,000 records** (500 EN, 500 TR)
  - **Total**: 750 matched pairs = **1,500 records** (750 EN, 750 TR)
- **Storage Location**: `data/ood/far/massive/` (ignored by Git).

---

## 7. Reproduction Commands

To reproduce all clean, perturbed, code-switched, and out-of-distribution benchmark datasets locally:

```bash
# 1. Clean in-distribution datasets
python scripts/prepare_data.py --dataset sib200
python scripts/prepare_data.py --dataset multifin
python scripts/prepare_data.py --dataset mn_ds

# 2. Noisy robustness benchmark (combined/medium)
python scripts/prepare_data.py --dataset noisy

# 3. Synthetic EN/TR code-switched benchmark (chunk_mix/balanced)
python scripts/prepare_data.py --dataset code_switch

# 4. Out-of-Distribution benchmarks (Near-OOD 7 folds + Far-OOD MASSIVE)
python scripts/prepare_data.py --dataset ood

# 5. Run full dataset-wide audit and verification
python scripts/validate_datasets.py
```

## 8. Release Provenance and Verification

`configs/data.yaml` locks the immutable revisions of [SIB-200](https://huggingface.co/datasets/Davlan/sib200) and [MultiFin](https://huggingface.co/datasets/awinml/MultiFin), the official [MN-DS checksum](https://zenodo.org/records/7394851), and SHA-256 hashes of the official MASSIVE v1.1 English/Turkish exports. Preparation verifies local MN-DS and MASSIVE sources against these locks.

Every clean dataset and derived benchmark writes a `manifest.json` (schema `1.1.0`). It records generation parameters and seeds, UTC generation time, upstream metadata, input-file SHA-256 hashes, output hashes/sizes/counts, taxonomy and label-map hashes, package versions, the Git revision, working-tree status, and hashes of the generation code/configuration. The aggregate noisy manifest includes all nested outputs.

The audit reads each manifest and verifies its complete inventory and recorded values against the current files. It also checks schemas, duplicate IDs, physical split names, aligned pairs, contamination and derived-record source fidelity. Missing, malformed, stale or incomplete manifests fail the audit. Regenerate old `1.0.0` manifests using the preparation command; do not edit hashes to make an audit pass.

For final benchmark runs, review and commit the generation code, regenerate and audit the release from that revision, then preserve the complete local `data/` release (including sources and manifests) with the experiment results. Changing source locks, generation code, taxonomy or input files requires regeneration. An audit pass verifies these structural rules; it does not assess label quality or semantic near-duplicates.
