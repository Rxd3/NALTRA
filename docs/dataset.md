# Dataset Card & Research Methodology Specification

**Dataset Title**: CORDIS – EU research projects under Horizon 2020 (2014–2020)
**Ontology**: European Science Vocabulary (EuroSciVoc)
**Intended Task**: Multilingual Hierarchical Multi-Label Scientific Text Classification
**Target Languages/Variants**: English (`en`), Turkish (`tr`), English–Turkish Code-Switched (`en-tr`)

---

## 1. Executive Summary & Scientific Framing

NALTRA investigates **Multilingual Hierarchical Multi-Label Scientific Text Classification** using a single authoritative source dataset:
**CORDIS Horizon 2020 Projects (2014–2020)** published by the European Commission and the Publications Office of the European Union.

### One Source Dataset Principle
The benchmark relies strictly on one original source dataset. We do **not** combine CORDIS with Horizon Europe, FP7, external parallel corpora, external translation datasets, or third-party web scrapes.

All language variants evaluate the **exact same underlying research projects**:
1. **English (`en`)**: The canonical project objective and description released by the European Union.
2. **Turkish (`tr`)**: A reproducibly derived machine-translated variant of the identical English project records.
3. **English–Turkish Code-Switched (`en-tr`)**: Controlled, synthetic code-switched variants generated deterministically from aligned sentences of the identical English and Turkish versions.

All three variants maintain strict traceability:
- Identical project ID (`project_id`)
- Identical alignment pair ID (`pair_id = cordis:<project_id>`)
- Identical split assignment (`train`, `validation`, `test`)
- Identical direct EuroSciVoc labels (`labels_direct`)
- Identical hierarchy-closed target sets (`labels`)

---

## 2. Source Provenance & Official Distribution

- **Canonical Landing Page**: [CORDIS EU research projects under Horizon 2020 (2014-2020)](https://data.europa.eu/data/datasets/cordish2020projects)
- **Publisher**: European Commission / Publications Office of the European Union
- **Official Distribution URL**: `https://cordis.europa.eu/data/cordis-h2020projects-csv.zip`
- **Retrieval Date**: October 2026
- **Data Format**: Semicolon-delimited CSV with UTF-8 text encoding.
- **License / Reuse Terms**: Creative Commons Attribution 4.0 International (CC BY 4.0) under European Commission open data decisions.
- **Raw Archive Size**: 55,219,250 bytes (~52.6 MB)
- **Raw Archive SHA-256 Checksum**:
  `d4a8e3645d0b1896b967618e8e7a7f7928738b469737363a30816c217cbd48f2`
- **Extracted Files**:
  - `project.csv`: 78,743,942 bytes, SHA-256: `60221f84edb4a23c058120179ef786331dd39cf209a93ab5bf19b0daae0b5e7a` (35,389 project records)
  - `euroSciVoc.csv`: 14,020,349 bytes, SHA-256: `cb3c1344d3d32b7eeb7c983e3a86a4c0ba37151ff8ad3e87bf134d9306e396b7` (134,812 project-category assignment rows)

---

## 3. Classification Ontology: EuroSciVoc

EuroSciVoc (European Science Vocabulary) is the authoritative taxonomy developed by the European Union Publications Office to classify scientific activities across Horizon 2020 and Horizon Europe.

### Active Taxonomy Reconstruction
The active taxonomy is reconstructed directly from the official hierarchical classification paths included in the CORDIS Horizon 2020 release (`euroSciVoc.csv`). Every category path encodes the full root-to-leaf hierarchy (e.g. `/23/43/253/751` -> `natural sciences / physical sciences / nuclear physics / nuclear fusion`).

- **Taxonomy Version**: `0.4.0`
- **OECD Root Domains (Level 1)**: Exactly 6 root domains
  1. `natural_sciences` (`/23`)
  2. `engineering_and_technology` (`/25`)
  3. `medical_and_health_sciences` (`/21`)
  4. `agricultural_sciences` (`/27`)
  5. `social_sciences` (`/29`)
  6. `humanities` (`/31`)
- **Hierarchy Structure**: Strict single-parent tree hierarchy (zero polyhierarchy conflicts in active paths).
- **Depth**: Ranges from Level 1 (roots) to Level 7 (specialized leaves), with the mode at Level 4.

### Active Label Support Policy
EuroSciVoc contains a long tail of rare leaf categories. To prevent unlearnable categories and ensure robust multi-label evaluation, NALTRA applies a minimum direct-support threshold of **50 occurrences**:

| Support Threshold | Retained Direct Labels | Ancestors Added | Total Active Labels | Retained Projects | % Projects Retained | Min Positives | Stratifiable |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 5 | 965 | 301 | 990 | 32,171 | 99.88% | 5 | Yes |
| 10 | 857 | 285 | 904 | 32,093 | 99.64% | 10 | Yes |
| 20 | 714 | 263 | 791 | 31,944 | 99.17% | 20 | Yes |
| **50 (Selected)** | **473** | **225** | **586** | **31,407** | **97.51%** | **50** | **Yes** |
| 100 | 297 | 174 | 409 | 30,395 | 94.37% | 100 | Yes |

**Rationale for Threshold 50**:
- Every retained direct category contains at least 50 positive examples across the dataset, guaranteeing meaningful support across `train` (70% = ~35 examples), `validation` (15% = ~7 examples), and `test` (15% = ~7 examples).
- Retains **97.51%** of all labeled projects (31,407 out of 32,210), discarding only 803 projects that possessed exclusively rare tail categories.
- Retains the full depth of the EuroSciVoc ontology (depths 1 through 7) across all 6 OECD root domains.
- Required ancestors are **never** discarded; an ancestor is retained regardless of its direct frequency.

---

## 4. Multi-Label & Hierarchy Semantics

### Direct vs. Hierarchy-Closed Labels
In hierarchical multi-label classification, two distinct label sets must be preserved:
1. `labels_direct`: The canonical EuroSciVoc categories directly assigned to the CORDIS project record.
2. `labels`: The hierarchy-closed target set comprising all direct labels plus all active ancestor categories up to the root domain.

**Mathematical Invariants**:
$$\text{labels\_direct} \subseteq \text{labels}$$
$$\forall l \in \text{labels}, \quad \text{parent}(l) \neq \text{None} \implies \text{parent}(l) \in \text{labels}$$

### Empirical Multi-Label Statistics (Canonical Benchmark)
- **Total Valid English Projects**: 31,407
- **Multi-Label Projects (>1 direct label)**: 26,473 (**84.3%**)
- **Single-Label Projects**: 4,934 (15.7%)
- **Direct Label Cardinality**: Mean = 3.21, Median = 4.0, Max = 5
- **Hierarchy-Closed Cardinality**: Mean = 10.24, Median = 10.0, Max = 24
- **Multi-Root Projects (spanning >1 OECD root domain)**: 19,628 (**62.5%**)
- **Hierarchy Violation Rate in Clean Targets**: Exactly **0.00%**

---

## 5. Unified Record Schema

Processed records are serialized as JSON Lines (JSONL). Every record complies strictly with the schema validated by `validate_record()` in `src/naltra/data/preprocessing.py`:

```json
{
  "id": "cordis:633263:en",
  "project_id": "633263",
  "pair_id": "cordis:633263",
  "source_id": "633263",
  "title": "RESEARCHER'S NIGHT IN FRANCE",
  "text": "ERN has gained momentum in France...",
  "labels_direct": ["scientific_popularisation"],
  "labels": ["social_sciences", "educational_sciences", "scientific_popularisation"],
  "language": "en",
  "source": "cordis_h2020",
  "license": "CC BY 4.0",
  "split": "train",
  "taxonomy_version": "0.4.0",
  "content_fingerprint": "a3f8b912e74c...",
  "human_validated": "NA",
  "synthetic_language_variant": false,
  "source_label_ids": ["/29/81/405"],
  "source_labels": ["scientific popularisation"]
}
```

### Derived Turkish Variant Schema Additions
Turkish records include additional provenance metadata:
- `variant_of`: `"cordis:633263:en"`
- `synthetic_language_variant`: `true`
- `translation_backend`: Name of translator (e.g. `"huggingface_local"`)
- `translation_model`: Model name (e.g. `"Helsinki-NLP/opus-mt-tc-big-en-tr"`)
- `translation_source_hash`: SHA-256 of the source English text
- `sentence_alignment`: List of aligned sentence pairs `[{"index": 0, "en": "...", "tr": "..."}]`

Existing translations can be reused when their English source, taxonomy, labels, project
splits, and model provenance match the current release. Older sentence boundaries are
accepted only when the aligned English chunks reproduce the complete source in order;
the Turkish chunks must reproduce the stored translation exactly. Empty chunks and
missing or reordered source content are rejected.

Imported archives with older manifests need an audited manifest migration, preserving
the original manifest and archive checksum in provenance. Corrected translations with
stale alignment metadata require targeted repair before import. A review CSV alone is
not a complete translation release.

---

## 6. Text Preprocessing & Cleaning Policy

### Classification Input Field
The primary input text for classification is the project **`objective`** (summary description).

Fields strictly excluded from the classifier input:
- Participant names and consortium members
- Funding scheme and programme names (e.g. `MSCA`, `ERC`)
- Funding topic codes (e.g. `H2020-MSCA-NIGHT-2014`)
- EuroSciVoc title text (to prevent trivial label leakage)

### Cleaning Operations
- HTML markup tags (e.g. `<p>`, `<b>`, `<li>`) are stripped via regex.
- HTML entities (e.g. `&amp;`, `&quot;`, `&deg;`) are unescaped.
- Consecutive whitespace and duplicated line breaks are collapsed.
- **Strictly Preserved**: Chemical/biomedical notation (e.g. $CO_2$, $pH$, $300^\circ C$), mathematical formulas, numbers, punctuation, acronyms, and casing.

---

## 7. Data Splitting & Leakage Controls

### Pre-Translation Project-Level Stratification
Data partitioning is computed **strictly at the project level** prior to language translation or code-switch generation. All language variants of project $X$ are bound to the exact same partition:
$$\text{split}(\text{cordis:}X\text{:en}) = \text{split}(\text{cordis:}X\text{:tr}) = \text{split}(\text{cordis:}X\text{:codeswitch})$$

### Partition Sizes (Deterministic Seed: 42)
Partitioning applies grouped multi-label iterative stratification (`grouped_multilabel_stratified_split`):

| Partition | Record Count | Percentage |
| :--- | :---: | :---: |
| `train` | 21,985 | 70.0% |
| `validation` | 4,711 | 15.0% |
| `test` | 4,711 | 15.0% |
| **Total** | **31,407** | **100.0%** |

### Leakage Audit Results
1. **Project ID Overlap**: Exactly **0** projects cross splits.
2. **Pair ID Overlap**: Exactly **0** pair IDs cross splits.
3. **Exact Content Fingerprint Leakage**: In the raw dataset, 27 groups of projects (spanning 56 projects total) contain identical objective text under different grant numbers. Content fingerprints are used as grouping keys, guaranteeing that duplicate texts never cross split boundaries.

---

## 8. Translation Pipeline & Turkish Derivation

> [!IMPORTANT]
> **One Source Dataset Principle**: CORDIS Horizon 2020 itself is an **English source dataset**.
> Turkish text is a **derived machine-translated variant**.
> English–Turkish mixed text is a **synthetic derived variant**.
> Hugging Face is used exclusively to obtain the pretrained OPUS-MT neural translation model weights and is **NOT** a benchmark data source.
> Translations have **NOT** been human validated (human review status is `pending`).

### Pretrained Machine Translation Model
- **Model Identifier**: Production Turkish translation uses `Helsinki-NLP/opus-mt-tc-big-en-tr`, an English-to-Turkish Marian/Transformer model from the OPUS-MT project.
- **Architecture**: MarianMT / Marian Transformer encoder-decoder
- **Model Revision**: `e539fc16a8a1a0ea5950eb339b595bfcce990e90`
- **Model License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Local Model Location**: Hugging Face Hub local cache (`~/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-tc-big-en-tr`)
- **Inference Mode**: PyTorch `torch.inference_mode()`, evaluation mode (`model.eval()`), FP32 / CUDA.

### GPU Compute Environment
- **Accelerator**: NVIDIA GeForce RTX 3060 (12,288 MiB VRAM)
- **NVIDIA Driver**: 595.79
- **Driver CUDA Version**: 13.2
- **PyTorch Runtime**: `2.14.1+cu126` (CUDA 12.6 backend)
- **Transformers Library**: `5.18.0`
- **SentencePiece**: `0.2.2`

### Translation Architecture & Engineering
1. **Sentence-Level Alignment**: Text is segmented into sentences using abbreviation-protected boundary heuristics (preserving *et al.*, *e.g.*, *vs.*, *approx.*). Each sentence is translated independently to yield aligned sentence pairs.
2. **Deterministic Chunking Fallback**: MarianMT enforces a 512-position limit. Across the entire CORDIS corpus (280,039 sentences across 31,407 projects), sentence length statistics are:
   - Mean tokens: 40.14
   - Median tokens: 34.0
   - 95th percentile: 87.0
   - 99th percentile: 148.0
   - Maximum tokens: 454 (zero sentences exceed 512 tokens).
   A deterministic clause-level chunking fallback (`chunk_sentence_if_needed`, max 500 tokens) is active to protect against any unexpected sequence overflows.
3. **Persistent Resumable Disk Cache**: Every translated document is keyed by `SHA-256(project_id:source_hash:backend:model)` and saved as an individual JSON entry in `data/cache/translations/cordis_h2020/`. The translator checks the cache first, allowing uninterrupted restarts and zero redundant computation.
4. **Automated QA Validation Suite (15 Checks)**:
   - Project ID unchanged and non-empty
   - Alignment pair ID unchanged (`cordis:<pid>`)
   - Split assignment unchanged
   - Direct labels and hierarchy-closed labels 100% identical to English source
   - English source text non-empty
   - Turkish output text non-empty and not identical to English source
   - Source text SHA-256 hash matches English record
   - Translation backend != mock (production must use neural MT)
   - Provenance records model ID and model revision
   - Sentence count / order preserved
   - No Unicode corruption; valid Turkish characters verified (`ç`, `Ç`, `ğ`, `Ğ`, `ı`, `I`, `İ`, `ö`, `Ö`, `ş`, `Ş`, `ü`, `Ü`)
   - Reasonable character length ratio ($0.35 \le \text{ratio} \le 3.0$)
   - No orphan translations (every Turkish record maps to an existing English project)
5. **Stratified Human Review Sample**:
   - Location: `data/review/translation_review_sample.csv`
   - Size: 100 representative projects sampled across all 6 OECD root domains, depths 1–7, varied text lengths, single-label, and multi-label records.
   - Formal Human Review Status: `review_status = pending` (no fabricated human scores; true human audit pending).

6. **AI-Assisted Translation Quality Audit of a Stratified 100-Project Sample**:
   - Location: `data/review/translation_ai_review.csv` (`reviewer_type = ai_assisted` for all 100 rows).
   - Sample Definition: `data/review/translation_review_sample.csv` (100 representative projects sampled across all 6 OECD root domains, depths 1–7, single-label, and multi-label records).
   - Methodology: Rigorous semantic and terminological evaluation comparing English CORDIS source and Turkish neural MT output across all 100 sampled projects before and after pipeline hardening.
   - Comparative Quality Metrics (100-Project Sample):
     - **Meaning Preservation**:
       - *Before Hardening*: `yes`: 73 (73.0%), `partial`: 24 (24.0%), `no`: 3 (3.0%)
       - *After Hardening*: `yes`: 83 (83.0%), `partial`: 17 (17.0%), `no`: 0 (0.0%)
     - **Terminology Quality (scale 1–5)**:
       - *Before Hardening*: Mean: 4.34 | Median: 5.0 | $\le 2$: 11
       - *After Hardening*: Mean: 4.52 | Median: 5.0 | $\le 2$: 5
     - **Turkish Fluency (scale 1–5)**:
       - *Before Hardening*: Mean: 3.71 | Median: 4.0 | $\le 2$: 3
       - *After Hardening*: Mean: 3.84 | Median: 4.0 | $\le 2$: 0
     - **Structural Defect Elimination**:
       - *Repetition Loops*: Reduced from 1 to 0 (eliminated via `no_repeat_ngram_size=4` and period normalization).
       - *Severe Truncation/Omissions*: Reduced from 6 to 0 (eliminated via list/bullet segmentation and chunking).
       - *Total Important Omissions*: Reduced from 18 to 4.
   - Known Remaining Limitations:
     - Pretrained model lexical polysemy in specialized sub-disciplines (*harvesters* $\rightarrow$ *biçerdöverler*, *technology adoption* $\rightarrow$ *evlat edinme*, *stemness* $\rightarrow$ *saplık*, *crickets* $\rightarrow$ *kriketler*, *knockout* $\rightarrow$ *nakavt*).
     - Calque phrasing in complex multi-clause sentences.
   - Formal Human Review Status: **PENDING** (this audit is explicitly AI-assisted, NOT human validation; human expert review remains pending).

### Bulk Translation Execution Metrics (RTX 3060)
- **Source English Projects**: 31,407
- **Successful Turkish Projects**: 31,407 (100.0% completion; 0 failures; 0 mock records)
- **Split Distribution**: `train`: 21,985 | `validation`: 4,711 | `test`: 4,711
- **Bulk Execution Runtime**: 234.52 minutes (~3.91 hours)
- **Overall Throughput**: 133.9 projects/minute (~15.2 sentences/second)
- **Peak VRAM Allocated**: 2,837.2 MB (2.84 GB out of 12.0 GB)
- **Output Files**: `data/processed/cordis_h2020/tr/{train,validation,test}.jsonl` and `data/processed/cordis_h2020/tr/manifest.json`.

---

## 9. Code-Switch Benchmark Track

Controlled code-switched records are generated strictly from aligned English and Turkish sentence pairs of the **exact same project record**:
- **Strategy**: Deterministic sentence mixing (`sentence_mix`) and chunk mixing (`chunk_mix`).
- **Strengths**: `balanced` (approximately 50% English, 50% Turkish sentences) and `light` (20-30% Turkish).
- **Pairing Guarantee**: Never mixes sentences across different project IDs.
- **Seed Determinism**: Per-pair SHA-256 seed derived from `pair_id`, strategy, strength, and base seed (42).
- **Benchmark Role**: Evaluation-only robustness track (held out in validation and test).
- **Cardinality**: Exactly matching the source splits (4,711 validation, 4,711 test records per track; 18,844 total code-switched records generated across `sentence_mix` and `chunk_mix` balanced configurations).
- **Output Files**: `data/processed/cordis_h2020/code_switch/{sentence_mix,chunk_mix}/balanced/{validation,test}.jsonl`.

---

## 10. Limitations & Ethical Considerations

1. **Semi-Automatic Label Caveat**: CORDIS classifications are generated in part by European Commission machine learning algorithms and keyword indexing. They are subject to label noise and should not be treated as gold human ground truth.
2. **Translation Quality Limitation**: Machine translation may introduce terminological inaccuracies or syntactic artifacts, especially in technical scientific domains. Semantic fidelity must be audited through the provided sample before drawing clinical or legal conclusions.
3. **Domain Bias**: CORDIS reflects European research priorities funded under Horizon 2020, with high concentrations in Natural Sciences and Engineering.
4. **Pretrained Translation Tool**: The OPUS-MT translation model is an external tool used strictly for language transformation; no benchmark labels or test texts were derived from Hugging Face datasets.

## Integrated release commands and model target contract

Preparation and the installed CLI now target CORDIS. Use `python scripts/prepare_data.py --dataset cordis_h2020 --download` for the locked original English release, then `--dataset translation --device cuda` for derived Turkish. Use `--dataset noisy` and `--dataset code_switch` for evaluation variants. `--dataset all --download` executes the full sequence. Validate clean EN/TR with `python scripts/validate_datasets.py --languages en tr`; add `--derived` to require all configured variants.

Stored `labels` remains hierarchy-closed. The neural baseline trains on `labels_direct`, with 473 supported direct targets and ancestor paths from the 586-node taxonomy. Ordinary metrics compare direct labels; hierarchical metrics expand ancestors. CORDIS OOD domain splits are not implemented by the historical SIB-200 generators.

Manifest schema 1.1.0 records and verifies the complete output/input inventory, raw snapshot locks, source lineage, taxonomy, split map, and generation code/configuration. Source/code changes require regeneration. Translation loads the pinned revision, uses a revision/implementation-specific cache identity, rejects implicit mock production output, and fails release publication for missing splits or failed translations. Automated QA is structural screening; the tracked AI-assisted review and pending human review do not establish human validation.
