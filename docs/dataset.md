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
2. **Turkish (`tr`)**: A machine-translated variant of the identical English project records. The shipped TR 1.1.0 is a frozen imported release that the committed scripts cannot rebuild exactly (see [Shipped Turkish Release](#shipped-turkish-release-110)).
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
- **Raw Archive Size**: 55,219,250 bytes (55.2 MB, 52.66 MiB)
- **Raw Archive SHA-256 Checksum**:
  `d4a8e3645d0b1896b967618e8e7a7f7928738b469737363a30816c217cbd48f2`
- **Extracted Files**:
  - `project.csv`: 78,743,942 bytes, SHA-256: `60221f84edb4a23c058120179ef786331dd39cf209a93ab5bf19b0daae0b5e7a` (35,389 project records)
  - `euroSciVoc.csv`: 14,020,349 bytes, SHA-256: `cb3c1344d3d32b7eeb7c983e3a86a4c0ba37151ff8ad3e87bf134d9306e396b7` (111,829 project-category assignment rows: 1,053 distinct category codes over 32,210 projects)

---

## 3. Classification Ontology: EuroSciVoc

EuroSciVoc (European Science Vocabulary) is the authoritative taxonomy developed by the European Union Publications Office to classify scientific activities across Horizon 2020 and Horizon Europe.

### Active Taxonomy Reconstruction
The active taxonomy is reconstructed directly from the official hierarchical classification paths included in the CORDIS Horizon 2020 release (`euroSciVoc.csv`); `python scripts/build_taxonomy.py` rebuilds `taxonomy/taxonomy.json` and `taxonomy/label_map.json` from the raw archive. Every category path encodes the full root-to-leaf hierarchy (e.g. `/23/43/253/751` -> `natural sciences / physical sciences / nuclear physics / nuclear fusion`).

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

| Support Threshold | Retained Direct Labels | Required Ancestor Set | New Ancestor Nodes | Total Active Labels | Retained Projects | % Projects Retained | Min Positives | Stratifiable |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 5 | 965 | 301 | 25 | 990 | 32,171 | 99.88% | 5 | Yes |
| 10 | 857 | 285 | 47 | 904 | 32,093 | 99.64% | 10 | Yes |
| 20 | 714 | 263 | 77 | 791 | 31,944 | 99.17% | 20 | Yes |
| **50 (Selected)** | **473** | **225** | **113** | **586** | **31,407** | **97.51%** | **50** | **Yes** |
| 100 | 297 | 174 | 112 | 409 | 30,395 | 94.37% | 100 | Yes |

The required ancestor set is every proper ancestor of a retained direct label. Many of those
ancestors are direct labels themselves (112 of the 225 at threshold 50), so only the new
ancestor nodes enlarge the tree: total active labels = retained direct labels + new
ancestor nodes (473 + 113 = 586).

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
- **Direct Label Cardinality**: Mean = 3.21, Median = 3.0, Max = 5
- **Hierarchy-Closed Cardinality**: Mean = 10.24, Median = 10.0, Max = 24
- **Multi-Root Projects (spanning >1 OECD root domain)**: 18,794 of 31,407 (**59.8%**)
- **Hierarchy Violation Rate in Clean Targets**: Exactly **0.00%**

---

## 5. Unified Record Schema

Processed records are serialized as JSON Lines (JSONL). Every record complies strictly with the schema validated by `validate_record()` in `src/naltra/data/preprocessing.py`.
This is the shipped English record `cordis:641585:en` from `data/processed/cordis_h2020/en/train.jsonl.gz`,
with every field in its stored order; only `text` (1,991 characters) is cut short with `…`:

```json
{
  "id": "cordis:641585:en",
  "project_id": "641585",
  "pair_id": "cordis:641585",
  "source_id": "641585",
  "title": "Combined Positioning-Reflectometry Galileo Code Receiver for Forest Management",
  "text": "Biomass mapping has gained increased interest for bioenergy, climate research and mitigation activities, … The consortium includes universities and companies for successful services and technology exploitation.",
  "labels_direct": ["drones", "global_navigation_satellite_system", "radar", "satellite_technology", "silviculture"],
  "labels": ["aerospace_engineering", "agricultural_sciences", "agriculture_forestry_and_fisheries",
             "autonomous_robots", "drones", "electrical_engineering_electronic_engineering_information_engineering",
             "electronic_engineering", "engineering_and_technology", "forestry",
             "global_navigation_satellite_system", "information_engineering", "mechanical_engineering",
             "navigation_systems", "radar", "radio_technology", "robotics", "satellite_navigation_system",
             "satellite_technology", "silviculture", "social_geography", "social_sciences",
             "telecommunications", "transport", "vehicle_engineering"],
  "language": "en",
  "source": "cordis_h2020",
  "license": "CC BY 4.0",
  "taxonomy_version": "0.4.0",
  "content_fingerprint": "31d5fed606ac0ef20c8d179aa105f2f7098f0fce543530c8efa25f4363078b20",
  "human_validated": "false",
  "synthetic_language_variant": false,
  "source_label_ids": ["/25/73/30020/455/1225/1727", "/25/73/453/459/1229/1729", "/25/75/461/1239/1739",
                       "/27/81/495/1283", "/29/99/547/1351/1765/1835"],
  "source_labels": ["drones", "global navigation satellite system", "radar", "satellite technology", "silviculture"],
  "split": "train"
}
```

`human_validated` is copied from the `Human-validated` column of CORDIS `project.csv`
(`NA` when empty; 30,697 `false`, 659 `true` and 51 `NA` records). It describes the
CORDIS record, not the NALTRA translations.

### Derived Turkish Variant Schema
Turkish records (`cordis:641585:tr` for the example) have the English fields from `id` to
`labels`, with `id` ending in `:tr` and a translated `text` (`title` stays English in all
31,407 records), plus `language` (`tr`), `source`, `license`, `split`, `taxonomy_version` and
`synthetic_language_variant` (`true`). They drop `content_fingerprint`, `human_validated`,
`source_label_ids` and `source_labels`, and add translation provenance:
- `variant_of`: the English record, e.g. `"cordis:641585:en"`
- `english_source_text`: the English `text` that was translated
- `translation_backend`: name of the translator (`"huggingface_local"`)
- `translation_model`, `translation_model_revision`: `Helsinki-NLP/opus-mt-tc-big-en-tr` @ `e539fc16a8a1a0ea5950eb339b595bfcce990e90`
- `translation_source_hash`: SHA-256 of the source English text
- `sentence_alignment`: list of aligned sentence pairs `[{"index": 0, "en": "...", "tr": "..."}]`
- the 30,809 imported records also keep `translation_requested_model`, the legacy alias `"Helsinki-NLP/opus-mt-en-tr"`, which `src/naltra/data/translation.py` normalises to the checkpoint above; the 598 records repaired in 1.1.0 carry `import_archive_sha256` and `repair_source_chunk_tokens` instead

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
- Funding call identifiers (e.g. `H2020-MSCA-NIGHT-2014`)
- EuroSciVoc title text (to prevent trivial label leakage)

### Cleaning Operations
- HTML markup tags (e.g. `<p>`, `<b>`, `<li>`) are stripped via regex.
- HTML entities (e.g. `&amp;`, `&quot;`, `&deg;`) are unescaped.
- Consecutive whitespace and duplicated line breaks are collapsed.
- Everything else is kept as written: casing, numbers, punctuation, acronyms, chemical and biomedical notation (e.g. $CO_2$, $pH$, $300^\circ C$) and formulas, except for the angle-bracket spans below.

#### Known limitation: angle-bracket comparisons

The tag pattern in `preprocess_record_text` (`src/naltra/data/preprocessing.py`) removes
every `<...>` span, not only HTML tags, so the text between a `<` and a later `>` is lost.
In the locked raw archive, 62 retained projects (48 train, 5 validation, 9 test) contain such
a non-tag span, and 17,290 characters are removed from them in total; the stored English
text confirms the loss. Examples:

- `694343` (test): "after (z<3) and before (z>3)" becomes "after (z 3)".
- `670462` (test): "high space (< 1 m) and time (>10 Hz) resolution" becomes "high space ( 10 Hz) resolution".

Affected projects: 641004, 645759, 646858, 650006, 653296, 654013, 670462, 670557, 679933,
681514, 694343, 704030, 715770, 716862, 720720, 723954, 724958, 735218, 737884, 738295,
738654, 742095, 745460, 748759, 751939, 755731, 758638, 761093, 762394, 764675, 766617,
768583, 774285, 791751, 807366, 808774, 811640, 817139, 825246, 827343, 832329, 836805,
838077, 852992, 863222, 866570, 867637, 869898, 869922, 871741, 872800, 882740, 884111,
885695, 886322, 947852, 958837, 964883, 971145, 101006941, 101009882, 101033663.

Labels are unaffected: they come from `euroSciVoc.csv`, not from the text. The Turkish
release was translated from the cleaned English text, so it lacks the same spans. Dropping
the 9 affected test projects moves each member's full-set EN micro-F1 by at most 0.0002
(recomputed from the reported dumps and thresholds; 2 of them are in the 120-pair sample,
where the change is at most 0.003); the effect of the text lost from the 48
training projects has not been measured.
`tests/test_preprocessing.py::test_comparison_operators_survive_cleaning` documents the
behaviour as a strict expected failure (`xfail(strict=True)`). The 1.1.0 text is locked by
its manifest hashes and is not changed in place; fixing the cleaner needs a new, versioned
release.

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
- **Model Identifier**: Production Turkish translation uses `Helsinki-NLP/opus-mt-tc-big-en-tr`, an English-to-Turkish Marian/Transformer model from the OPUS-MT project. Release 1.1.0 records name it in `translation_model`; the legacy alias `Helsinki-NLP/opus-mt-en-tr` for this same checkpoint survives only in `translation_requested_model`.
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
2. **Deterministic Chunking Fallback**: MarianMT enforces a 512-position limit. Across the entire CORDIS corpus (280,039 sentences across 31,407 projects: a historical count from the 1.0.0 translation run, whose log was not kept and which no manifest records; today's `segment_sentences` splits the same English text into 312,782, the segmenter drift that the [prebuilt audit](#shipped-turkish-release-110) tolerates, and the shipped 1.1.0 alignments hold 283,002 sentence pairs), sentence length statistics are:
   - Mean tokens: 40.14
   - Median tokens: 34.0
   - 95th percentile: 87.0
   - 99th percentile: 148.0
   - Maximum tokens: 454 (zero sentences exceed 512 tokens).
   A deterministic clause-level chunking fallback (`chunk_sentence_if_needed`, max 500 tokens) is active to protect against any unexpected sequence overflows.
3. **Persistent Resumable Disk Cache**: Every translated document is keyed by `SHA-256(namespace:project_id:source_hash:backend:model)`, where `namespace` is the sorted-key JSON of the checkpoint `revision`, the SHA-256 of `src/naltra/data/translation.py` (`code`) and the translator's `max_length` (`TranslationCache.compute_key` and `translate_cordis_dataset` in that file), and saved as an individual JSON entry in `data/cache/translations/cordis_h2020/`. The translator checks the cache first, allowing uninterrupted restarts and zero redundant computation.
4. **Automated Structural QA** (`check_translation_qa` in `src/naltra/data/translation.py`, run on every record during translation and again by the validator, which adds the record checks in `src/naltra/data/validation.py`). These are structural screens, not a judgement of translation quality:
   - Turkish text non-empty, and not identical to the English source (ignoring case and surrounding space)
   - Character length ratio TR/EN within $0.35 \le \text{ratio} \le 3.0$
   - No ASCII control characters (U+0000–U+0008, U+000B, U+000C, U+000E–U+001F)
   - `labels`, `labels_direct`, `pair_id`, `split`, `source` and `license` identical to the English record; `language` is `tr`, `id` is `cordis:<project_id>:tr` and `variant_of` names the English record
   - `translation_source_hash` equals the SHA-256 of the English text
   - `sentence_alignment` is a non-empty list of pairs indexed 0..n-1 with non-empty `en` and `tr` parts; the English parts cover the source text in order, and the Turkish parts joined with spaces reproduce the Turkish text (the [prebuilt audit](#shipped-turkish-release-110) tolerates and reports segmenter drift)
   - Every English project of a split has exactly one translation, and the splits stay disjoint
   - One translation provenance across all records: backend `huggingface_local` (no mock output) and, for a rebuilt release, the locked model and revision

   Nothing checks for Turkish characters, replacement characters (U+FFFD) or that the output is Turkish at all: a Turkish text containing U+FFFD, or unrelated English text of a plausible length, passes.
5. **Stratified Human Review Sample**:
   - Location: `data/review/translation_review_sample.csv`
   - Size: 100 validation projects. Their direct labels reach all 6 OECD root domains (natural sciences 94 projects, engineering and technology 70, social sciences 45, medical and health sciences 37, agricultural sciences 20, humanities 7). `hierarchy_depth`, the depth of the deepest direct label, covers depths 4–7 only (4: 12, 5: 56, 6: 30, 7: 2), and every project is multi-label (5 direct labels: 93, 4: 5, 3: 2); there are no single-label records.
   - Column caveat: despite its name, `top_level_domain` holds each project's first direct label in alphabetical order (e.g. `4g`, `5g`), not its root domain; the committed file predates the current `create_stratified_human_review_sample`, which writes the root.
   - Formal Human Review Status: `review_status = pending` (no fabricated human scores; true human audit pending). The human-review columns hold no ratings yet: in all 100 rows `meaning_preserved`, `terminology_quality` and `fluency` read `pending`, and `notes` says the review is pending.
   - Reviewed text: the sample and the AI-assisted audit below predate TR 1.1.0. 82 of the 100 rows match the shipped 1.1.0 Turkish exactly (unchanged since 1.0.0); the other 18 are among the 598 records repaired in 1.1.0, and their sample text matches neither 1.0.0 nor 1.1.0. The sample has not been regenerated, so those 18 rows show Turkish that no release ships. The audit covers the same 100 projects.

6. **AI-Assisted Translation Quality Audit of a Stratified 100-Project Sample**:
   - Location: `data/review/translation_ai_review.csv` (`reviewer_type = ai_assisted` for all 100 rows).
   - Sample Definition: `data/review/translation_review_sample.csv` (the 100 validation projects described above).
   - Methodology: AI-assisted semantic and terminological rating of the English CORDIS source against the Turkish neural MT output, one row per sampled project, after pipeline hardening.
   - Ratings in the file (100 projects): meaning preserved `yes` 83, `partial` 17, `no` 0; terminology quality (1–5) mean 4.52, median 5, $\le 2$: 5; Turkish fluency (1–5) mean 3.84, median 4, $\le 2$: 0.
   - Historical figures without a tracked source: a before-hardening pass reported meaning `yes` 73, `partial` 24, `no` 3; terminology mean 4.34 ($\le 2$: 11); fluency mean 3.71 ($\le 2$: 3); and 1 repetition loop, 6 severe truncations or omissions and 18 important omissions, reduced after hardening (`no_repeat_ngram_size=4`, period normalization, list/bullet segmentation and chunking) to 0, 0 and 4. None of these can be recomputed from the repository.
   - Known Remaining Limitations:
     - Pretrained model lexical polysemy in specialized sub-disciplines (*harvesters* $\rightarrow$ *biçerdöverler*, *technology adoption* $\rightarrow$ *evlat edinme*, *stemness* $\rightarrow$ *saplık*, *crickets* $\rightarrow$ *kriketler*, *knockout* $\rightarrow$ *nakavt*).
     - Calque phrasing in complex multi-clause sentences.
   - Formal Human Review Status: **PENDING** (this audit is explicitly AI-assisted, NOT human validation; human expert review remains pending).

### Bulk Translation Execution Metrics (RTX 3060)
- **Source English Projects**: 31,407
- **Successful Turkish Projects**: 31,407 (100.0% completion; 0 failures; 0 mock records)
- **Split Distribution**: `train`: 21,985 | `validation`: 4,711 | `test`: 4,711
- **Bulk Execution Runtime**: 234.52 minutes (~3.91 hours)
- **Overall Throughput**: 133.9 projects/minute (~19.9 sentences/second: 280,039 sentences / (234.52 × 60) s). The runtime and sentence count are historical figures from the 1.0.0 translation run; no manifest records them.
- **Peak VRAM Allocated**: 2,837.2 MB as logged by the 1.0.0 run, of the card's 12,288 MiB (the logging code is not in the repository, so the unit may be MiB)
- **Output Files**: `data/processed/cordis_h2020/tr/{train,validation,test}.jsonl` and `data/processed/cordis_h2020/tr/manifest.json`.

### Shipped Turkish Release (1.1.0)
- All four manifests (EN, TR, both code-switch tracks) are 1.1.0; the EN splits are byte-identical to 1.0.0. 1.1.0 is the 1.0.0 import with the 598 records the 1.0.0 audit flagged (text not matching the stored sentence alignment; 3 also failed the length-ratio QA check) re-translated in 128-token chunks by the pinned checkpoint (`generation_parameters`: `operation` `validated_archive_import_with_targeted_alignment_repair`, `repaired_count` 598, `legacy_model_names_normalized` 30,809). All 598 alignments were rewritten; the TR text changed in 300 of them (214 train, 49 validation, 37 test). Labels are identical.
- TR 1.1.0 is a frozen imported release. Its targeted repair (128-token source chunks, `repair_source_chunk_tokens` 128 in `generation_parameters`) was run by a teammate, and the repair script is not in the repository. EN can be rebuilt from the raw archive with the committed scripts, and so can the derived tracks from the EN/TR release they are built on, but a standard translation run (chunks of up to 500 tokens) builds a different TR release, which needs a new release version.
- Every member, including the zero-shot Kev, Jev and Laya, is evaluated on 1.1.0, and NB, SVM and `hybrid_knn` are trained on it. The teammate-trained `bilstm` and `transformer` (release `cordis_v0.5.0`, whose recorded TR split SHA-256s match this release) refit their heads on 1.1.0, on encoders trained in run v0.4.0; the data XLM-R's encoder was fine-tuned on there is not recorded.
- `scripts/translation_exceptions.py` reports the projects the audit flagged and scores every member with and without them: none on 1.1.0; on 1.0.0 its 598 projects moved any member's micro-F1 by at most 0.0011 (a historical figure; that run's output is not tracked).

**Validating the shipped release.** The four 1.1.0 manifests (EN, TR, both code-switch
tracks) record the generating checkout: `code.git_revision` `ea00e6b` (merged from PR #9)
and the SHA-256 of every data-code file in `code.files`. The data code has changed since,
and the raw CSVs that the manifests hash exist only inside the committed archive until
`prepare_data.py` extracts them. The full validator (`python scripts/validate_datasets.py`
without `--prebuilt-release`, and every command that uses it) therefore refuses the shipped
release and prints a hint to use `--prebuilt-release`.

`python scripts/validate_datasets.py --prebuilt-release` is the audit for the shipped data.
It checks every EN and TR record (identity, hierarchy closure, fingerprints, split leakage,
EN/TR alignment) and the split checksums and counts each manifest declares (a missing EN or
TR manifest or split checksum fails), instead of the code hashes. Alignment-text mismatches and
QA failures are reported, not enforced; on 1.1.0 both are 0. It reports
15,434 segmenter-drift records, which it tolerates; training records them in each
artifact's provenance. Like `origin/main`'s validator, it accepts alignments whose English
parts equal `segment_sentences(source)`, which rewrites bullet and list marks; 204 repaired
records rely on this. It does not audit the two code-switch tracks: their manifests lock the
same `ea00e6b` code hashes, combining `--prebuilt-release` with `--code-switch` or
`--derived` is a usage error, and no records-only
code-switch audit exists. Training and noisy or code-switch generation on the shipped data
take the same `--prebuilt-release` flag. A release rebuilt with the committed scripts is checked
with the full validator instead.

---

## 9. Code-Switch Benchmark Track

Controlled code-switched records are generated strictly from aligned English and Turkish sentence pairs of the **exact same project record**:
- **Strategy**: Deterministic sentence mixing (`sentence_mix`) and chunk mixing (`chunk_mix`).
- **Primary Language**: Each record's seeded generator picks English or Turkish as the primary language with probability 1/2; the other is the secondary language. It is stored in `primary_language`.
- **Strengths** (`mix_aligned_sentences` in `src/naltra/data/code_switching.py`): `balanced` alternates primary and secondary sentences (`sentence_mix`) or takes the first half of the sentences in the primary language and the rest in the secondary (`chunk_mix`), so about half of the sentences are in each language. `light` puts every third sentence (`sentence_mix`, about 1/3) or the last quarter of the sentences (`chunk_mix`, about 1/4) in the secondary language. A single-sentence document instead joins the first half of its primary-language words with the second half of its secondary-language words. The shipped tracks use `balanced` only.
- **Pairing Guarantee**: Never mixes sentences across different project IDs.
- **Seed Determinism**: Per-pair SHA-256 seed derived from `pair_id`, strategy, strength, and base seed (42).
- **Benchmark Role**: Evaluation-only robustness track (held out in validation and test).
- **Cardinality**: Exactly matching the source splits (4,711 validation, 4,711 test records per track; 18,844 total code-switched records generated across `sentence_mix` and `chunk_mix` balanced configurations).
- **Output Files**: `data/processed/cordis_h2020/code_switch/{sentence_mix,chunk_mix}/balanced/{validation,test}.jsonl`.

On a rebuilt release, audit just these tracks with `python scripts/validate_datasets.py
--languages en tr --code-switch`; optional noise files are not required. The audit
reconstructs every record from its exact current aligned source pair and rejects stale
translations, changed labels/lineage, incomplete inventories, and split contamination. It
uses the full validator, so it refuses the shipped release (see
[Shipped Turkish Release](#shipped-turkish-release-110)). No mixed-language
training split is generated: the reported code-switch scores come from the trained members
scoring both test tracks through `predict_all.py` and `evaluate_ensemble.py`
([runbook](runbook.md#3-dump-predictions), [evaluation](evaluation.md#code-switching-evaluation)).

---

## 10. Limitations & Ethical Considerations

1. **Semi-Automatic Label Caveat**: CORDIS classifications are generated in part by European Commission machine learning algorithms and keyword indexing. They are subject to label noise and should not be treated as gold human ground truth.
2. **Translation Quality Limitation**: Machine translation may introduce terminological inaccuracies or syntactic artifacts, especially in technical scientific domains. Semantic fidelity must be audited through the provided sample before drawing clinical or legal conclusions.
3. **Domain Bias**: CORDIS reflects European research priorities funded under Horizon 2020, with high concentrations in Natural Sciences and Engineering.
4. **Pretrained Translation Tool**: The OPUS-MT translation model is an external tool used strictly for language transformation; no benchmark labels or test texts were derived from Hugging Face datasets.
5. **Cleaning Loss**: The cleaner removes `<...>` comparison spans as if they were HTML tags, in 62 retained projects; see [Known limitation: angle-bracket comparisons](#known-limitation-angle-bracket-comparisons).

## Integrated release commands and model target contract

Preparation and the installed CLI now target CORDIS. Use `python scripts/prepare_data.py --dataset cordis_h2020 --download` for the locked original English release, then `--dataset translation --device cuda` for derived Turkish. Use `--dataset noisy` (EN/TR noisy test copies in `data/noisy/combined/medium/cordis_h2020/`) and `--dataset code_switch` for evaluation variants; add `--prebuilt-release` to derive them from the shipped 1.1.0 release. `--dataset all --download` executes the full sequence. Validate a rebuilt EN/TR release with `python scripts/validate_datasets.py --languages en tr`; add `--derived` to require all configured variants. The shipped release fails this full validation and is audited with `--prebuilt-release` instead ([Shipped Turkish Release](#shipped-turkish-release-110)). These commands rebuild EN and the derived tracks, not TR 1.1.0: `--dataset translation` builds a different TR release, which needs a new release version (see [Shipped Turkish Release](#shipped-turkish-release-110)).

Stored `labels` remains hierarchy-closed. Every trained member learns the `labels_direct` targets, with 473 supported direct targets and ancestor paths from the 586-node taxonomy. Ordinary metrics compare direct labels; hierarchical metrics expand ancestors. Near-OOD holds the humanities root out of training (`scripts/run_ood.py`).

Manifest schema 1.1.0 records and verifies the complete output/input inventory, raw snapshot locks, source lineage, taxonomy, split map, and generation code/configuration. Source/code changes require regeneration. Translation loads the pinned revision, uses a revision/implementation-specific cache identity, rejects implicit mock production output, and fails release publication for missing splits or failed translations. Automated QA is structural screening; the tracked AI-assisted review and pending human review do not establish human validation.
