# Dataset plan

NALTRA will maintain English, Turkish, and English-Turkish code-switched subsets. The data owner records provenance and licensing before adding a source.

## Required record fields

- Stable example identifier
- Raw text
- Canonical label identifiers
- Language tag (`en`, `tr`, or `en-tr`)
- Source and license metadata
- Split name after splitting

## Processing stages

1. Preserve immutable source exports under `data/raw/` locally.
2. Normalize records and map labels into `data/processed/`.
3. Produce deterministic, stratified train/validation/test files under `data/splits/`.
4. Keep OOD samples under `data/ood/` and generated perturbations under `data/noisy/`.
5. Publish aggregate statistics, checksums, and reproduction commands without committing large data files.

No dataset is bundled with the initial scaffold.

## Candidate Dataset: MultiFin

MultiFin is a multilingual financial NLP dataset containing 10,048 real-world
article headlines across 15 languages. It supports hierarchical classification
with 23 low-level topic labels and 6 high-level topic groups.

For NALTRA, the `all_languages_lowlevel` configuration is currently being
evaluated because it provides multi-label annotations.

### English and Turkish subset

NALTRA currently targets only English and Turkish samples from MultiFin.

| Split | English | Turkish | Total EN/TR |
|---|---:|---:|---:|
| Train | 1,747 | 1,436 | 3,183 |
| Validation | 437 | 359 | 796 |
| Test | 546 | 449 | 995 |
| Total | 2,730 | 2,244 | 4,974 |

The filtered EN/TR subset contains 4,974 examples.

### Multi-label statistics

- 23 unique low-level labels
- 3,373 examples contain one label
- 1,363 examples contain two labels
- 238 examples contain three labels
- 1,601 examples (32.19%) are multi-label

### Benchmark considerations

MultiFin is useful for:

- English/Turkish cross-language evaluation
- Multi-label classification
- Hierarchical classification
- Model comparison on identical train/validation/test splits

However, label frequencies are not balanced equally between English and
Turkish. For example, `VAT & Customs` is strongly represented in Turkish,
while `M&A & Valuations` is much more common in English.

Because of this imbalance, NALTRA should report per-language and per-label
metrics rather than relying only on aggregate scores.

MultiFin is also primarily a financial/business-domain dataset, so additional
datasets will be required to cover broader NALTRA topics.

### Provenance

- Dataset: MultiFin
- Paper: "MultiFin: A Dataset for Multilingual Financial NLP"
- Authors: Rasmus Jørgensen et al.
- Published: Findings of EACL 2023
- Hugging Face dataset: `awinml/MultiFin`
- Original dataset license: CC BY-NC 4.0

## Candidate Dataset: MN-DS

MN-DS is an English news classification dataset containing 10,917 annotation
rows representing 10,491 unique news articles.

The dataset uses a two-level hierarchical taxonomy with 17 broad level-1
categories and 109 fine-grained level-2 categories.

### Dataset structure

- 10,917 annotation rows
- 10,491 unique articles
- 17 level-1 categories
- 109 level-2 categories
- No missing titles, article content, or category labels
- Full news articles rather than only headlines

### Category balance

The level-2 categories are highly balanced.

- Smallest level-2 category: 100 annotations
- Largest level-2 category: 107 annotations
- Median level-2 category size: 100
- No level-2 category contains fewer than 50 examples

The level-1 category sizes vary because each parent contains a different
number of level-2 categories.

### Multi-label structure

Some articles appear multiple times because they have more than one category
annotation.

- 10,099 articles contain one annotation
- 392 articles (3.74%) contain multiple annotations
- Maximum annotations for one article: 4
- 137 articles span more than one level-1 category
- 392 articles contain multiple level-2 categories

During preprocessing, rows sharing the same article ID should be merged into
one article record containing multiple labels.

### Benchmark role

MN-DS is useful for:

- English hierarchical classification
- Broad topic coverage
- Fine-grained 109-label classification
- Parent/child taxonomy evaluation
- Balanced per-label benchmark comparisons

MN-DS is less suitable as NALTRA's main multi-label benchmark because only a
small portion of articles contain multiple annotations.

It complements MultiFin by providing much broader general-news coverage while
MultiFin provides English/Turkish multilingual evaluation and a higher
proportion of multi-label samples.

### Provenance

- Dataset: MN-DS
- Language: English
- Domain: General news
- License: CC BY 4.0
- Source: Zenodo


## Candidate Dataset: SIB-200

SIB-200 is a multilingual topic-classification dataset covering more than
200 languages and dialects. NALTRA currently uses the English (`eng_Latn`)
and Turkish (`tur_Latn`) configurations.

Both languages contain the same examples, labels, and official data splits.

### English and Turkish subsets

| Split | English | Turkish |
|---|---:|---:|
| Train | 701 | 701 |
| Validation | 99 | 99 |
| Test | 204 | 204 |
| Total | 1,004 | 1,004 |

The dataset contains seven topic categories:

- science/technology
- travel
- politics
- sports
- health
- entertainment
- geography

The category distribution is identical for English and Turkish.

### Cross-language alignment

English and Turkish use matching `index_id` values for every split.

- All train IDs and categories match
- All validation IDs and categories match
- All test IDs and categories match
- Every ID is unique inside its split
- No IDs overlap between train, validation, and test

This makes SIB-200 suitable for controlled English/Turkish cross-language
evaluation because models can be compared on translations of the same
underlying examples.

### Benchmark role

SIB-200 will primarily be used for:

- English/Turkish cross-language comparison
- language robustness evaluation
- clean multilingual topic classification
- controlled comparison using identical splits and labels

The categories are not perfectly balanced, so macro-F1 should be reported
alongside aggregate metrics.

### Provenance

- Dataset: SIB-200
- Hugging Face dataset: `Davlan/sib200`
- Paper: "SIB-200: A Simple, Inclusive, and Big Evaluation Dataset for
  Topic Classification in 200+ Languages and Dialects"
- License: CC BY-SA 4.0
