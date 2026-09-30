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
