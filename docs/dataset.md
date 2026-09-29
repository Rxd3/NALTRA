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
