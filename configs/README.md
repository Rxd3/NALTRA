# Configs

YAML read by the scripts and models; overview in the [repository map](../README.md#repository-map).

- `data.yaml`: CORDIS archive and translation-model locks (URL, SHA-256, revision)
- `training.yaml`: defaults for `scripts/train_all.py`
- `benchmark.yaml`: benchmark protocol for `scripts/evaluate_ensemble.py` (primary metric,
  system roles)
- `evaluation.yaml`: calibration bins, read only by `scripts/evaluate_all.py`
- `models/*.yaml`: one file per model; their hashes are bound into trained artifacts
