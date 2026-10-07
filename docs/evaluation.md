# Evaluation plan

All model families are evaluated through the same `PredictionResult` schema and fixed dataset splits.

## Areas

- Classification: precision, recall, micro-F1, macro-F1, and confusion matrices.
- Hierarchy: ancestor consistency and hierarchy-aware precision/recall/F1 using the 586-node EuroSciVoc taxonomy.
- Language: English, Turkish, code-switched, and cross-language transfer slices.
- Calibration: expected calibration error (ECE), Brier score, and reliability plots.
- OOD: detection AUROC and false-positive rate at 95% true-positive rate.
- Robustness: performance deltas and Jaccard prediction stability under perturbation.
- Efficiency: warmed-up latency distributions under documented hardware and batch sizes.
- Ensemble comparison: hard, soft, and weighted-soft voting against each of the six component models.

Benchmark scripts must save their configuration, taxonomy version, dataset checksum, random seed, and environment details. Ensemble runs must also record the voting method, global/per-label thresholds, and complete model-weight mapping. Never add invented or manually filled experimental values; generated tables belong under `results/`.

CORDIS ordinary classification compares `labels_direct` against direct predictions over the fixed supported label universe. Hierarchical metrics expand ancestors for both sides. English, translated Turkish, and synthetic code-switch records inherit their source project split. The evaluation orchestration scripts still contain placeholder summaries and must be connected to real artifacts and predictions before reporting experimental results. CORDIS OOD domain folds need a separately specified protocol.
