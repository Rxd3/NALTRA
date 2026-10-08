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

CORDIS ordinary classification compares `labels_direct` against direct predictions over
the fixed supported label universe. Hierarchical metrics expand ancestors for both sides.
English, translated Turkish, and synthetic code-switch records inherit their source project split.

## Evaluate the trained CORDIS neural models

Run from the repository root after bilingual training finishes:

```powershell
.venv\Scripts\python.exe scripts\evaluate_all.py --device cuda --batch-size 8
```

The command loads the existing BiLSTM and Transformer under
`models/cordis_v0.4.0/cordis_h2020/en_tr_direct`, audits the dataset, and evaluates all
4,711 English and 4,711 Turkish test records. It never trains or changes model artifacts.
It verifies the artifact taxonomy, complete 473-label head, direct targets, and training
release output hashes. Smoke artifacts and incompatible releases are rejected.

The console prints micro-F1 and macro-F1 per language and for the pooled bilingual set.
`results/metrics/cordis_v0.4.0/evaluation_summary.json` contains precision, recall, F1,
sample Jaccard, exact-set accuracy, ancestor-expanded hierarchical metrics, ECE, and
Brier score. Macro scores include the complete fixed label universe, including labels
absent from a slice. Calibration includes every direct-label probability, including
labels below the prediction threshold; its binary labelwise definition and reliability
bin counts are recorded. Environment, batch size, saved thresholds, artifact checksums,
dataset manifests, seed, and evaluation code hashes are saved with the scores.

For a quick command check, use validation data and a separate output directory:

```powershell
.venv\Scripts\python.exe scripts\evaluate_all.py --split validation --max-records 16 --device cuda --batch-size 8 --output-dir results/metrics/cordis_validation_check
```

Limited runs are explicitly marked `scope: subset`. Full test scores require a run without
`--max-records`. Existing results are protected; select a new `--output-dir` or explicitly
use `--overwrite`. `--models bilstm` or `--models transformer` selects one model, and
`--languages en` or `--languages tr` selects one language. Custom artifact roots use
`--model-dir` pointing to the directory containing the model family folders.

By default, thresholds remain those stored in the trained artifact. To select a global
cutoff for each model on full bilingual validation, run:

```powershell
.venv\Scripts\python.exe scripts\evaluate_all.py --split validation --tune-threshold --device cuda --batch-size 8 --output-dir results/metrics/cordis_validation_tuned
```

The search maximizes pooled micro-F1 over cutoffs 0.00 to 1.00 in steps of 0.01, with
ties resolved in favor of the higher cutoff. Each model has one cutoff shared across
English and Turkish. The summary stores every candidate, the chosen thresholds,
`baseline_slices` at the artifact thresholds, and `slices` at the selected thresholds.
These validation scores were used for selection and are not an unbiased final estimate.
Subset, single-language, and test tuning are rejected. No model files are changed.
Ranking metrics (micro and sample average precision) use complete probabilities and
do not depend on a prediction threshold. Low binary labelwise ECE or Brier scores alone
do not establish useful classification in this sparse label space.

Once validation performance is acceptable, freeze the settings and evaluate test:

```powershell
.venv\Scripts\python.exe scripts\evaluate_all.py --split test --thresholds-file results/metrics/cordis_validation_tuned/evaluation_summary.json --device cuda --batch-size 8 --output-dir results/metrics/cordis_test_tuned
```

The runner verifies that the threshold source is a successful full validation tuning
report with matching artifact and dataset output hashes. Test evaluation applies the
frozen thresholds without selecting or calibrating on test data.

This command covers clean CORDIS neural classification. Code-switch, noise, ensemble,
external-provider, and OOD experiments remain separate integration work. CORDIS OOD
domain folds need a separately specified protocol. `scripts/run_benchmark.py` still
contains placeholder values and is not a source of measured latency results.
