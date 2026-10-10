# Contribution: Abdalraouf Alkabi

- **Name:** Abdalraouf Alkabi
- **Student ID:** 2309015482
- **GitHub:** afel6
- **Role:** Classical ML member (Naive Bayes, SVM, ensemble voting), plus the benchmark, evaluation and data-release integration that ties every member together.

## What I built

**Classical and retrieval members**
- TF-IDF Naive Bayes and a calibrated linear SVM (one-vs-rest, Platt calibration with folds grouped by project pair) on a shared classical-model base with saved, SHA-256-checked artifacts.
- The sparse+dense retrieval kNN member: TF-IDF and multilingual-e5 neighbours fused with reciprocal rank fusion, with a pinned encoder revision.

**New-method members**
- Kev-0.8B (local server) and Jev (hosted service) adapters that ask one yes/no question per label for all 473 labels, with strict validation of service settings and responses.
- Integrated the teammates' local Laya model into the same prediction path (`prepare_laya.py`, `predict_all.py --models laya`) so it is scored like Kev and Jev.

**Benchmark and evaluation**
- `predict_all.py` (hash-bound prediction dumps for every member and test set) and `evaluate_ensemble.py`: member thresholds tuned on validation half A, vote settings on half B (split by pair hash), test sets scored once.
- NALTRA votes: strict majority, tuned k-of-M, hierarchy-closed, soft and weighted soft, with a leave-one-out ablation.
- Paired project-grouped bootstrap (10,000 resamples) with Holm correction for votes against the best member and for EN→TR, code-switch and noisy drops.
- Benchmark configuration separating baselines from new methods and architecture from concrete model, and the label-skew justification for micro-F1 (predicting nothing already gives 99.3% per-label accuracy).
- Near-OOD experiment (humanities root held out of training), translation-audit check, CPU latency benchmark.

**Data release**
- Integrated release 1.1.0 (EN, machine-translated TR, two code-switch tracks) with manifests, `unpack_data.py`, the record-level `validate_datasets.py --prebuilt-release` audit, the taxonomy rebuild script, the noisy test sets and the fixed 80/120-pair evaluation samples.
- Found and disclosed a cleaning limitation (an HTML-tag pattern that also removes `<…>` comparisons in 62 projects), pinned by a test.

**Reporting and documentation**
- Interactive results page (`docs/results/index.html`), figures, tracked result summaries with commit, environment and seed provenance.
- README, dataset card, evaluation and runbook documentation, and a results report for expert review.
- Test suite for data integrity, leakage, members, ensembles, evaluation and the results page (about 780 tests).

## Key results

On the 4,711-project test sets (micro-F1, EN / TR):
- **SVM:** 0.558 / 0.489. It is the best system on every test set, and none of the tested votes beats it.
- **Weighted soft vote:** 0.523 / 0.472.
- **Retrieval kNN:** 0.393 / 0.382, losing 0.011 on Turkish against 0.070 for the SVM.

On the 120-pair sample, Jev scores 0.277 / 0.279 with no significant EN→TR drop. Held-out-root rejection reaches an AUROC of 0.93 with the SVM.
