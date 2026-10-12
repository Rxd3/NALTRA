# Result files

Git tracks the small summaries behind the reported results and their documentation
charts in `docs/figures/cordis_v1.1.0/`. Model weights and prediction dumps are
distributed separately in the shared
[Google Drive folder](https://drive.google.com/drive/u/2/folders/1Bb8_apsbu09pVMg6qy6OW83JzUim8ilC).
Plots from new runs and OOD model artifacts are not tracked in Git.

## Committed summaries

| Path | Contents | Producer |
| --- | --- | --- |
| `metrics/cordis_v1.1.0_ensemble/summary.{json,md}` | Full-set metrics (EN, TR, both code-switch tracks, noisy EN/TR), thresholds, votes, and bootstrap comparisons | `scripts/evaluate_ensemble.py` |
| `metrics/cordis_v1.1.0_sample/summary.{json,md}` | Fixed 80/120-pair EN/TR samples including Kev and Jev, with Laya scored but not voting | `scripts/evaluate_ensemble.py` |
| `metrics/label_coverage.json` | Label coverage of the support filter against the CORDIS closure within the active 586-node taxonomy, plus unpruned code-path counts | `scripts/extra_analyses.py coverage` |
| `metrics/cordis_v1.1.0_sample/pairwise.json` | Zero-shot versus trained members on the 120-pair sample, with interval half-widths | `scripts/extra_analyses.py pairwise` |
| `metrics/cordis_v1.1.0_ensemble/error_analysis.json` | EN and TR test errors by label frequency, root, and gold labels per project | `scripts/extra_analyses.py errors` |
| `metrics/cordis_v1.1.0_ensemble/robustness.json` | Each member's EN micro-F1 drop on TR, both code-switch tracks, and noisy EN, and SVM's drop minus each other member's (paired bootstrap, Holm within each condition) | `scripts/extra_analyses.py drops` |
| `metrics/translation_checks.json` | Possible truncations and repetitions in the TR splits, screened from the data alone | `scripts/extra_analyses.py translation` |
| `metrics/translation_quality.json` | Language ID of every EN and TR record, and round-trip chrF of 300 seeded TR test records plus every screen-flagged test record (automatic proxies, not human judgement) | `scripts/extra_analyses.py mtquality` (only when named) |
| `metrics/translation_exceptions.json` | Translation audit exceptions and their scoring impact | `scripts/translation_exceptions.py` |
| `ood/cordis_v1.1.0/summary.json` | Humanities-held-out detection results | `scripts/run_ood.py` |
| `benchmarks/cordis_v1.1.0/benchmark_summary.{json,md}` | Latency, throughput, and artifact sizes | `scripts/run_benchmark.py` |

Read the findings in [Results](../docs/results.md) or open the
[interactive dashboard](../docs/results/index.html).

The [results guide](../docs/results.md#regenerate-the-charts) includes accuracy,
robustness, efficiency, size and OOD charts. Regenerate them from the committed
summaries with `python scripts/plot_results.py`; model artifacts are not required.

## Provenance

The two `evaluate_ensemble.py` summaries record `code_commit` (`git describe --always
--dirty`), `environment` and `bootstrap_seed`; the OOD summary records `code_commit`,
`environment` and `bootstrap_seeds`. The six `extra_analyses.py` outputs record
`code_commit`, `environment` and the SHA-256 values of the files they read (`inputs`);
`translation_quality.json` also records the back-translation model revision and decoding
settings. All six were regenerated together in one run from a clean tree. The prediction dumps are
checked against the hashes in the summary they cite. The committed latency summary was
measured before its producer added top-level code and environment fields: it records the
runtime and hardware, but no `code_commit`.

The sample summary's `baselines` key lists the members that are scored but never vote
(`evaluate_ensemble.py --baselines`); each system's role is in
`benchmark.systems.<name>.role`. Result paths are repository-relative.

## Prediction dumps

Install the original dumps under `predictions/cordis_v1.1.0/<family>/`.
Each set needs both `<set>.npz` and its `<set>.json` metadata sidecar.

For `naive_bayes`, `svm`, `hybrid_knn`, `bilstm`, and `transformer`, the full-set names are:

```text
en_validation    tr_validation
en_test          tr_test
cs_chunk_test    cs_sentence_test
en_test_noisy    tr_test_noisy
```

For those five families plus `kev`, `jev`, and the non-voting `laya`, the sample-set
names are:

```text
en_val_kev       tr_val_kev
en_test_kev      tr_test_kev
```

The reported full and sample experiments therefore require **72 NPZ/JSON pairs**
(144 files; 40 full-set and 32 sample dumps, as listed under `dumps` in the two
summaries). The sidecars identify input checksums, label order, record counts,
artifact fingerprints, and producing settings.

## Generated outputs

Plots go under `plots/<run>/`. The separately trained OOD artifacts go under
`ood/<run>/models/`. New evaluation or training runs should use fresh output
directories, preserving the published release.

## Checking the CPU re-run

The CPU-only Naive Bayes and SVM re-run in the [reproduction guide](../docs/reproducibility.md)
writes `metrics/cpu_check`, `metrics/cpu_check_sample` and `ood/cpu_check`. Against the
committed summaries, these must match: `metrics.<set>.naive_bayes` and `.svm`, their
`thresholds` and `val_a_micro_f1`, `significance.cross_condition.naive_bayes` and `.svm`,
`dumps.<member>/<set>.npz_sha256` (the score matrices are byte-identical) and the OOD
`results`. These can differ by design: provenance (`code_commit`, `environment`, the OOD
`models.*.commit`), artifact hashes (`artifact_*`, `naltra.json`, `estimator.joblib`;
artifacts record code hashes and OS-specific paths), and everything that depends on the
member set (`leave_one_out`, the votes' metrics, `vs_best_member`, `k` and the soft
thresholds), because Naive Bayes and SVM vote there instead of the four members.

See [reproducibility](../docs/reproducibility.md) for artifact installation and
[the runbook](../docs/runbook.md) for evaluation commands.
