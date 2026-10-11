# Generated results

Generated evaluation outputs are grouped into `metrics/`, `plots/`, `predictions/`, `benchmarks/`, and `ood/`. Git tracks only the small summaries behind the README Results; everything else is ignored and stays local.

| Tracked file | Written by |
| --- | --- |
| `metrics/cordis_v1.1.0_ensemble/summary.json`, `summary.md` | `scripts/evaluate_ensemble.py` on the full EN, TR, code-switch and noisy test sets |
| `metrics/cordis_v1.1.0_sample/summary.json`, `summary.md` | `scripts/evaluate_ensemble.py` on the fixed 80/120-pair samples, with Kev, Jev and Laya |
| `metrics/label_coverage.json`, `metrics/cordis_v1.1.0_sample/pairwise.json`, `metrics/cordis_v1.1.0_ensemble/error_analysis.json` | `scripts/extra_analyses.py`: label coverage of the support filter, zero-shot vs trained members on the sample, error analysis |
| `metrics/cordis_v1.1.0_ensemble/robustness.json`, `metrics/translation_checks.json` | `scripts/extra_analyses.py drops translation`: each member's EN micro-F1 drop on TR, both code-switch tracks and noisy EN, and SVM's drop minus each other member's (paired bootstrap, Holm within each condition); possible truncations and repetitions in the TR splits (from the data only, no dumps) |
| `metrics/translation_exceptions.json` | `scripts/translation_exceptions.py` |
| `ood/cordis_v1.1.0/summary.json` | `scripts/run_ood.py` (humanities held out) |
| `benchmarks/cordis_v1.1.0/benchmark_summary.json`, `benchmark_summary.md` | `scripts/run_benchmark.py` |

The two `evaluate_ensemble.py` `summary.json` files record `code_commit` (`git describe --always --dirty`), `environment` and `bootstrap_seed`; the OOD summary records `code_commit`, `environment` and `bootstrap_seeds`, and `benchmark_summary.json`, measured before that field existed, records the runtime and hardware but no `code_commit`. The five `extra_analyses.py` outputs record `code_commit`, `environment` and the SHA-256s of the files they read (`inputs`); the prediction dumps are checked against the hashes in the summary they cite. The sample summary includes Laya as a non-voting member: its `baselines` key lists the members that are scored but never vote (the `evaluate_ensemble.py --baselines` flag), while each system's role is in `benchmark.systems.<name>.role`. Result JSONs store repository-relative paths. The commands that write them are in [docs/runbook.md](../docs/runbook.md).

Not tracked: the prediction dumps (`predictions/<run>/<family>/<set>.npz` and their `.json` sidecars), plots, the OOD run's retrained models (`ood/<run>/models/`), and every other run directory. The README's [Reproduce](../README.md#reproduce) section lists what a fresh clone can regenerate on CPU and what needs shared artifacts, a GPU or service access.

## Checking the CPU re-run

The README's CPU commands write `metrics/cpu_check`, `metrics/cpu_check_sample` and `ood/cpu_check`. Against the committed summaries, these must match: `metrics.<set>.naive_bayes` and `.svm`, their `thresholds` and `val_a_micro_f1`, `significance.cross_condition.naive_bayes` and `.svm`, `dumps.<member>/<set>.npz_sha256` (the score matrices are byte-identical) and the OOD `results`. These can differ by design: provenance (`code_commit`, `environment`, the OOD `models.*.commit`), artifact hashes (`artifact_*`, `naltra.json`, `estimator.joblib`; artifacts record code hashes and OS-specific paths), and everything that depends on the member set (`leave_one_out`, the votes' metrics, `vs_best_member`, `k` and the soft thresholds), because NB and SVM vote there instead of the four members.
