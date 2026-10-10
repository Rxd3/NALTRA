# Generated results

Generated evaluation outputs are grouped into `metrics/`, `plots/`, `predictions/`, `benchmarks/`, and `ood/`. Git tracks only the small summaries behind the README Results; everything else is ignored and stays local.

| Tracked file | Written by |
| --- | --- |
| `metrics/cordis_v1.1.0_ensemble/summary.json`, `summary.md` | `scripts/evaluate_ensemble.py` on the full EN, TR, code-switch and noisy test sets |
| `metrics/cordis_v1.1.0_sample/summary.json`, `summary.md` | `scripts/evaluate_ensemble.py` on the fixed 80/120-pair samples, with Kev and Jev |
| `metrics/translation_exceptions.json` | `scripts/translation_exceptions.py` |
| `ood/cordis_v1.1.0/summary.json` | `scripts/run_ood.py` (humanities held out) |
| `benchmarks/cordis_v1.1.0/benchmark_summary.json`, `benchmark_summary.md` | `scripts/run_benchmark.py` |

The two `evaluate_ensemble.py` `summary.json` files record `code_commit` (`git describe --always --dirty`), `environment` and `bootstrap_seed`; the OOD summary records `code_commit`, `environment` and `bootstrap_seeds`, and `benchmark_summary.json` records `code_commit` and `environment`. Result JSONs store repository-relative paths. The commands that write them are in [docs/runbook.md](../docs/runbook.md).

Not tracked: the prediction dumps (`predictions/<run>/<family>/<set>.npz` and their `.json` sidecars), plots, the OOD run's retrained models (`ood/<run>/models/`), and every other run directory. The README's [Reproduce](../README.md#reproduce) section lists what a fresh clone can regenerate on CPU and what needs shared artifacts, a GPU or service access.
