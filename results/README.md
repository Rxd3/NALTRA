# Result files

Git tracks the small summaries behind the reported results and their documentation
charts in `docs/figures/cordis_v1.1.0/`. Large prediction dumps, plots from new runs,
and OOD model artifacts stay local.

## Committed summaries

| Path | Contents | Producer |
| --- | --- | --- |
| `metrics/cordis_v1.1.0_ensemble/summary.{json,md}` | Full-set metrics, thresholds, votes, and bootstrap comparisons | `scripts/evaluate_ensemble.py` |
| `metrics/cordis_v1.1.0_sample/summary.{json,md}` | Fixed EN/TR samples including Kev and Jev | `scripts/evaluate_ensemble.py` |
| `metrics/translation_exceptions.json` | Translation audit exceptions and their scoring impact | `scripts/translation_exceptions.py` |
| `ood/cordis_v1.1.0/summary.json` | Humanities-held-out detection results | `scripts/run_ood.py` |
| `benchmarks/cordis_v1.1.0/benchmark_summary.{json,md}` | Latency, throughput, and artifact sizes | `scripts/run_benchmark.py` |

Read the findings in [Results](../docs/results.md) or open the
[interactive dashboard](../docs/results/index.html).

The [results guide](../docs/results.md#regenerate-the-charts) includes accuracy,
robustness, efficiency, size and OOD charts. Regenerate them from the committed
summaries with `python scripts/plot_results.py`; model artifacts are not required.

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

For those five families plus `kev` and `jev`, the sample-set names are:

```text
en_val_kev       tr_val_kev
en_test_kev      tr_test_kev
```

The reported full and sample experiments therefore require **68 NPZ/JSON pairs**
(136 files). The sidecars identify input checksums, label order, record counts,
artifact fingerprints, and producing settings.

## Generated outputs

Plots go under `plots/<run>/`. The separately trained OOD artifacts go under
`ood/<run>/models/`. New evaluation or training runs should use fresh output
directories, preserving the published release.

Evaluation and OOD summaries record code/environment metadata and bootstrap seeds.
The latency producer records runtime environment details; the committed latency
summary lacks the top-level code/environment fields added to the current producer.
Result paths are repository-relative.

See [reproducibility](../docs/reproducibility.md) for artifact installation and
[the runbook](../docs/runbook.md) for evaluation commands.
