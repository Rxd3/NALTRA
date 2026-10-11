# Scripts

All 18 scripts run from the repository root; `python scripts/<name>.py --help` lists the
options. Inputs and outputs are in the runbook's [script reference](../docs/runbook.md#script-reference),
full commands in the [runbook](../docs/runbook.md) and the
[reproduction guide](../docs/reproducibility.md).

- `unpack_data.py`: committed gzip release -> manifest-verified JSONL splits
- `build_taxonomy.py`: raw CORDIS archive -> `taxonomy/taxonomy.json` and `label_map.json`
- `prepare_data.py`: raw archive or shipped release -> EN, TR, code-switch and noisy tracks
- `validate_datasets.py`: release audit (full, or `--prebuilt-release` record by record)
- `train_all.py`: splits + `configs/` -> trained artifacts under `models/`
- `prepare_laya.py`: pinned Laya checkpoint -> portable local artifact (no training)
- `sample_sets.py`: evaluation sets -> fixed pair-hash samples for Kev, Jev and Laya
- `predict_all.py`: artifacts + evaluation sets -> score-matrix dumps
- `classify.py`: artifacts + text -> labels, taxonomy closure, language and OOD decision;
  `--benchmark-settings` applies the frozen benchmark thresholds and votes
- `evaluate_ensemble.py`: dumps -> tuned thresholds, member/ensemble metrics, bootstrap tests
- `evaluate_all.py`: direct, resumable BiLSTM/XLM-R/Laya evaluator (no reported number uses it)
- `translation_exceptions.py`: scores with and without the projects the prebuilt audit flagged
- `extra_analyses.py`: label coverage, zero-shot member pairs on the 120-pair sample, errors,
  SVM's input-shift drop vs each other member's (`drops`), TR truncation and repetition
  screens (`translation`); `mtquality`, run only when named (`--device`, `--batch-size`),
  adds language ID of every EN/TR record and round-trip chrF through a pinned TR->EN model
- `make_figures.py`: summary + dumps -> paper figures
- `plot_results.py`: committed full, sample, latency and OOD summaries -> eight PNG/SVG
  charts and a source-checksum `manifest.json` in `docs/figures/cordis_v1.1.0/` (no models
  or dumps)
- `run_benchmark.py`: artifacts -> latency, throughput and size
- `run_ood.py`: NB/SVM retrained without one root -> near-OOD AUROC and FPR@95%TPR
- `build_dashboard.py`: summaries + dumps -> the results page `docs/results/index.html`
