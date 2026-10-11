# Scripts

Run from the repository root; inputs, outputs and options are in the root README's
[Scripts](../README.md#scripts) table, full commands in [Reproduce](../README.md#reproduce).

- `unpack_data.py`: committed gzip release -> manifest-verified JSONL splits
- `build_taxonomy.py`: raw CORDIS archive -> `taxonomy/taxonomy.json` and `label_map.json`
- `prepare_data.py`: raw archive or shipped release -> EN, TR, code-switch and noisy tracks
- `validate_datasets.py`: release audit (full, or `--prebuilt-release` record by record)
- `train_all.py`: splits + `configs/` -> trained artifacts under `models/`
- `prepare_laya.py`: pinned Laya checkpoint -> portable local artifact (no training)
- `sample_sets.py`: evaluation sets -> fixed pair-hash samples for Kev, Jev and Laya
- `predict_all.py`: artifacts + evaluation sets -> score-matrix dumps
- `classify.py`: artifacts + text -> labels, taxonomy closure, language and OOD decision
- `evaluate_ensemble.py`: dumps -> tuned thresholds, member/ensemble metrics, bootstrap tests
- `evaluate_all.py`: direct, resumable BiLSTM/XLM-R/Laya evaluator (no reported number uses it)
- `extra_analyses.py`: label coverage, zero-shot member pairs on the 120-pair sample, errors,
  SVM's input-shift drop vs each other member's (`drops`), TR truncation and repetition
  screens (`translation`)
- `translation_exceptions.py`: scores with and without the projects the prebuilt audit flagged
- `make_figures.py`: summary + dumps -> paper figures
- `run_benchmark.py`: artifacts -> latency, throughput and size
- `run_ood.py`: NB/SVM retrained without one root -> near-OOD AUROC and FPR@95%TPR
- `build_dashboard.py`: summaries + dumps -> the results page `docs/results/index.html`
