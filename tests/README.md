# Tests

Pytest suite on small fixtures; no data download, weights or GPU needed
([Reproduce](../README.md#reproduce)). On Windows, run one file per process, for example
`python -m pytest tests/test_noise.py`. Files are `test_<name>.py`; fixtures in `conftest.py`.

- Data: build_taxonomy, taxonomy, preprocessing, unpack_data, data_audit, manifest,
  line_endings, code_switching, noise, sample_sets, translation_exceptions
- Models: model_interface, classical_models, neural_models, hybrid_knn, kev_model, jev_model,
  laya_model, external_models, training_cli
- Inference: pipeline, core_pipeline, hierarchy, ensemble, prediction_schema, classify
- Evaluation and reporting: metrics, matrix_metrics, bootstrap, offline_benchmark, predict_all,
  evaluate_ensemble, evaluation_cli, extra_analyses, run_benchmark, run_ood, provenance,
  figures_data, make_figures, build_dashboard, dashboard_dom (needs headless Chromium)
