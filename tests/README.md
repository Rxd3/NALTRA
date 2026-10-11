# Tests

Pytest suite of 43 `test_<name>.py` files on small fixtures (`conftest.py`); no data
download, weights or GPU needed. Run `python -m pytest`; on Windows run one file per
process ([command and reason](../CONTRIBUTING.md#development-setup)).

- Data: build_taxonomy, taxonomy, preprocessing, unpack_data, data_audit, manifest,
  line_endings, code_switching, noise, sample_sets, translation_exceptions
- Models: model_interface, classical_models, neural_models, hybrid_knn, kev_model, jev_model,
  laya_model, external_models, training_cli
- Inference: pipeline, core_pipeline, hierarchy, ensemble, prediction_schema, classify
- Evaluation and reporting: metrics, matrix_metrics, bootstrap, offline_benchmark, predict_all,
  evaluate_ensemble, evaluation_cli, extra_analyses, run_benchmark, run_ood, provenance,
  figures_data, make_figures, plot_results, build_dashboard, dashboard_dom (skipped without
  a headless Chromium browser)
- Packaging: packaging (builds and installs the wheel offline, then runs inference outside
  the checkout)
