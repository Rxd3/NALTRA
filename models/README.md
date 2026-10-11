# Local model artifacts

Trained weights, vectorizers, checkpoints and downloaded pretrained files are written here
and are not committed: Git tracks only this README, so a clone contains no model. Train the
artifacts with `scripts/train_all.py` (Naive Bayes and SVM on CPU as in the README's
[Reproduce](../README.md#reproduce) section, the GPU families as in
[runbook section 2](../docs/runbook.md#2-train)) and build the Laya artifact with
`scripts/prepare_laya.py`.

Artifacts use `<run>/<dataset>/<track>/<family>/` directories, for example
`cordis_v0.5.0/cordis_h2020/en_tr_direct/bilstm/`. Neural artifacts include configuration,
taxonomy hashes, history, provenance and thresholds in `naltra.json`. BiLSTM adds weights
and vocabulary; Transformer saves local model/tokenizer files. Smoke tracks have a
`_smoke` suffix and are not benchmark models. A `laya` artifact (built by
`scripts/prepare_laya.py`, not trained) holds the pinned checkpoint files with their
SHA-256 values, which loading verifies, and the 473 label questions. See
[core ML operation](../docs/core_ml.md).

Load only artifacts from a trusted source. `naive_bayes`, `svm` and `hybrid_knn` store
`estimator.joblib`, which `joblib.load` unpickles, and unpickling can run code; the SHA-256
in the sibling `naltra.json` detects corruption, not tampering, because whoever can replace
the file can rewrite it too. These artifacts also refuse to load under a scikit-learn
version other than the one that saved them (1.9.1 for the models behind the reported
results, pinned in `requirements.txt`).
