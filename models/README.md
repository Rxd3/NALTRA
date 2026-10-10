# Local model artifacts

Trained weights, vectorizers, checkpoints, and downloaded pretrained artifacts belong here locally and are ignored by Git. Document how an artifact was produced and share large files through the team's approved storage channel.

Artifacts use `<run>/<dataset>/<track>/<family>/` directories, for example
`cordis_v0.5.0/cordis_h2020/en_tr_direct/bilstm/`. Neural artifacts include configuration,
taxonomy hashes, history, provenance and thresholds in `naltra.json`. BiLSTM adds weights
and vocabulary; Transformer saves local model/tokenizer files. Smoke tracks have a
`_smoke` suffix and are not benchmark models. A `laya` artifact (built by
`scripts/prepare_laya.py`, not trained) holds the pinned checkpoint files with their
SHA-256 values, which loading verifies, and the 473 topic questions. See
[core ML operation](../docs/core_ml.md).

Load only artifacts from a trusted source. `naive_bayes`, `svm` and `hybrid_knn` store
`estimator.joblib`, which `joblib.load` unpickles, and unpickling can run code; the SHA-256
in the sibling `naltra.json` detects corruption, not tampering, because whoever can replace
the file can rewrite it too. These artifacts also refuse to load under a scikit-learn
version other than the one that saved them (1.9.1 for the shipped models).
