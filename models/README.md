# Model artifacts

Git tracks only this README, so a clone contains no trained model. Install the shared
benchmark artifacts at the paths below, relative to the repository root. Share complete
folders without editing their contents.

Folders follow `<run>/<dataset>/<track>/<family>/`. `scripts/train_all.py` writes new
artifacts there (Naive Bayes and SVM train on CPU, see the
[reproduction guide](../docs/reproducibility.md); the GPU families are in
[runbook section 2](../docs/runbook.md#2-train)), and `scripts/prepare_laya.py` builds
the Laya artifact.

## Required benchmark models

| Model | Folder | Files recorded in the benchmark |
| --- | --- | --- |
| Naive Bayes | `cordis_v1.1.0/cordis_h2020/en_tr_direct/naive_bayes/` | `estimator.joblib`, `naltra.json` |
| SVM | `cordis_v1.1.0/cordis_h2020/en_tr_direct/svm/` | `estimator.joblib`, `naltra.json` |
| Hybrid kNN | `cordis_v1.1.0_knn/cordis_h2020/en_tr_direct/hybrid_knn/` | `estimator.joblib`, `naltra.json` |
| BiLSTM | `cordis_v0.5.0/cordis_h2020/en_tr_direct/bilstm/` | `weights.pt`, `vocabulary.json`, `naltra.json` |
| Transformer | `cordis_v0.5.0/cordis_h2020/en_tr_direct/transformer/` | `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `naltra.json` |

All folders above belong inside `models/`. Hybrid kNN also needs
`intfloat/multilingual-e5-base` at revision
`d128750597153bb5987e10b1c3493a34e5a4502a`. Provide its complete downloaded snapshot
for offline use; the kNN artifact does not bundle that encoder.

The separate humanities-held-out Naive Bayes and SVM models belong under
`results/ood/cordis_v1.1.0/models/{naive_bayes,svm}/`, each with
`estimator.joblib` and `naltra.json`.

## Metadata and verification

`naltra.json` records resolved configuration, labels, taxonomy identity, training
metadata, thresholds, and provenance. Neural artifacts also include training history.
Retain every tokenizer, vocabulary, and configuration file alongside the weights.

The full and sample benchmark summaries record per-file model SHA-256 values under
`dumps[*].artifact_files`. Classification with `--benchmark-settings` verifies the
artifact fingerprints and taxonomy before inference.

Classical artifacts require the same scikit-learn version used to save them
(1.9.1 for the reported runs, pinned in `requirements.txt`). Only load trusted
`estimator.joblib` files: joblib deserialization can execute code. A checksum verifies
file identity, rather than whether its source is trustworthy: whoever can replace the
file can also rewrite the SHA-256 in the sibling `naltra.json`.

## Other artifacts

- Original `cordis_v0.4.0` BiLSTM and Transformer folders are needed to replay the
  classifier-head refit and investigate original training provenance.
- Smoke tracks have a `_smoke` suffix and are excluded from the benchmark.
- Kev uses a local server; Jev uses a hosted service. Their recorded predictions
  are essential for sample result replay.
- Laya is prepared, rather than trained, using `scripts/prepare_laya.py`. Its folder
  contains pinned checkpoint files, their SHA-256 values (verified on loading), and the
  473 label questions. It is scored on the fixed samples but never votes (0.068 EN /
  0.046 TR micro-F1).

Artifact download links and independent verification are pending the handoff.
See [reproducibility](../docs/reproducibility.md) for the release checklist and
[model usage](../docs/core_ml.md) for loading and inference.
