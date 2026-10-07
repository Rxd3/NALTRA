# Local model artifacts

Trained weights, vectorizers, checkpoints, and downloaded pretrained artifacts belong here locally and are ignored by Git. Document how an artifact was produced and share large files through the team's approved storage channel.

Neural artifacts use `<dataset>/<track>/<model>/` directories and include configuration,
taxonomy hashes, history, provenance and thresholds in `naltra.json`. BiLSTM adds weights
and vocabulary; Transformer saves local model/tokenizer files. Smoke tracks have a
`_smoke` suffix and are not benchmark models. See [core ML operation](../docs/core_ml.md).
