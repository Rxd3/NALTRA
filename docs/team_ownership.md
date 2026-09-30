# Team ownership

Ownership identifies the primary reviewer and the contributor responsible for keeping an area coherent. It does not prevent collaboration. Contributors should coordinate before changing another member's area and avoid overlapping edits whenever possible.

## joker : almutesim alakili

helps with all the tasks

## Technical Lead / Core ML - Rasit Isaoglu

Owns:

```text
src/naltra/models/bilstm/
src/naltra/models/transformer/
src/naltra/models/jev/
src/naltra/models/laya/
src/naltra/pipeline/
src/naltra/schemas/
configs/
scripts/train_all.py
```

The Technical Lead also coordinates architecture, shared prediction interfaces, integration, and final experiments.

## UI Member and Sildes - Mohmed kamaledin

Owns:

```text
dashboard/
```

## Data & Taxonomy Member - omar

Owns:

```text
data/
taxonomy/
src/naltra/data/
docs/dataset.md
docs/taxonomy.md
```

## Classical ML Member - raouf

Owns:

```text
src/naltra/models/naive_bayes/
src/naltra/models/svm/
configs/models/naive_bayes.yaml
configs/models/svm.yaml
src/naltra/pipeline/ensemble.py
configs/ensemble.yaml
```

Raouf is responsible for the ensemble voting implementation and configuration. This file-specific ownership is an exception to the Technical Lead's general ownership of `src/naltra/pipeline/`; shared pipeline contract changes still require coordination with the Technical Lead.

## Evaluation Member - Abdullah jamal

Owns:

```text
src/naltra/evaluation/
results/
docs/evaluation.md
scripts/evaluate_all.py
scripts/run_benchmark.py
```

Changes to `BaseNALTRAModel`, `PredictionResult`, taxonomy identifiers, split definitions, or evaluation semantics affect multiple owners and require explicit coordination in the pull request.
