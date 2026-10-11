# Documentation

Start with the [project README](../README.md) for installation and a classification
example. Use the guides below for a specific task.

| Guide | Use it to |
| --- | --- |
| [Model usage](core_ml.md) | Load artifacts, classify text, configure ensembles, and use external models |
| [Reproducibility](reproducibility.md) | Install shared artifacts and understand what can be reproduced |
| [Experiment runbook](runbook.md) | Run preparation, training, predictions, evaluation, and dashboard generation |
| [Results](results.md) | Read the findings, comparisons, confidence intervals, and limitations |
| [Evaluation](evaluation.md) | Understand the protocol, metric choice, significance tests, and code-switch evaluation |
| [Dataset](dataset.md) | Understand the NALTRA CORDIS EN-TR benchmark, release 1.1.0: source, record schema, splits, translation, and validation (dataset card: [data/README.md](../data/README.md)) |
| [Taxonomy](taxonomy.md) | Understand direct labels, ancestors, support filtering, and rebuilding |
| [Architecture](architecture.md) | Understand the module boundaries and prediction flow |
| [Contributing](../CONTRIBUTING.md) | Set up development checks and submit changes |
| [Team ownership](team_ownership.md) | Find the reviewer responsible for each area |
| [Audit report](audit_report.md) | Read the original audit and the fixes applied afterward |
| [Presentation](presentation/README.md) | The final deck: five slides, two hidden backups, speaker notes and their number sources |

The [interactive dashboard](results/index.html) is a self-contained HTML file.
Download it and open it in a browser, or serve it locally as described in the
[runbook](runbook.md#6-interactive-results-page). The static charts used by the
results guide are in [figures/cordis_v1.1.0](figures/cordis_v1.1.0/)
([regenerate](results.md#regenerate-the-charts)).

Directory-specific guides: [data](../data/README.md), [models](../models/README.md),
[results files](../results/README.md), [taxonomy files](../taxonomy/README.md),
[scripts](../scripts/README.md), [tests](../tests/README.md),
[configs](../configs/README.md), [source](../src/README.md), and
[contributions](../contributions/README.md).
