# NALTRA

**Natural Language Analysis & Taxonomy Robust Architecture** classifies EU research
project objectives into scientific fields using the EuroSciVoc taxonomy. It compares
individual models and ensembles on English, translated Turkish, synthetic
code-switched text, and noisy text.

The benchmark contains **31,407 projects**, **473 direct labels**, and a **586-node
hierarchy**. Projects share the same train, validation, and test assignment across
language variants.

[Documentation](docs/README.md) · [Results](docs/results.md) ·
[Interactive dashboard](docs/results/index.html) · [Reproduction guide](docs/reproducibility.md)

## Quick start

Run commands from the repository root. Python 3.11+ is supported; Python 3.13 is
recommended for the pinned experiment environment.

Create an environment:

```bash
python -m venv .venv
```

Activate it using the command for your shell:

| Shell | Command |
| --- | --- |
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Linux/macOS shell | `source .venv/bin/activate` |

Install the development package, restore the committed datasets, and validate them:

```bash
python -m pip install -e ".[dev]"
python scripts/unpack_data.py
python scripts/validate_datasets.py --prebuilt-release
python -m pytest
```

These steps need no model artifacts or API credentials. For the original experiment
dependency versions, follow the [environment instructions](docs/reproducibility.md#environment).

## Classify a project

Place the shared SVM artifact in the folder shown in the [artifact guide](models/README.md),
then run:

```bash
python scripts/classify.py --artifacts models/cordis_v1.1.0/cordis_h2020/en_tr_direct --models svm --benchmark-settings results/metrics/cordis_v1.1.0_ensemble/summary.json --text "This project develops laser sensors for environmental monitoring."
```

The output is one JSON object containing selected labels, ancestor labels, detected
language, and the out-of-distribution decision. The benchmark settings load the
published threshold and verify the artifact fingerprints before inference.

See [model usage](docs/core_ml.md) for Python examples, ensembles, batch input, and
Kev/Jev/Laya setup.

## Models

| Model | Approach | Evaluation |
| --- | --- | --- |
| Naive Bayes | TF-IDF with one classifier per label | Full benchmark; non-voting baseline |
| SVM | TF-IDF with calibrated linear SVMs | Full benchmark |
| Hybrid kNN | TF-IDF and multilingual-e5 retrieval | Full benchmark |
| BiLSTM | Learned embeddings and bidirectional LSTM | Full benchmark |
| Transformer | Fine-tuned XLM-RoBERTa | Full benchmark |
| Kev | Local zero-shot yes/no service | Fixed EN/TR samples |
| Jev | Hosted zero-shot yes/no service | Fixed EN/TR samples |
| Laya | Local zero-shot choice-question checkpoint | Fixed EN/TR samples; non-voting baseline |

Hard and soft ensembles are evaluated offline from saved prediction matrices.
Locally trained models see both EN and TR training records. Their EN-to-TR comparison
measures performance on translated input, rather than zero-shot language transfer.

## Results

Reported micro-F1 on the full test sets, with 4,711 projects per language:

| System | English | Turkish |
| --- | --- | --- |
| SVM | **0.558** | **0.489** |
| Weighted soft ensemble | 0.523 | 0.472 |
| Hybrid kNN | 0.393 | 0.382 |
| BiLSTM | 0.289 | 0.271 |
| Naive Bayes | 0.278 | 0.275 |
| Transformer | 0.248 | 0.240 |

![Full benchmark: English and Turkish micro-F1](docs/figures/cordis_v1.1.0/full_f1.png)

SVM is the strongest member across the six tested input conditions; none of the
tested ensembles beats it. Full tables, uncertainty intervals, sample comparisons,
and limitations are in [Results](docs/results.md). Machine translation, synthetic
input variants, and a single training seed limit the conclusions.

## Reproduce

The raw archive, compressed dataset splits, manifests, taxonomy, and small result
summaries are committed. Trained models and prediction dumps are distributed separately
and installed at documented paths.

The [reproduction guide](docs/reproducibility.md) distinguishes recalculating metrics,
rerunning inference, and retraining. Artifact handoff and checksum verification remain
pending. Download links will be added when the release bundles are available.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/naltra/` | Model implementations, prediction pipeline, data utilities, and evaluation |
| `scripts/` | Preparation, training, inference, evaluation, and dashboard commands |
| `configs/` | Dataset, training, benchmark, and model settings |
| `data/` | Committed dataset release and local generated variants |
| `taxonomy/` | EuroSciVoc hierarchy, label map, and validator |
| `models/` | Locally installed model artifacts |
| `results/` | Committed summaries and local prediction dumps |
| `tests/` | Offline tests using small fixtures |
| `docs/` | Guides, research methods, results, and audit history |
| `contributions/` | Individual contribution records |

For development commands and the pull request workflow, see
[Contributing](CONTRIBUTING.md). Script inputs and outputs are listed in the
[runbook](docs/runbook.md#script-reference).

## License

Code: MIT ([LICENSE](LICENSE)). Data and taxonomy: CC BY 4.0
([data/LICENSE](data/LICENSE)), derived from European Union CORDIS.
