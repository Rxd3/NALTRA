# NALTRA

**Natural Language Analysis & Taxonomy Robust Architecture** is a student research project for multilingual, multi-label, hierarchical text classification. The repository is organized so five contributors can develop data, models, ensemble decisions, evaluation, and the user interface in parallel behind shared interfaces.

## Research goal

NALTRA compares classical, neural, transformer, and external-service approaches under the same prediction contract. The project studies not only label quality, but also cross-language transfer, hierarchy consistency, out-of-distribution (OOD) behavior, confidence calibration, robustness, and inference latency.

Supported language settings:

- English
- Turkish
- English-Turkish code-switching

Classification setting:

- Multi-label: one text may receive multiple labels.
- Hierarchical: predictions are resolved into valid taxonomy paths.

Planned model families:

- Naive Bayes
- Support Vector Machine (SVM)
- BiLSTM
- Multilingual Transformer
- Jev integration
- Laya integration
- Configurable ensemble voting across all six model families

The ensemble layer supports strict hard-majority voting, average-probability soft voting, and configurable weighted soft voting. All methods vote independently per label and return the same `PredictionResult` schema as an individual model, so hierarchy, OOD, evaluation, and dashboard stages remain unchanged.

Evaluation covers standard classification metrics, cross-language evaluation, OOD detection, confidence calibration, robustness, and latency.

## Repository map

| Path | Purpose |
| --- | --- |
| `configs/` | Data, taxonomy, evaluation, model, and ensemble configuration |
| `data/` | Local raw/processed datasets and generated splits (large files ignored) |
| `taxonomy/` | Versioned taxonomy, label map, and validation utility |
| `src/naltra/` | Installable Python package and shared interfaces |
| `dashboard/` | Streamlit user interface owned by the UI contributor |
| `scripts/` | Thin project entry points for common workflows |
| `tests/` | Lightweight tests that require no model or dataset download |
| `models/` | Local trained artifacts (large binaries ignored) |
| `results/` | Generated metrics, plots, predictions, and benchmarks |
| `docs/` | Architecture, dataset, taxonomy, evaluation, and ownership notes |

See [team ownership](docs/team_ownership.md) before editing shared areas.

## Setup

NALTRA requires Python 3.11 or newer.

```bash
python -m venv .venv
```

Activate the environment, then install the package and development tools:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

make install
```

If `make` is unavailable, use:

```bash
python -m pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and fill in only the credentials needed for your work. Never commit `.env`.

## Common commands

```bash
make install      # install the package and developer tools
make test         # run lightweight tests
make lint         # run Ruff checks and Black's formatting check
make format       # apply Ruff fixes and Black formatting
make dashboard    # start the Streamlit dashboard
```

BiLSTM and Transformer support training, batch inference, and offline artifact reload.
The shared pipeline supports language detection, thresholds, hierarchy, OOD, and ensemble decisions.
Jev/Laya provide configurable adapters; their live services remain disabled until real provider
contracts and credentials are supplied. Naive Bayes and SVM remain placeholders.
No datasets or pretrained weights are downloaded by setup or by the tests.

See [core ML operation and integration](docs/core_ml.md) for training, artifact loading,
the prediction contract, and external setup. After preparing and auditing data:

```bash
python scripts/train_all.py --smoke --device cpu --output-dir models/smoke
python scripts/train_all.py --device auto
```

Prepare and audit the implemented benchmark datasets from the repository root:

```bash
python scripts/prepare_data.py --dataset all
python scripts/validate_datasets.py
```

MN-DS needs the verified local CSV first; see [data setup](data/README.md). Individual generators use `--dataset noisy` and `--dataset code_switch`. The installed `naltra-prepare-data` command accepts the same dataset choices. Source versions, manifests and evaluation policies are documented in [the dataset specification](docs/dataset.md).

## Collaboration

All work is reviewed through pull requests; direct commits to `main` are not allowed. Read [CONTRIBUTING.md](CONTRIBUTING.md) for the branching, testing, secret-handling, and review rules.
