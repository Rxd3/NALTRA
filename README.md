# NALTRA

**Natural Language Analysis & Taxonomy Robust Architecture** is a student research project for multilingual text classification. The repository is organized so five contributors can develop data, models, ensemble decisions, evaluation, and the user interface in parallel behind shared interfaces.

## Research goal

NALTRA compares classical, neural, transformer, and external-service approaches under the same prediction contract. The active benchmark focuses on multilingual single-label topic classification using SIB-200 across English and Turkish, studying label quality, cross-language transfer, out-of-distribution (Near-OOD) detection, confidence calibration, typographical/orthographic noise robustness, synthetic English–Turkish code-switching, and inference latency.

Supported language settings:

- English (`en`)
- Turkish (`tr`)
- English–Turkish code-switching (`en-tr`, synthetic chunk-mixing derived from aligned SIB-200 pairs)

Active benchmark setting:

- **Source Dataset**: SIB-200 (`Davlan/sib200`) as the single source dataset.
- **Classification Task**: Multilingual single-label topic classification across seven canonical topics (`science_technology`, `travel`, `politics`, `sport`, `health`, `arts_culture_entertainment_media`, `geography`).
- **Taxonomy**: Version `0.3.0`, flat canonical taxonomy where all seven topics are root categories (`parent: null`).
- **Robustness Tracks**: Clean in-distribution evaluation, typographical/orthographic noise robustness, synthetic code-switching, and SIB-200 leave-one-topic-out Near-OOD folds.
*(Note: Core pipeline utilities maintain generic capability for multi-label thresholding and hierarchical resolution for future experimental extensions, but the active benchmark task is strictly single-label topic classification on SIB-200).*

Planned model families:

- Naive Bayes
- Support Vector Machine (SVM)
- BiLSTM
- Multilingual Transformer
- Jev integration
- Laya integration
- Configurable ensemble voting across all six model families

The ensemble layer supports strict hard-majority voting, average-probability soft voting, and configurable weighted soft voting. All methods vote independently per label and return the same `PredictionResult` schema as an individual model, so downstream OOD, evaluation, and dashboard stages remain unchanged.

Evaluation covers standard classification metrics, cross-language evaluation, Near-OOD detection, confidence calibration, noise robustness, and latency.

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

The model packages currently contain interface-compatible placeholders only. No datasets or pretrained weights are downloaded by setup or by the tests.

## Collaboration

All work is reviewed through pull requests; direct commits to `main` are not allowed. Read [CONTRIBUTING.md](CONTRIBUTING.md) for the branching, testing, secret-handling, and review rules.
