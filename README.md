# NALTRA

**Natural Language Analysis & Taxonomy Robust Architecture** is a student research project for multilingual text classification. The repository is organized so five contributors can develop data, models, ensemble decisions, evaluation, and the user interface in parallel behind shared interfaces.

## Research goal

NALTRA compares model families under one prediction contract using CORDIS Horizon 2020 project objectives and the hierarchical EuroSciVoc taxonomy.

Supported language settings:

- English: original CORDIS objectives.
- Turkish: machine-translated versions of the same projects.
- English?Turkish code-switching: synthetic evaluation variants from aligned sentences.

Classification setting:

- Multi-label scientific topic classification.
- 473 directly supported target labels within a 586-node taxonomy (v0.4.0).
- Models train on `labels_direct`; stored `labels` contains the ancestor closure.
- Project groups and duplicate objective texts remain in one split across languages.

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

Prepare the checksum-locked CORDIS release from the repository root:

```bash
python scripts/prepare_data.py --dataset cordis_h2020 --download
python scripts/validate_datasets.py --languages en
python scripts/prepare_data.py --dataset translation --device cuda
python scripts/validate_datasets.py --languages en tr
python scripts/train_all.py --datasets cordis_h2020 --languages en tr --device auto --output-dir models/cordis_v0.4.0
```

Translation runs locally and can resume from its ignored cache. Use `--device cpu` for translation when CUDA is unavailable. To train on the audited English source before deriving Turkish, use `--languages en`. Each language selection produces a separate artifact track. Old dataset artifacts remain separate and cannot be loaded under the new taxonomy.

Optional evaluation variants use `--dataset noisy` and `--dataset code_switch`; audit the complete clean and derived release with `python scripts/validate_datasets.py --derived`. `--dataset all --download` prepares English, Turkish, noise, and both code-switch tracks. The installed `naltra-prepare-data` command uses the same implementation. CORDIS held-out-domain OOD folds require a separate protocol; historical SIB-200 folds are not CORDIS benchmark inputs.

Source locks, manifests, translation limitations, and data provenance are described in [the dataset specification](docs/dataset.md) and [data workspace](data/README.md).

## Collaboration

All work is reviewed through pull requests; direct commits to `main` are not allowed. Read [CONTRIBUTING.md](CONTRIBUTING.md) for the branching, testing, secret-handling, and review rules.
