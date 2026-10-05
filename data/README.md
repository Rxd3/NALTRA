# Data Workspace

This directory contains the SIB-200 dataset and the derived benchmark data used by NALTRA.

Large dataset files are intentionally ignored by Git through `.gitignore`. Only documentation, code, and placeholder files are versioned.

## Directory Layout

- `processed/sib200/`
  - Clean normalized SIB-200 records.
  - 2,008 records total.
  - 1,004 aligned English/Turkish pairs.
  - Splits:
    - train: 1,402 records
    - validation: 198 records
    - test: 408 records

- `processed/code_switch/`
  - Synthetic English/Turkish code-switched evaluation records derived from aligned SIB-200 pairs.

- `noisy/`
  - Deterministic noisy robustness evaluation data derived from SIB-200 validation and test records.
  - Default path:
    `noisy/combined/medium/sib200/`

- `ood/near/sib200/`
  - Seven leave-one-topic-out Near-OOD folds derived from SIB-200.

## Source Dataset

NALTRA uses a single source dataset:

- **Dataset**: SIB-200
- **Hugging Face ID**: `Davlan/sib200`
- **Languages used**:
  - English
  - Turkish
- **License**: CC BY-SA 4.0

The English and Turkish records are aligned so that corresponding translations share a common `pair_id`.

## SIB-200 Topics

The project uses the following seven canonical topics:

- `science_technology`
- `travel`
- `politics`
- `sport`
- `health`
- `arts_culture_entertainment_media`
- `geography`

The original SIB-200 labels are mapped through `taxonomy/label_map.json`.

## Prerequisites

Activate the project virtual environment and install the project with development dependencies:

```bash
python -m pip install -e ".[dev]"