# Data Workspace

This directory contains the datasets and generated benchmark data used by NALTRA.

Large dataset files are intentionally ignored by Git (`.gitignore`). Only documentation, code, and `.gitkeep` placeholders are versioned.

## Directory Layout

- `raw/`: Original dataset downloads and source files.
  - `raw/mn_ds/`: MN-DS source CSV.
  - `raw/massive/`: English and Turkish MASSIVE source files used for Far-OOD.

- `processed/`: Clean normalized records using the NALTRA unified schema.
  - `processed/sib200/`: 2,008 records forming 1,004 aligned English/Turkish pairs.
  - `processed/multifin/`: 4,974 English/Turkish MultiFin records.
  - `processed/mn_ds/`: 10,491 MN-DS English news articles across 109 fine-grained categories.
  - `processed/code_switch/`: Synthetic English/Turkish code-switched benchmark records.

- `splits/`: Alternative experiment partitions.

- `noisy/`: Deterministic noisy robustness benchmark data.
  - Example: `noisy/combined/medium/{dataset}/`

- `ood/`: Out-of-Distribution evaluation data.
  - `ood/near/sib200/{heldout_topic}/`: SIB-200 leave-one-topic-out Near-OOD folds.
  - `ood/far/massive/`: Paired English/Turkish Amazon MASSIVE Far-OOD records.

## Prerequisites

Before preparing the datasets, complete the project setup in the root `README.md`.

Make sure the project virtual environment is activated and the dependencies are installed:

```bash
python -m pip install -e ".[dev]"
