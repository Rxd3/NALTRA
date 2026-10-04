# Data Workspace

This directory contains the datasets and generated benchmark data used by NALTRA.

Large dataset files are intentionally ignored by Git (`.gitignore`). Only documentation, code, and `.gitkeep` placeholders are versioned.

## Directory Layout

- `raw/`: Original dataset downloads and source files.
  - `raw/mn_ds/`: MN-DS source CSV (`MN-DS-news-classification.csv`).
  - `raw/massive/`: English and Turkish MASSIVE source files used for Far-OOD.

- `processed/`: Clean normalized records using the NALTRA unified schema.
  - `processed/sib200/`: 2,008 records forming 1,004 aligned English/Turkish pairs.
  - `processed/multifin/`: 4,974 English/Turkish MultiFin records (official track).
  - `processed/mn_ds/`: 10,491 MN-DS English news articles across 109 fine-grained categories.
  - `processed/code_switch/`: Synthetic English/Turkish code-switched benchmark records.

- `splits/`: Alternative experiment partitions.
  - `splits/multifin/leakage_free/`: Leakage-free MultiFin evaluation track (train: 3,183, validation: 651, test: 838).

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
```

## Data Setup Instructions

### 1. MN-DS Download and Verification

MN-DS is distributed via Zenodo and must be downloaded to `data/raw/mn_ds/`:

- **Zenodo Record**: 7394851
- **File Name**: `MN-DS-news-classification.csv`
- **Destination**: `data/raw/mn_ds/MN-DS-news-classification.csv`
- **Official Zenodo MD5**: `cffbf48aeb7de4af2d4c615a0cdb4667`

#### Local Verification Example

**Windows (PowerShell)**:
```powershell
Get-FileHash -Algorithm MD5 data/raw/mn_ds/MN-DS-news-classification.csv
# Hash should match: CFFBF48AEB7DE4AF2D4C615A0CDB4667
```

**Python**:
```python
import hashlib
from pathlib import Path

csv_path = Path("data/raw/mn_ds/MN-DS-news-classification.csv")
with open(csv_path, "rb") as f:
    digest = hashlib.md5(f.read()).hexdigest()

assert digest == "cffbf48aeb7de4af2d4c615a0cdb4667", f"Checksum mismatch: {digest}"
print("Checksum verified: OK")
```

### 2. SIB-200 & MultiFin

SIB-200 (`Davlan/sib200`) and MultiFin (`awinml/MultiFin`) are automatically downloaded from the Hugging Face Hub during preparation.

### 3. MASSIVE (Far-OOD)

Amazon MASSIVE v1.1 is downloaded from the official Amazon archive and its English/Turkish exports are verified against the SHA-256 locks in `configs/data.yaml`. Existing verified files under `data/raw/massive/` are reused.

---

## Dataset Preparation Commands

To prepare all datasets and benchmark tracks deterministically:

```bash
# Run all preparation pipelines end-to-end
python scripts/prepare_data.py --dataset all
```

Or prepare specific benchmark components individually:

```bash
# Clean SIB-200 English & Turkish
python scripts/prepare_data.py --dataset sib200

# Clean MultiFin (official track and train/evaluation leakage-free track)
python scripts/prepare_data.py --dataset multifin

# Clean MN-DS (content-grouped stratified splits)
python scripts/prepare_data.py --dataset mn_ds

# Noisy robustness benchmark (validation & test)
python scripts/prepare_data.py --dataset noisy

# Synthetic EN/TR code-switch benchmark (validation & test)
python scripts/prepare_data.py --dataset code_switch

# Out-of-Distribution benchmarks (Near-OOD 7-folds & Far-OOD MASSIVE)
python scripts/prepare_data.py --dataset ood
```

---

## Benchmark Audit and Validation

After preparation, run the comprehensive dataset audit script to verify schema compliance, partition counts, pair alignment, ID leakage, and content-fingerprint leakage across all 8 tracks:

```bash
python scripts/validate_datasets.py
```

The audit verifies manifest contents, source/code/output hashes, duplicate records, split membership and contamination. Old manifests must be regenerated. Use `data/splits/multifin/leakage_free/` for model selection and final evaluation; official MultiFin and its noisy variants retain documented upstream overlap. See [the dataset specification](../docs/dataset.md) for the overlap policy and release provenance.
