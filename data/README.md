# Data Workspace

This directory is owned by the Data & Taxonomy contributor. Large dataset files are intentionally ignored by Git (`.gitignore`); only documentation and `.gitkeep` placeholders are versioned.

## Directory Layout

- `raw/`: Immutable source exports and downloads (e.g. `raw/massive/`).
- `processed/`: Normalized clean records adhering to the unified schema:
  - `processed/sib200/`: 2,008 SIB-200 English and Turkish aligned pairs.
  - `processed/multifin/`: 4,974 MultiFin English and Turkish multi-label records.
  - `processed/mn_ds/`: 10,491 MN-DS English news articles with 109 categories.
  - `processed/code_switch/`: Synthetic EN/TR code-switched records (e.g. `chunk_mix/balanced/`).
- `splits/`: Alternative experiment partitions.
- `noisy/`: Deterministic perturbations preserving metadata:
  - `noisy/{strategy}/{severity}/{dataset}/`: e.g. `combined/medium/`.
- `ood/`: Out-of-Distribution evaluation benchmarks:
  - `ood/near/sib200/{heldout_topic}/`: SIB-200 7-fold leave-one-topic-out sets.
  - `ood/far/massive/`: Amazon MASSIVE non-news assistant commands (paired EN/TR).

## Data Generation

To generate all benchmark datasets locally:

```bash
python scripts/prepare_data.py --dataset all
python scripts/validate_datasets.py
```
