# Data Workspace

This directory manages the CORDIS Horizon 2020 source data and derived multilingual benchmark tracks used by NALTRA.

Large raw files, processed corpora, and translation caches are ignored by Git through `.gitignore`. Only documentation, schema definitions, and `.gitkeep` directory anchors are tracked.

## Directory Layout

```
data/
├── raw/
│   ├── cordis_h2020/            # Official EU open data release
│   │   ├── cordis-h2020projects-csv.zip
│   │   ├── project.csv          # 35,389 project records
│   │   └── euroSciVoc.csv       # EuroSciVoc category paths & project assignments
│   └── euroscivoc/
│       └── euroSciVoc.csv       # Snapshot copy of official EuroSciVoc paths
├── processed/
│   └── cordis_h2020/
│       ├── en/                  # Canonical English scientific classification benchmark
│       │   ├── train.jsonl      # 21,985 projects (70.0%)
│       │   ├── validation.jsonl # 4,711 projects (15.0%)
│       │   ├── test.jsonl       # 4,711 projects (15.0%)
│       │   └── manifest.json    # SHA-256 provenance & parameter manifest
│       ├── tr/                  # Derived Turkish machine-translated records
│       └── code_switch/         # Derived sentence-aligned EN/TR code-switched records
├── splits/
│   └── cordis_h2020/
│       └── project_splits.json  # Pre-translation project-level split assignment map
├── cache/
│   └── translations/
│       └── cordis_h2020/        # Resumable SHA-256 translation cache
└── noisy/
    └── combined/medium/
        └── cordis_h2020/en/     # Typographical/orthographic noise robustness evaluation splits
```

## Source Dataset

NALTRA strictly uses **one source dataset**:

- **Title**: CORDIS - EU research projects under Horizon 2020 (2014-2020)
- **Publisher**: European Commission / Publications Office of the European Union
- **Canonical URL**: `https://cordis.europa.eu/data/cordis-h2020projects-csv.zip`
- **License**: CC BY 4.0 (reuse under European Union open data principles)
- **Primary Text Field**: `objective` (project summary text)
- **Ontology**: European Science Vocabulary (EuroSciVoc), 6 OECD root fields, 586 active categories (v0.4.0)

## Derived Multilingual Variants

1. **English (`en`)**: Original source text from CORDIS project objectives.
2. **Turkish (`tr`)**: Reproducibly translated variant derived from the identical English project text.
3. **Code-Switching (`en-tr`)**: Controlled sentence-aligned mixed variants derived from aligned English and Turkish sentence pairs of the identical projects.

All derived records preserve:
- Source project ID (`project_id`) and pair ID (`pair_id = cordis:<project_id>`)
- Split assignment (`train`, `validation`, `test`)
- Direct EuroSciVoc labels (`labels_direct`) and hierarchy closure (`labels`)

## Data Integrity Guarantees

- **Zero Project Leakage**: Stratification is computed strictly on source project IDs *before* any language derivation occurs.
- **Zero Content Leakage**: Projects with identical normalized English descriptions are grouped by SHA-256 content fingerprints so identical text never crosses train, validation, or test splits.
- **Zero Label Mutation**: Machine translation affects only textual content; canonical target labels and hierarchical ontology structures remain identical across language variants.
## Prepare, audit, and train

From the repository root, run:

```powershell
python scripts/prepare_data.py --dataset cordis_h2020 --download
python scripts/validate_datasets.py --languages en
python scripts/prepare_data.py --dataset translation --device cuda
python scripts/validate_datasets.py --languages en tr
python scripts/train_all.py --datasets cordis_h2020 --languages en tr --device auto --output-dir models/cordis_v0.4.0
```

The English-only training selection is `--languages en`. The installed `naltra-prepare-data` command has the same options. `--dataset all --download` additionally produces English noise and both code-switch tracks; `validate_datasets.py --derived` audits them.

Manifests use schema 1.1.0 and verify source, output, generation-code/configuration, taxonomy, and split-map hashes. Relevant generation-code changes require genuine regeneration. Translation caches include checkpoint revision and implementation identity; legacy cache entries are not reused silently. Do not commit raw/processed data, translation caches, or model weights. Human translation review remains pending until reviewers complete the sample.
