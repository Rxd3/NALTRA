# EuroSciVoc Taxonomy (v0.4.0)

Canonical scientific taxonomy for NALTRA based on the official European Science Vocabulary
(EuroSciVoc) used by the European Union CORDIS Horizon 2020 project database.

## Overview

- **Ontology**: European Science Vocabulary (EuroSciVoc)
- **Publisher**: Publications Office of the European Union / European Commission
- **Taxonomy Version**: `0.4.0`
- **Active Labels**: 586 canonical labels
  - **Supported Direct Labels**: 473 (support >= 50)
  - **Total Required Ancestors**: 225
  - **Ancestors Already Direct**: 112 (in DIRECT ∩ ANCESTORS_ALL)
  - **Newly Added Ancestors**: 113 (in ANCESTORS_ALL - DIRECT)
- **Root Domains**: 6 top-level OECD Fields of Science
- **Direct Label Support Threshold**: >= 50 projects in CORDIS H2020
- **Max Hierarchy Depth**: 7 levels

## Root Scientific Domains (Depth 1)

- **medical and health sciences** (`medical_and_health_sciences`) - EuroSciVoc code: `/21`
- **natural sciences** (`natural_sciences`) - EuroSciVoc code: `/23`
- **engineering and technology** (`engineering_and_technology`) - EuroSciVoc code: `/25`
- **agricultural sciences** (`agricultural_sciences`) - EuroSciVoc code: `/27`
- **social sciences** (`social_sciences`) - EuroSciVoc code: `/29`
- **humanities** (`humanities`) - EuroSciVoc code: `/31`

## Hierarchy Semantics

EuroSciVoc is a hierarchical ontology. The active NALTRA benchmark uses:
- **Direct Labels**: Specific EuroSciVoc categories directly assigned to each project.
- **Hierarchy-Closed Labels**: Direct labels expanded along parent paths to root domains.
- **Strict Tree Hierarchy**: Each category node has exactly one parent or is root.

## File Artifacts

- `taxonomy.json`: Canonical hierarchy definition (IDs, parents, depths, codes, paths).
- `label_map.json`: Mapping from codes, paths, and titles to canonical label IDs.
- `validation.py`: Schema, cycle and dangling-parent checks (`python taxonomy/validation.py taxonomy/taxonomy.json`).

## Rebuilding

Both JSON files are built from `euroSciVoc.csv` inside the committed raw archive
`data/raw/cordis_h2020/cordis-h2020projects-csv.zip`:

```bash
python scripts/build_taxonomy.py   # --archive <zip> --output-dir taxonomy --min-support 50
```

Support counts distinct projects per EuroSciVoc code; codes with support >= 50 become direct
labels and every code prefix above them is added as an ancestor. `taxonomy.json` is reproduced
byte for byte. `label_map.json` gets the same aliases, but its committed key order is not
reproducible, so an existing file with an unchanged mapping is left as is: dataset manifests and
trained models pin both files' SHA-256. Details: [docs/taxonomy.md](../docs/taxonomy.md#6-rebuilding-the-taxonomy).
