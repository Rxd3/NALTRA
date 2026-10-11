# Data files

This directory contains the CORDIS Horizon 2020 source archive and NALTRA's prepared
English, Turkish, and synthetic code-switch tracks. Dataset methods, record fields,
and limitations are documented in the [dataset guide](../docs/dataset.md).
Attribution and licensing are in [LICENSE](LICENSE).

## Directory layout

| Path | Contents | In Git |
| --- | --- | --- |
| `raw/cordis_h2020/cordis-h2020projects-csv.zip` | Original source archive | Yes |
| `processed/cordis_h2020/en/` | English train, validation, and test splits | Compressed files and manifest |
| `processed/cordis_h2020/tr/` | Turkish versions of the same projects | Compressed files and manifest |
| `processed/cordis_h2020/code_switch/{sentence_mix,chunk_mix}/balanced/` | Synthetic mixed-language validation/test splits | Compressed files and manifests |
| `splits/cordis_h2020/project_splits.json` | Shared project-to-split assignment | Yes |
| `review/` | Translation review sample and AI-assisted review | Yes |
| `noisy/` | Generated noisy test copies | No |
| `samples/{val80,test120}/` | Fixed EN/TR evaluation samples | No |
| `cache/` | Translation caches | No |

Extracted CSVs and decompressed `.jsonl` files also stay local.

## Restore the committed release

Run from the repository root:

```bash
python scripts/unpack_data.py
python scripts/validate_datasets.py --prebuilt-release
```

Unpacking verifies each split's checksum and size before replacing its local
`.jsonl`. Matching files are skipped.

English and Turkish each have 21,985 training, 4,711 validation, and 4,711 test
projects. Variants preserve project/pair identity, split assignment, and labels.

## Generate variants or rebuild data

Commands for noisy copies, samples, and a new dataset release are in the
[runbook](../docs/runbook.md#0-data). A rebuild changes tracked manifests and may
change the split map or review sample; use a separate checkout and a new release
version.

The shipped Turkish 1.1.0 release includes a targeted repair of 598 records.
A standard translation run does not recreate that repair. The repair implementation
is part of the [outstanding provenance handoff](../docs/reproducibility.md#training-and-data-provenance).

## Translation review status

Human review remains pending. The 100-row review sample and its AI-assisted audit
predate Turkish 1.1.0: 82 translations match the current release, while the other
18 refer to repaired projects and match neither archived release. All review rows
are validation projects. The `top_level_domain` column stores the first direct
label, rather than a root domain. See [translation review](../docs/dataset.md#translation-review).
