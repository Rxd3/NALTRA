# Taxonomy

NALTRA builds its active EuroSciVoc hierarchy from category paths in the committed
CORDIS Horizon 2020 archive. The canonical files are
`taxonomy/taxonomy.json` and `taxonomy/label_map.json`.

## Active hierarchy

| Property | Value |
| --- | --- |
| Version | 0.4.0 |
| Supported direct labels | 473 |
| Total active nodes | 586 |
| Roots | 6 |
| Maximum depth | 7 |
| Structure | Single-parent hierarchy with no cycles |

The six roots form a forest of trees. Every non-root node has one parent.

| Root ID | Source code | Active descendants |
| --- | --- | --- |
| `natural_sciences` | `/23` | 218 |
| `engineering_and_technology` | `/25` | 136 |
| `medical_and_health_sciences` | `/21` | 90 |
| `social_sciences` | `/29` | 82 |
| `agricultural_sciences` | `/27` | 22 |
| `humanities` | `/31` | 32 |

## Support filtering

A directly assigned category is retained if at least **50 distinct projects**
in the full source archive carry it. This calculation precedes splitting, so it
includes projects later assigned to validation and test. It is not a training-only
support threshold and does not guarantee a particular count in each split. In the
shipped splits, every retained direct label has at least 34 train, 6 validation,
and 6 test projects.

The source has 1,053 directly assigned categories across 32,210 projects. At the
chosen threshold, 473 categories retain 31,407 projects (97.51% of labeled projects).
Their 225 required ancestors include 112 already-direct labels; adding the remaining
113 produces the 586-node active hierarchy. Ancestors are retained even when
their direct support is zero.

| Minimum support | Direct labels | Added ancestors | Active nodes | Retained projects |
| --- | --- | --- | --- | --- |
| 5 | 965 | 25 | 990 | 32,171 |
| 10 | 857 | 47 | 904 | 32,093 |
| 20 | 714 | 77 | 791 | 31,944 |
| **50** | **473** | **113** | **586** | **31,407** |
| 100 | 297 | 112 | 409 | 30,395 |

## Direct labels and ancestors

Dataset `labels_direct` stores the assigned canonical labels that pass the support
threshold. Codes below it are dropped, not rolled up to their nearest supported
ancestor: 8,128 of the 31,407 retained projects lose at least one assigned code
([dataset card, known issue 2](../data/README.md#known-issues)). Dataset `labels`
adds every ancestor. Models learn and score direct labels; the prediction pipeline
provides root-to-label `hierarchy_paths` without inventing ancestor probabilities.

Ordinary classification compares direct labels. Hierarchical metrics close both
predictions and targets upward before scoring. A hierarchy violation occurs when
a set contains a non-root node but omits its parent; the violation rate is the
fraction of records with at least one such omission. Archived target sets have
zero violations.

## Label mapping

The label map contains 1,752 aliases for source codes, full title paths, and titles.
Root titles and paths coincide. For example, a source code or title path resolves
to a canonical ID such as `data_science`.

The builder derives IDs by lowercasing titles and replacing runs of other
characters with underscores. Nodes are ordered by depth and source code;
`is_leaf` means that the active hierarchy contains no child.

## Rebuild and validate

Run from the repository root:

```bash
python scripts/build_taxonomy.py
python taxonomy/validation.py taxonomy/taxonomy.json
```

The builder reads `euroSciVoc.csv` directly from the committed ZIP, computes
distinct-project support, includes code-prefix ancestors, and writes the canonical
files. `--min-support`, `--archive`, and `--output-dir` select alternatives.

The archived taxonomy checksum is
`0dda384c528199fc0dd4f170ce2a677bb5009f569802b963b5c9cd6ed1299b84`.
Its CRLF byte representation is pinned by dataset/model manifests. The label-map
key order originated from an unseeded set; an equivalent existing mapping is left
untouched to preserve its byte checksum. Builder tests verify both identities.

Changing labels or file identity can invalidate datasets and model artifacts.
Keep the archived taxonomy for result replay and version any new hierarchy.
