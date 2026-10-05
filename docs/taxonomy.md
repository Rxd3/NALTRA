# Taxonomy Design

The canonical NALTRA taxonomy is stored in `taxonomy/taxonomy.json`.

NALTRA currently uses SIB-200 as its single source dataset, so the project taxonomy is a flat seven-topic taxonomy derived directly from the SIB-200 topic set.

## Current Taxonomy

Taxonomy version: `0.3.0`

The seven canonical labels are:

| Canonical ID | Display Name | SIB-200 Source Label |
| --- | --- | --- |
| `science_technology` | Science and Technology | `science/technology` |
| `travel` | Travel | `travel` |
| `politics` | Politics | `politics` |
| `sport` | Sport | `sports` |
| `health` | Health | `health` |
| `arts_culture_entertainment_media` | Arts, Culture, Entertainment and Media | `entertainment` |
| `geography` | Geography | `geography` |

The source-to-canonical mapping is stored in:

```text
taxonomy/label_map.json