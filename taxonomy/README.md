# Taxonomy files

NALTRA uses EuroSciVoc taxonomy version **0.4.0**: 473 directly supported labels,
586 total nodes, six roots, and a maximum depth of seven.

| File | Purpose |
| --- | --- |
| `taxonomy.json` | Canonical labels, parents, depths, source codes, and paths |
| `label_map.json` | Source codes, titles, and paths mapped to canonical label IDs |
| `validation.py` | Checks schema, cycles, and parent references |

Models predict direct labels. Hierarchy paths add their ancestors. The 50-project
support threshold is computed over the source archive before splitting.

Validate from the repository root:

```bash
python taxonomy/validation.py taxonomy/taxonomy.json
```

Rebuild from the committed source archive:

```bash
python scripts/build_taxonomy.py
```

Rebuilding can invalidate model and dataset fingerprints. Preserve the original
release for benchmark replay. The label-map writer leaves an existing equivalent
mapping untouched because archived key order affects its checksum.

See the [taxonomy guide](../docs/taxonomy.md) for roots, support filtering,
hierarchy semantics, and rebuilding details. Licensing is covered by
[data/LICENSE](../data/LICENSE).
