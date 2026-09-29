# Taxonomy

`taxonomy.json` is the canonical hierarchical label definition. `label_map.json` maps source-specific labels to canonical taxonomy identifiers. Keep identifiers stable after datasets or model artifacts begin using them.

Validate the taxonomy after editing it:

```bash
python taxonomy/validation.py taxonomy/taxonomy.json
```

Validation checks required fields, duplicate identifiers, missing parents, cycles, and disconnected paths.
