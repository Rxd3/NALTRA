# Taxonomy

`taxonomy.json` defines the canonical flat 7-topic taxonomy for SIB-200 (version `0.3.0`). `label_map.json` maps source-specific category labels to canonical taxonomy identifiers. Keep identifiers stable after datasets or model artifacts begin using them.

Validate the taxonomy after editing it:

```bash
python taxonomy/validation.py taxonomy/taxonomy.json
```

Validation checks required fields, duplicate identifiers, missing parents, cycles, and disconnected paths.
