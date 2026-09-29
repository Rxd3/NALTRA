# Taxonomy design

The canonical taxonomy lives in `taxonomy/taxonomy.json`. Each label has a stable machine identifier, a display name, an optional parent identifier, and a short description. Root labels use `null` as their parent.

## Change rules

1. Discuss identifier changes with the Data & Taxonomy and Technical Lead owners.
2. Add new labels without reusing or renaming existing identifiers once data is published.
3. Update `label_map.json` when a source dataset uses a different label vocabulary.
4. Run `python taxonomy/validation.py taxonomy/taxonomy.json` and the test suite.
5. Record breaking or semantic changes by incrementing the taxonomy version.

Hierarchy resolution should add required ancestors to a prediction path without inventing a score. The final scoring policy will be documented alongside experiments.
