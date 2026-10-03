# Taxonomy design

The canonical taxonomy lives in `taxonomy/taxonomy.json`. Each label has a stable machine identifier, a display name, an optional parent identifier, and a short description. Root labels use `null` as their parent.

## Change rules

1. Discuss identifier changes with the Data & Taxonomy and Technical Lead owners.
2. Add new labels without reusing or renaming existing identifiers once data is published.
3. Update `label_map.json` when a source dataset uses a different label vocabulary.
4. Run `python taxonomy/validation.py taxonomy/taxonomy.json` and the test suite.
5. Record breaking or semantic changes by incrementing the taxonomy version.

Hierarchy resolution should add required ancestors to a prediction path without inventing a score. The final scoring policy will be documented alongside experiments.

## Current taxonomy

Version `0.2.0` replaces the initial scaffold with the first dataset-backed
NALTRA taxonomy.

The taxonomy combines labels from the three selected benchmark datasets:

- MN-DS provides the broad English news hierarchy and 109 fine-grained topics.
- MultiFin contributes 23 finance, business, technology, health, and related
  fine-grained topics.
- SIB-200 contributes mappings for its seven multilingual topic labels.

Source dataset labels are mapped to stable NALTRA identifiers through
`taxonomy/label_map.json`.

MN-DS level-2 labels are generated reproducibly with
`scripts/add_mn_ds_taxonomy.py`. The script verifies that every fine-grained
label has one parent and prevents conflicting label identifiers.

The current taxonomy contains 151 canonical labels.

Dataset-specific labels should not be forced into unrelated categories.
Mappings are only added when the source meaning matches an existing canonical
label or when a suitable new child label is created.
