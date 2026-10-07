# Taxonomy Design: European Science Vocabulary (EuroSciVoc)

The canonical NALTRA taxonomy is defined in `taxonomy/taxonomy.json` and validated by `taxonomy/validation.py`.

Under the CORDIS Horizon 2020 migration, NALTRA restores full hierarchical multi-label classification using the official **European Science Vocabulary (EuroSciVoc)** ontology maintained by the Publications Office of the European Union.

---

## 1. Active Taxonomy Overview

- **Taxonomy Version**: `0.4.0`
- **Ontology**: EuroSciVoc (OECD Fields of Science and Technology baseline)
- **Active Canonical Labels**: 586
- **Root Domains (Level 1)**: Exactly 6 OECD fields
- **Hierarchy Depth**: Ranges from Level 1 (roots) to Level 7 (specialized leaf subdisciplines)
- **Mathematical Structure**: Strict single-parent directed tree (no cycles, no polyhierarchy conflicts in active paths)
- **Parent Prediction Requirement**: `require_parent_predictions: true` (ancestor paths accompany each child prediction)

---

## 2. Root Domains (OECD Fields of Science)

The six top-level scientific root domains (Depth 1) are:

| Canonical ID | EuroSciVoc Code | Display Name | Depth | Child Nodes in Active Set |
| :--- | :---: | :--- | :---: | :---: |
| `natural_sciences` | `/23` | Natural Sciences | 1 | 248 |
| `engineering_and_technology` | `/25` | Engineering and Technology | 1 | 185 |
| `medical_and_health_sciences` | `/21` | Medical and Health Sciences | 1 | 78 |
| `social_sciences` | `/29` | Social Sciences | 1 | 42 |
| `agricultural_sciences` | `/27` | Agricultural Sciences | 1 | 18 |
| `humanities` | `/31` | Humanities | 1 | 15 |

---

## 3. Active Label Support & Pruning Policy

EuroSciVoc contains over 1,000 leaf categories in the full Horizon 2020 distribution, including a long tail of very rare categories (< 5 instances).

To ensure statistical significance and balanced stratified partitioning across splits:
- A minimum direct-label support threshold of **50 instances** is applied.
- 473 directly supported categories satisfy this threshold.
- All **225 required ancestors** are propagated upward to the OECD roots.
- Total active canonical label set comprises **586 categories**.
- Direct labels retain **97.51%** of all labeled projects (31,407 out of 32,210).

Ancestors are **never pruned** merely because they are rarely annotated directly; structural integrity from root to leaf is strictly preserved.

---

## 4. Hierarchy Closure & Violations

Stored dataset targets preserve both sets:
1. `labels_direct`: Specific EuroSciVoc categories directly assigned to the project.
2. `labels`: Direct labels plus all required ancestors up to the root domain.

Models train on `labels_direct`. Prediction `labels` contains direct selections, and `hierarchy_paths` contains required ancestors without invented scores. Ordinary metrics compare direct assignments; hierarchical metrics expand ancestors.

### Hierarchy Violation Metric
A prediction set $P$ violates hierarchy consistency if it predicts child label $c$ but fails to predict its canonical parent $p = \text{parent}(c)$:
$$\text{violation}(P) = \sum_{c \in P} \mathbb{I}(\text{parent}(c) \neq \text{None} \land \text{parent}(c) \notin P)$$
$$\text{Hierarchy Violation Rate} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{violation}(P_i) > 0)$$

In clean benchmark target labels, the hierarchy violation rate is strictly **0.00%**.

---

## 5. Source-to-Canonical Label Mapping

Upstream CORDIS data references categories by slash-delimited codes (e.g. `/23/43/253/751`) and full title paths (e.g. `natural sciences/physical sciences/nuclear physics/nuclear fusion`).

The file `taxonomy/label_map.json` contains 1,752 alias mappings that resolve both official numeric codes and full English path strings to deterministic canonical label IDs (e.g. `nuclear_fusion`).

---

## 6. Taxonomy Validation

To verify taxonomy schema compliance and ensure zero cycles or dangling nodes:

```bash
python taxonomy/validation.py taxonomy/taxonomy.json
```
