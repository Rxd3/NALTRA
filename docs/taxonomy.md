# Taxonomy Design: European Science Vocabulary (EuroSciVoc)

The canonical NALTRA taxonomy is defined in `taxonomy/taxonomy.json`, built from the raw CORDIS archive by `scripts/build_taxonomy.py` (section 6) and validated by `taxonomy/validation.py`.

Under the CORDIS Horizon 2020 migration, NALTRA restores full hierarchical multi-label classification using the official **European Science Vocabulary (EuroSciVoc)** ontology maintained by the Publications Office of the European Union.

---

## 1. Active Taxonomy Overview

- **Taxonomy Version**: `0.4.0`
- **Ontology**: EuroSciVoc (OECD Fields of Science and Technology baseline)
- **Active Canonical Labels**: 586
- **Root Domains (Level 1)**: Exactly 6 OECD fields
- **Hierarchy Depth**: Ranges from Level 1 (roots) to Level 7 (specialized leaf subdisciplines)
- **Mathematical Structure**: Strict single-parent directed tree (no cycles, no polyhierarchy conflicts in active paths)

---

## 2. Root Domains (OECD Fields of Science)

The six top-level scientific root domains (Depth 1) are:

| Canonical ID | EuroSciVoc Code | Display Name | Depth | Active Descendants |
| :--- | :---: | :--- | :---: | :---: |
| `natural_sciences` | `/23` | Natural Sciences | 1 | 218 |
| `engineering_and_technology` | `/25` | Engineering and Technology | 1 | 136 |
| `medical_and_health_sciences` | `/21` | Medical and Health Sciences | 1 | 90 |
| `social_sciences` | `/29` | Social Sciences | 1 | 82 |
| `agricultural_sciences` | `/27` | Agricultural Sciences | 1 | 22 |
| `humanities` | `/31` | Humanities | 1 | 32 |

---

## 3. Active Label Support & Pruning Policy

The Horizon 2020 release assigns 1,053 distinct EuroSciVoc categories directly to 32,210 projects, with a long tail of rare categories (88 of them have fewer than 5 projects).

To ensure statistical significance and balanced stratified partitioning across splits:
- A minimum direct-label support threshold of **50 projects** is applied.
- 473 directly supported categories satisfy this threshold.
- All **225 required ancestors** are propagated upward to the OECD roots; 112 of them are direct labels already, so 113 are added.
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

Upstream CORDIS data references categories by slash-delimited codes (e.g. `/23/47/299`) and full title paths (e.g. `natural sciences/computer and information sciences/data science`).

The file `taxonomy/label_map.json` contains 1,752 alias mappings that resolve official numeric codes, full English path strings and category titles to deterministic canonical label IDs (e.g. `data_science`): one code, path and title per active label, where a root's path and title coincide.

---

## 6. Rebuilding the Taxonomy

`scripts/build_taxonomy.py` rebuilds both files from the committed archive, reading `euroSciVoc.csv` straight from the zip:

```bash
python scripts/build_taxonomy.py   # --archive data/raw/cordis_h2020/cordis-h2020projects-csv.zip --output-dir taxonomy --min-support 50
```

1. **Support**: `direct_project_support` is the number of distinct projects that `euroSciVoc.csv` assigns a code to directly. Every classified project has a non-empty objective in `project.csv`, so restricting the count to usable projects changes nothing.
2. **Direct labels**: codes with support >= `--min-support`.
3. **Ancestors**: every code prefix of a direct label (`/23/47/299` -> `/23/47`, `/23`), named by the matching prefix of its title path, so ancestors never annotated directly (support 0) still enter the tree.
4. **Labels**: `id` is the lowercased title with each run of other characters replaced by `_`; labels are ordered by depth, then code; `is_leaf` means no active child.
5. **Label map**: each label's code, path and title map to its `id`.

It prints a JSON status line such as `{"status": "success", "labels": 586, "direct_labels": 473, "aliases": 1752, "written": ["taxonomy.json"]}`. Other thresholds reproduce the sensitivity table in [dataset.md](dataset.md): 5, 10, 20, 50 and 100 give 965, 857, 714, 473 and 297 direct labels.

`taxonomy.json` is rebuilt byte for byte (SHA-256 `0dda384c...` in the CRLF checkout that the dataset manifests and trained models pin). `label_map.json` is rebuilt with the same 1,752 aliases, but the committed key order came from an unseeded Python set and cannot be regenerated; because the manifests and models also pin its hash, the script leaves an existing file whose mapping is unchanged untouched (`written` lists what it rewrote). `tests/test_build_taxonomy.py` checks both hashes.

---

## 7. Taxonomy Validation

To verify taxonomy schema compliance and ensure zero cycles or dangling nodes:

```bash
python taxonomy/validation.py taxonomy/taxonomy.json
```
