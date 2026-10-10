# Team Member Contribution Report

- **Name**: Omar Ibrahim Fahmi Alzyod
- **Student ID**: 2409015412
- **GitHub Username**: Devinci101
- **Role**: Data & Taxonomy Lead

---

## What I Owned
I was responsible for the dataset preparation and taxonomy structure for our project. My main job was making sure our datasets were clean, properly split without data leakage, and that our science taxonomy was structured correctly for model training and hierarchical evaluation.

Files and folders I managed:
- `data/` (raw, cleaned, and split datasets)
- `taxonomy/` (EuroSciVoc tree taxonomy structure)
- `src/naltra/data/` (data loading and pre-processing modules)
- `docs/dataset.md` and `docs/taxonomy.md` (data and taxonomy documentation)

---

## What I Did

1. **In-Depth Data Cleaning**
   - Processed the raw CORDIS H2020 project dataset, cleaning text records and removing invalid or incomplete entries (filtering 35,389 raw projects down to 31,407 clean, validated records).
   - Standardized text fields and removed corrupt formatting to ensure stable input features for both classical ML and deep learning models.

2. **Multi-Label Data Splitting**
   - Created project-grouped, multi-label stratified splits for training, validation, and testing (21,985 train / 4,711 validation / 4,711 test records).
   - Ensured zero data leakage between training and testing splits while preserving original label frequency distributions across all 473 target categories.

3. **Taxonomy Setup & Filtering**
   - Structured the EuroSciVoc science taxonomy into a 586-node, depth-7 hierarchical tree rooted across 6 main science domains.
   - Mapped and filtered parent-child target categories to ensure every label had sufficient training support (minimum support threshold ≥ 50).

4. **Multilingual & Noise Datasets**
   - Prepared translated Turkish evaluation sets for cross-lingual testing.
   - Built synthetic code-switched and noisy text evaluation slices to measure how well our models perform on messy, real-world data.

5. **Documentation & Verifiability**
   - Wrote comprehensive technical guides in `docs/dataset.md` and `docs/taxonomy.md`.
   - Verified data integrity across all generated splits using SHA-256 integrity checksums for full experiment reproducibility.