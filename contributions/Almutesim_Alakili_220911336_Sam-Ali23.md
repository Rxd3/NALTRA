# Team Member Contribution Report

- **Name**: ALMUTESİM ALAKİLİ
- **Student ID**: 220911336
- **GitHub Username**: Sam-Ali23
- **Role**: Multilingual & Shift Data Lead

---

## What I Owned
I was responsible for preparing, building, and documenting our multilingual test pipelines and synthetic input shift datasets. My main focus was making sure our evaluation setup had realistic Turkish translation sets, code-switched text, and noisy data to stress-test our models under real-world conditions.

Files and folders I managed:
- `data/` (multilingual, code-switched, and noisy dataset splits)
- `docs/dataset.md` (dataset pipeline documentation)
- `docs/taxonomy.md` (taxonomy and distribution documentation)

---

## What I Did

1. **Multilingual Data Translation**
   - Integrated machine translation pipelines (Helsinki-NLP / Opus-MT) to translate English test records into Turkish.
   - Built paired, parallel evaluation sets to measure model cross-lingual generalization performance.

2. **Input Shift & Noise Generation**
   - Created synthetic sentence-level and chunk-level code-switched evaluation sets to test model performance on mixed-language inputs.
   - Generated realistic noisy test sets (introducing typographical and structural noise) to evaluate model robustness under real-world input degradation.

3. **Documentation & Verification**
   - Authored comprehensive documentation in `docs/dataset.md` and `docs/taxonomy.md` explaining multilingual translation methodologies and noise injection rules.
   - Verified that all generated shift test splits were locked with SHA-256 checksums to ensure reproducible benchmark results across the team.
