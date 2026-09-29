# Data workspace

This directory is owned by the Data & Taxonomy contributor. Large files are intentionally ignored by Git; only documentation and `.gitkeep` placeholders are versioned.

- `raw/`: immutable source exports, separated by source and language.
- `processed/`: normalized records ready for splitting.
- `splits/`: reproducible train, validation, and test partitions.
- `ood/`: out-of-distribution evaluation samples.
- `noisy/`: generated noise and adversarial variants.

Every dataset addition must document its source, license, collection date, language composition, label mapping, preprocessing steps, and checksum. Never place credentials or private participant data here.
