# Source

The `naltra` package (`pip install -e .`, built by `pyproject.toml` and `setup.py`);
folder purposes are in the [repository map](../README.md#repository-map).

- `naltra/data/`: cleaning, splitting, translation, code-switch, noise, manifests, release audits
- `naltra/models/`: shared `BaseNALTRAModel` contract and one package per model
- `naltra/pipeline/`: inference (normalisation, language, thresholds, hierarchy, OOD, voting)
- `naltra/evaluation/`: metrics, calibration, OOD, bootstrap and Holm, thresholds, latency
- `naltra/reporting/`: HTML template of the results page
- `naltra/schemas/`, `naltra/utils/`: typed prediction structures; config loading, provenance,
  and resource lookup (checkout files, or the copies packaged in an installed wheel)
- `naltra/cli.py`: `prepare_data` entry point
