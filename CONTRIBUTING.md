# Contributing to NALTRA

NALTRA is shared by six contributors (see [docs/team_ownership.md](docs/team_ownership.md)). Small, focused pull requests and clear ownership boundaries help everyone work independently.

## Git workflow

- `main` must stay stable and runnable.
- Do not commit directly to `main`.
- Create a separate branch for every feature or fix.
- Suggested branch prefixes are `feature/data-*`, `feature/ui-*`, `feature/classical-*`, `feature/evaluation-*`, and `feature/core-*`.
- Open a pull request before merging. Describe the change, tests run, and any configuration or data assumptions.
- Ask at least one teammate to review shared interfaces, schemas, taxonomy changes, and configuration changes.

## Ownership and coordination

Use [docs/team_ownership.md](docs/team_ownership.md) to identify the primary owner of a module. Ownership does not block collaboration, but avoid editing another member's module without discussing the change first. Coordinate interface changes before implementation so dependent branches do not break.

## Data, models, and secrets

- Datasets: Git tracks the raw CORDIS archive, the compressed processed splits with their manifests, the project split map `data/splits/cordis_h2020/project_splits.json` and the translation review files (see [data/README.md](data/README.md)). Keep decompressed `.jsonl` files, extracted CSVs, caches and generated copies in the ignored `data/` subdirectories.
- Do not commit large trained model weights or checkpoints. Store them outside Git and document their provenance.
- Do not commit API keys, tokens, passwords, or private endpoints.
- Put local credentials such as the Kev or Jev service keys in `.env`; `.env.example` holds only blank placeholders or non-secret local defaults.
- Before committing, review `git status` and the staged diff for generated artifacts or secrets.

## Code quality

- Target Python 3.11+ and add type hints to shared functions.
- Keep modules small and prefer readable code over premature abstraction.
- Format code with `make format` before submitting.
- Run `make lint` and `make test` before opening a pull request.
- Add or update tests whenever shared code or behavior changes.
- Tests must not download external datasets or pretrained models.
- Keep configuration in YAML or environment variables instead of hard-coding machine-specific paths.

## Pull request checklist

- The branch has a focused purpose and an ownership-appropriate name.
- Shared interfaces remain backward compatible, or affected contributors have agreed to the change.
- Tests and documentation reflect the new behavior.
- No datasets beyond the tracked release, model binaries, generated results beyond the tracked summaries ([results/README.md](results/README.md)), or secrets are included.
- Formatting, linting, and tests pass locally.
