# Contributing

Use a focused branch for each change and open a pull request before merging into
`main`. Include the problem, resulting behavior, checks run, and any changes to
configuration or data assumptions.

## Development setup

Follow the [quick start](README.md#quick-start) to create an environment and install
`python -m pip install -e ".[dev]"`. The package supports Python 3.11+; the pinned
experiment requirements need Python 3.12+ and were recorded with Python 3.13.1.

Run these commands from the repository root:

```bash
python -m ruff check .
python -m black --check .
python -m pytest
```

To apply formatting and automatic lint fixes:

```bash
python -m ruff check --fix .
python -m black .
```

If Make is available, `make lint`, `make test`, and `make format` run the equivalent
commands.

On Windows, run each test file in its own process and re-run a failing file once. A
process there occasionally stops with exit code 139 / 0xC0000005 and no traceback (a
native crash at varying points, for example inside transformers or scikit-learn), which
is not a test failure; one process per file confines it:

```powershell
Get-ChildItem tests\test_*.py | ForEach-Object { python -m pytest -q $_.FullName; if ($LASTEXITCODE -ne 0) { python -m pytest -q $_.FullName } }
# bash: for f in tests/test_*.py; do python -m pytest -q "$f" || python -m pytest -q "$f"; done
```

## Code and tests

Keep shared interfaces typed and modules focused. Put adjustable settings in
`configs/`, rather than machine-specific paths in code.

Add meaningful tests for behavior changes. Tests use small fixtures and must not
download datasets or pretrained models, call live APIs, or require a GPU. Preserve
the common prediction contract and complete direct-label score universe.

Use [team ownership](docs/team_ownership.md) to find reviewers. Coordinate changes
to shared schemas, taxonomy identifiers, split definitions, and evaluation semantics.

## Data and artifacts

Git stores the raw CORDIS archive, compressed release splits and manifests, split
map, translation review files, taxonomy, and small result summaries. See the
[data guide](data/README.md) and [result file guide](results/README.md).

Keep trained weights, prediction dumps, decompressed datasets, extracted CSVs,
and caches in their ignored locations. Share model releases using the
[artifact layout](models/README.md) and [reproduction checklist](docs/reproducibility.md).

Keep credentials in ignored `.env` files. `.env.example` contains only placeholders
and non-secret defaults. Review the staged diff for generated files and secrets.

Dataset changes require a new release and genuine regeneration of manifests.
Do not edit checksums to make changed data or code appear compatible.

## Before opening a pull request

- Keep the change focused and describe its effect.
- Update relevant tests and documentation.
- Run formatting, linting, and the applicable tests.
- Check `git status` and the staged diff.
- Identify any artifact, dataset, or interface compatibility changes.
