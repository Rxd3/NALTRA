.PHONY: install data test lint format

install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"

data:
	python scripts/unpack_data.py

test:
	python -m pytest

lint:
	python -m ruff check .
	python -m black --check .

format:
	python -m ruff check --fix .
	python -m black .
