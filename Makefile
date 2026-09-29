.PHONY: install test lint format dashboard

install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	python -m ruff check .
	python -m black --check .

format:
	python -m ruff check --fix .
	python -m black .

dashboard:
	python scripts/run_dashboard.py
