.PHONY: install lint typecheck test test-arch check data-sync wfo paper-trade

install:
	pip install -e ".[dev]"
	pre-commit install

lint:
	python3 -m ruff check src tests scripts
	python3 -m ruff format --check src tests scripts

typecheck:
	python3 -m mypy

test:
	python3 -m pytest -m "not integration"

test-arch:
	python3 -m pytest tests/test_architecture.py -v

check: lint typecheck test

# --- Trading workflows (module form: repo root must be on sys.path) ---
data-sync:
	python3 -m scripts.sync_data

wfo:
	python3 -m scripts.run_backtest --strategy $(S)

paper-trade:
	python3 -m scripts.run_paper --strategy $(S)
