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

# --- Trading workflows (implemented in later phases) ---
data-sync:
	python -m src.data.polygon_client

wfo:
	python -m src.backtest.wfo $(S)

paper-trade:
	python -m src.execution.stream_manager $(S)
