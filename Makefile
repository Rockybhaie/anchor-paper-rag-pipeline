# Convenience commands for Linux and macOS. README.md lists the same steps for Windows.
PYTHON ?= python3.11
VENV := .venv
BIN := $(VENV)/bin

.PHONY: setup data index serve test lint evaluate mlflow

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -r requirements-dev.txt
	test -f .env || cp .env.example .env

data:
	$(BIN)/python -m src.prepare_data

index:
	$(BIN)/python -m src.build_index

serve:
	$(BIN)/uvicorn src.app:app --reload

test:
	$(BIN)/pytest

lint:
	$(BIN)/ruff check .

evaluate:
	$(BIN)/python -m src.evaluate

mlflow:
	$(BIN)/mlflow ui --backend-store-uri sqlite:///mlflow.db
