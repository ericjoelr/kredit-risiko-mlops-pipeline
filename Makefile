.PHONY: install mlflow train monitor run test lint

# Optionale Overrides aus .env (siehe .env.example)
-include .env
export

install:
	uv sync

# Tracking-Server mit Registry; Datenbank und Artefakte liegen in mlruns/
mlflow:
	uv run mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db \
		--artifacts-destination mlruns/artifacts --host 127.0.0.1 --port 5050

train:
	uv run python -m src.pipelines.training_pipeline --data data/raw/loan_data.csv

monitor:
	uv run python -m src.monitoring.drift

# Startet API und Frontend in einem Terminal; Ctrl+C beendet beide.
# `kill 0` beendet die ganze Prozessgruppe, damit die API das Frontend nicht überlebt.
run:
	@trap 'kill 0' EXIT; \
	uv run uvicorn src.api.app:app --port 8000 & \
	uv run streamlit run frontend/app.py --server.port 8501

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
