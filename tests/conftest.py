"""Gemeinsame Fixtures. Alle Tests laufen gegen ein temporäres MLflow, ohne Netzwerk und ohne .env."""

import os
import tempfile
from pathlib import Path

# Muss vor dem ersten Import von src.config gesetzt sein
_MLFLOW_DIR = Path(tempfile.mkdtemp())
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{_MLFLOW_DIR / 'mlflow.db'}"

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI, RAW_DATA_PATH  # noqa: E402
from src.data.ingestion import load_data  # noqa: E402
from src.pipelines import training_pipeline  # noqa: E402


@pytest.fixture
def loans() -> pd.DataFrame:
    """Echte Beispieldaten mit vereinheitlichten Spaltennamen."""
    return load_data(RAW_DATA_PATH)


@pytest.fixture
def application() -> dict:
    """Ein gültiger Kreditantrag, wie ihn die API erwartet."""
    return {
        "credit_policy": 1,
        "purpose": "debt_consolidation",
        "int_rate": 0.12,
        "installment": 300.0,
        "log_annual_inc": 11.0,
        "dti": 15.0,
        "fico": 710,
        "days_with_cr_line": 4500.0,
        "revol_bal": 12000.0,
        "revol_util": 45.0,
        "inq_last_6mths": 1,
        "delinq_2yrs": 0,
        "pub_rec": 0,
    }


@pytest.fixture(scope="session")
def sample_csv(tmp_path_factory) -> Path:
    """Stratifizierte Stichprobe, damit das Mini-Training nur wenige Sekunden dauert."""
    df = pd.read_csv(RAW_DATA_PATH)
    sample = df.groupby("not.fully.paid").sample(frac=0.2, random_state=0)
    path = tmp_path_factory.mktemp("data") / "sample.csv"
    sample.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def champion(sample_csv) -> bool:
    """Einmal pro Testlauf echt trainieren, danach gibt es einen Champion in der Registry."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=(_MLFLOW_DIR / "artifacts").as_uri())
    return training_pipeline.run_training(sample_csv)


@pytest.fixture(autouse=True)
def prediction_log(tmp_path, monkeypatch) -> Path:
    """Jeder Test schreibt Vorhersagen in ein eigenes, leeres Log."""
    path = tmp_path / "predictions.csv"
    monkeypatch.setattr("src.monitoring.drift.PREDICTION_LOG_PATH", path)
    return path
