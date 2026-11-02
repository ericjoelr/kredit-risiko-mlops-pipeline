"""Verbindung zum MLflow-Tracking-Server mit verständlicher Fehlermeldung."""

import mlflow
import requests

from src.config import MLFLOW_TRACKING_URI


class MlflowUnavailableError(RuntimeError):
    """Der MLflow-Server antwortet nicht."""


def connect_mlflow() -> mlflow.MlflowClient:
    """Setzt die Tracking-URI. Bei einem Server wird vorher kurz geprüft, ob er erreichbar ist,
    statt MLflows minutenlange Wiederholungsversuche abzuwarten."""
    if MLFLOW_TRACKING_URI.startswith("http"):
        try:
            requests.get(f"{MLFLOW_TRACKING_URI}/health", timeout=3).raise_for_status()
        except requests.RequestException as exc:
            raise MlflowUnavailableError(
                f"MLflow-Server unter {MLFLOW_TRACKING_URI} nicht erreichbar. Starten mit: make mlflow"
            ) from exc
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.MlflowClient()
