"""Data-Drift-Erkennung: Vorhersage-Anfragen gegen die Trainingsdaten des Champions.

python -m src.monitoring.drift   → Report ausgeben, Exit-Code 1 bei Drift.
Drift ist ein Alarm zum Nachsehen, kein automatischer Retraining-Auslöser:
Neue Labels (Ausfall ja/nein) liegen bei Krediten erst Monate später vor.
"""

import csv
import io
import logging
import sys
from datetime import datetime

import mlflow
import pandas as pd
from scipy.stats import ks_2samp

from src.config import DRIFT_ALPHA, DRIFT_MIN_SAMPLES, INPUT_COLS, NUMERIC_COLS, PREDICTION_LOG_PATH
from src.models.predict import load_champion
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)

LOG_COLUMNS = ["timestamp", "model_version", "default_probability", *INPUT_COLS]


def log_prediction(application: dict, probability: float, model_version: str) -> None:
    """Hängt eine Vorhersage an das CSV-Log an."""
    path = PREDICTION_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model_version": model_version,
                "default_probability": round(probability, 4),
                **application,
            }
        )


def load_reference(run_id: str) -> pd.DataFrame:
    """Trainingsdaten, die beim Training mit dem Modell geloggt wurden."""
    text = mlflow.artifacts.load_text(f"runs:/{run_id}/reference/train.csv")
    return pd.read_csv(io.StringIO(text))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """KS-Test je numerischem Feature. Bonferroni-Korrektur, damit 12 Tests
    nicht zufällig Fehlalarme erzeugen."""
    if len(current) < DRIFT_MIN_SAMPLES:
        return {"status": "insufficient_data", "samples": len(current), "features": {}}

    alpha = DRIFT_ALPHA / len(NUMERIC_COLS)
    features = {}
    for col in NUMERIC_COLS:
        p_value = float(ks_2samp(reference[col], current[col]).pvalue)
        features[col] = {"p_value": round(p_value, 4), "drift": p_value < alpha}

    drifted = any(f["drift"] for f in features.values())
    return {"status": "drift" if drifted else "ok", "samples": len(current), "features": features}


def current_report(run_id: str) -> dict:
    """Alle geloggten Anfragen gegen die Trainingsdaten des angegebenen Modell-Runs."""
    exists = PREDICTION_LOG_PATH.exists()
    current = pd.read_csv(PREDICTION_LOG_PATH) if exists else pd.DataFrame(columns=LOG_COLUMNS)
    return drift_report(load_reference(run_id), current)


def main() -> None:
    setup_logging()
    champion = load_champion()
    report = current_report(champion.run_id)
    for col, result in report["features"].items():
        logger.info("%-18s p=%.4f %s", col, result["p_value"], "DRIFT" if result["drift"] else "")
    logger.info("Status: %s (%d Anfragen)", report["status"], report["samples"])
    sys.exit(1 if report["status"] == "drift" else 0)


if __name__ == "__main__":
    main()
