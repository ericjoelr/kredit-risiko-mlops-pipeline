"""Training von Daten bis Registry: python -m src.pipelines.training_pipeline --data <csv>

Exit-Code 0 = neues Modell freigegeben, 1 = abgelehnt (bleibt registriert, aber ohne Alias).
"""

import argparse
import logging
import sys
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from src.config import (
    CHAMPION_ALIAS,
    EXPERIMENT_NAME,
    INPUT_COLS,
    MIN_ROC_AUC,
    MODEL_NAME,
    RAW_DATA_PATH,
    TARGET_COL,
)
from src.data.ingestion import load_data
from src.data.transformation import clean, split_data
from src.data.validation import validate
from src.models.evaluate import compute_metrics, find_threshold, plot_roc_curve, should_promote
from src.models.train import CANDIDATES, train_candidate
from src.utils.logger import setup_logging
from src.utils.mlflow_connection import connect_mlflow

logger = logging.getLogger(__name__)

XY = tuple[pd.DataFrame, pd.Series]  # Features und Zielvariable eines Splits


def _prepare_data(data_path: Path) -> tuple[XY, XY, XY]:
    """Laden, prüfen, bereinigen, aufteilen → (X, y) für Train, Validation und Test."""
    df = load_data(data_path)
    validate(df)
    train, val, test = split_data(clean(df))
    return (
        (train[INPUT_COLS], train[TARGET_COL]),
        (val[INPUT_COLS], val[TARGET_COL]),
        (test[INPUT_COLS], test[TARGET_COL]),
    )


def _select_best_model(train: XY, val: XY) -> tuple[str, Pipeline]:
    """Jeder Kandidat als eigener Child-Run; gewählt wird nach ROC-AUC auf dem Validation-Split."""
    X_train, y_train = train
    X_val, y_val = val
    results = {}
    for name in CANDIDATES:
        with mlflow.start_run(run_name=name, nested=True):
            search = train_candidate(name, X_train, y_train)
            val_auc = roc_auc_score(y_val, search.predict_proba(X_val)[:, 1])
            mlflow.log_params(search.best_params_)
            mlflow.log_metrics({"cv_roc_auc": search.best_score_, "val_roc_auc": val_auc})
        results[name] = (val_auc, search.best_estimator_)
        logger.info("%s: Validation ROC-AUC %.3f", name, val_auc)

    best_name = max(results, key=lambda n: results[n][0])
    return best_name, results[best_name][1]


def _evaluate_on_test(name: str, model: Pipeline, val: XY, test: XY) -> dict[str, float]:
    """Schwellenwert auf Validation bestimmen, dann einmalig auf Test bewerten und loggen."""
    X_val, y_val = val
    X_test, y_test = test
    threshold = find_threshold(y_val, model.predict_proba(X_val)[:, 1])
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_prob, threshold)

    mlflow.log_params({"model_type": name, "threshold": round(threshold, 4)})
    mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
    mlflow.log_figure(plot_roc_curve(y_test, y_prob), "roc_curve.png")
    logger.info("%s: Test ROC-AUC %.3f, Schwellenwert %.2f", name, metrics["roc_auc"], threshold)
    return metrics


def _log_drift_reference(train: XY) -> None:
    """Trainingsdaten am Run speichern: Referenz für die Drift-Erkennung, versioniert mit dem Modell."""
    X_train, _ = train
    mlflow.log_text(X_train.to_csv(index=False), "reference/train.csv")


def _register_model(model: Pipeline, test: XY) -> str:
    """Neue Version in der Registry anlegen, zunächst ohne Alias."""
    X_test, _ = test
    info = mlflow.sklearn.log_model(model, name="model", registered_model_name=MODEL_NAME, input_example=X_test.head(3))
    return info.registered_model_version


def _champion_auc(client: mlflow.MlflowClient, test: XY) -> float | None:
    """ROC-AUC des aktuellen Champions auf denselben Testdaten wie der Kandidat; None beim ersten Training."""
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    except MlflowException:
        return None
    champion = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version.version}")
    X_test, y_test = test
    auc = roc_auc_score(y_test, champion.predict_proba(X_test)[:, 1])
    logger.info("Champion v%s: Test ROC-AUC %.3f", version.version, auc)
    return auc


def _promote(client: mlflow.MlflowClient, version: str) -> None:
    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version)
    client.set_model_version_tag(MODEL_NAME, version, "promotion", "accepted")
    logger.info("Version %s ist neuer Champion.", version)


def _reject(client: mlflow.MlflowClient, version: str, candidate_auc: float, champion_auc: float | None) -> None:
    if candidate_auc < MIN_ROC_AUC:
        reason = f"ROC-AUC {candidate_auc:.3f} unter Minimum {MIN_ROC_AUC}"
    else:
        reason = f"ROC-AUC {candidate_auc:.3f} schlechter als Champion ({champion_auc:.3f})"
    client.set_model_version_tag(MODEL_NAME, version, "promotion", "rejected")
    client.set_model_version_tag(MODEL_NAME, version, "reason", reason)
    logger.warning("Version %s abgelehnt: %s", version, reason)


def run_training(data_path: Path) -> bool:
    """Trainiert alle Kandidaten, registriert den besten und gibt ihn ggf. frei."""
    train, val, test = _prepare_data(data_path)
    client = connect_mlflow()
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="training"):
        mlflow.log_param("data", data_path.name)
        name, model = _select_best_model(train, val)
        metrics = _evaluate_on_test(name, model, val, test)
        _log_drift_reference(train)
        version = _register_model(model, test)
        champion_auc = _champion_auc(client, test)

    candidate_auc = metrics["roc_auc"]
    if should_promote(candidate_auc, champion_auc):
        _promote(client, version)
        return True
    _reject(client, version, candidate_auc, champion_auc)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Trainiert und registriert das Kreditrisiko-Modell.")
    parser.add_argument("--data", type=Path, default=RAW_DATA_PATH, help="CSV mit gelabelten Krediten")
    args = parser.parse_args()
    setup_logging()
    sys.exit(0 if run_training(args.data) else 1)


if __name__ == "__main__":
    main()
