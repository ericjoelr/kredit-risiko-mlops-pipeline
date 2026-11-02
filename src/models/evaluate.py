"""Metriken, Entscheidungs-Schwellenwert und Freigaberegel."""

import matplotlib

matplotlib.use("Agg")  # Plots nur als Datei, kein Fenster

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import (
    RocCurveDisplay,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import MIN_ROC_AUC


def find_threshold(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """Schwellenwert mit maximalem F1-Score. Wird auf dem Validation-Split bestimmt, nie auf Test."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])  # letzter Punkt der Kurve hat keinen Schwellenwert


def compute_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def should_promote(candidate_auc: float, champion_auc: float | None) -> bool:
    """Neues Modell wird nur freigegeben, wenn es die Mindestqualität erreicht
    und nicht schlechter ist als das aktuelle Champion-Modell."""
    if candidate_auc < MIN_ROC_AUC:
        return False
    return champion_auc is None or candidate_auc >= champion_auc


def plot_roc_curve(y_true: pd.Series, y_prob: np.ndarray) -> Figure:
    display = RocCurveDisplay.from_predictions(y_true, y_prob)
    return display.figure_
