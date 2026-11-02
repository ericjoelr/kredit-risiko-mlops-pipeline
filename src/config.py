"""Zentrale Konfiguration: Pfade, MLflow, Spalten und fachliche Schwellenwerte."""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Pfade ─────────────────────────────────────────────────────────────────────
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "loan_data.csv"
PREDICTION_LOG_PATH = ROOT_DIR / "data" / "predictions" / "predictions.csv"

# ── MLflow ────────────────────────────────────────────────────────────────────
# Tracking-Server (make mlflow bzw. Docker-Service "mlflow"); in Docker http://mlflow:5050.
# Port 5050, weil macOS Port 5000 für AirPlay belegt
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
EXPERIMENT_NAME = "credit-risk"
MODEL_NAME = "credit-risk-model"
CHAMPION_ALIAS = "champion"

# ── Spalten ───────────────────────────────────────────────────────────────────
TARGET_COL = "not_fully_paid"  # 1 = Kredit ausgefallen

NUMERIC_COLS = [
    "credit_policy",
    "int_rate",
    "installment",
    "log_annual_inc",
    "dti",
    "fico",
    "days_with_cr_line",
    "revol_bal",
    "revol_util",
    "inq_last_6mths",
    "delinq_2yrs",
    "pub_rec",
]
INPUT_COLS = ["purpose", *NUMERIC_COLS]  # Rohdaten, die das Modell erwartet

# "all_other" ist die Referenzkategorie und bekommt keine eigene Dummy-Spalte
PURPOSE_CATEGORIES = [
    "all_other",
    "credit_card",
    "debt_consolidation",
    "educational",
    "home_improvement",
    "major_purchase",
    "small_business",
]

# Warnsignale aus der EDA (notebooks/01_eda.ipynb)
ENGINEERED_COLS = [
    "subprime_flag",
    "high_revol_util_flag",
    "inquiry_flag",
    "delinquency_flag",
    "high_dti_flag",
    "risk_score",
]
PURPOSE_DUMMY_COLS = [f"purpose_{p}" for p in PURPOSE_CATEGORIES[1:]]

# Feste Spaltenreihenfolge, die das Modell nach dem Feature-Engineering sieht
FEATURE_COLS = NUMERIC_COLS + ENGINEERED_COLS + PURPOSE_DUMMY_COLS

# ── Feature-Engineering ───────────────────────────────────────────────────────
SUBPRIME_FICO = 680  # FICO darunter gilt als Subprime
SUBPRIME_INT_RATE = 0.13
HIGH_REVOL_UTIL = 60  # Kreditlinie zu mehr als 60 % ausgeschöpft
HIGH_INQUIRIES = 3  # Kreditanfragen in 6 Monaten
HIGH_DTI = 25  # EDA: deutlicher Anstieg der Ausfallquote erst ab 25

# ── Training und Freigabe ─────────────────────────────────────────────────────
VALIDATION_SIZE = 0.2
TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_ROC_AUC = 0.60  # Mindestqualität für die Freigabe

# ── Monitoring ────────────────────────────────────────────────────────────────
DRIFT_ALPHA = 0.05  # Signifikanzniveau des KS-Tests (über alle Features)
DRIFT_MIN_SAMPLES = 30  # darunter ist der KS-Test nicht aussagekräftig
