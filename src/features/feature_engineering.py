"""Fachliche Risiko-Features. Läuft als erster Schritt in der Modell-Pipeline,
damit Training und API garantiert dieselbe Transformation nutzen."""

import pandas as pd

from src.config import (
    FEATURE_COLS,
    HIGH_DTI,
    HIGH_INQUIRIES,
    HIGH_REVOL_UTIL,
    PURPOSE_CATEGORIES,
    SUBPRIME_FICO,
    SUBPRIME_INT_RATE,
)


def add_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Warnsignale, wie sie ein Kreditprüfer betrachten würde; Schwellen aus der EDA."""
    df = df.copy()
    df["subprime_flag"] = ((df["fico"] < SUBPRIME_FICO) | (df["int_rate"] > SUBPRIME_INT_RATE)).astype(int)
    df["high_revol_util_flag"] = (df["revol_util"] > HIGH_REVOL_UTIL).astype(int)
    df["inquiry_flag"] = (df["inq_last_6mths"] >= HIGH_INQUIRIES).astype(int)
    df["delinquency_flag"] = ((df["delinq_2yrs"] > 0) | (df["pub_rec"] > 0)).astype(int)
    df["high_dti_flag"] = (df["dti"] > HIGH_DTI).astype(int)
    df["risk_score"] = df[
        ["subprime_flag", "high_revol_util_flag", "inquiry_flag", "delinquency_flag", "high_dti_flag"]
    ].sum(axis=1)
    return df


def encode_purpose(df: pd.DataFrame) -> pd.DataFrame:
    # Feste Kategorien: auch eine einzelne API-Anfrage erzeugt alle Dummy-Spalten
    purpose = pd.Categorical(df["purpose"], categories=PURPOSE_CATEGORIES)
    dummies = pd.get_dummies(purpose, prefix="purpose", dtype=int).drop(columns="purpose_all_other")
    dummies.index = df.index
    return pd.concat([df.drop(columns="purpose"), dummies], axis=1)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rohdaten → Feature-Matrix in fester Spaltenreihenfolge."""
    return encode_purpose(add_risk_features(df))[FEATURE_COLS]
