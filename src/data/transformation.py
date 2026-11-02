"""Bereinigt gültige Rohdaten und teilt sie in Train, Validation und Test."""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TARGET_COL, TEST_SIZE, VALIDATION_SIZE


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Repariert kleine Datenfehler, statt die Zeilen zu verwerfen."""
    df = df.copy()
    df["revol_util"] = df["revol_util"].clip(upper=100)  # Auslastung über 100 % ist ein Erfassungsfehler
    return df.drop_duplicates().reset_index(drop=True)


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratifizierter 60/20/20-Split, damit jeder Teil ca. 16 % Ausfälle enthält.

    Train: Modelle fitten. Validation: Modell und Schwellenwert wählen. Test: nur Endbewertung.
    """
    train_val, test = train_test_split(df, test_size=TEST_SIZE, stratify=df[TARGET_COL], random_state=RANDOM_STATE)
    train, val = train_test_split(
        train_val,
        test_size=VALIDATION_SIZE / (1 - TEST_SIZE),
        stratify=train_val[TARGET_COL],
        random_state=RANDOM_STATE,
    )
    return train, val, test
