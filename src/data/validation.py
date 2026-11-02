"""Prüft Rohdaten vor dem Training. Was hier scheitert, wird abgelehnt statt repariert."""

import pandas as pd

from src.config import INPUT_COLS, PURPOSE_CATEGORIES, TARGET_COL

NON_NEGATIVE_COLS = ["installment", "dti", "revol_bal", "revol_util", "inq_last_6mths", "delinq_2yrs", "pub_rec"]


class DataValidationError(ValueError):
    """Die Daten sind für das Training unbrauchbar."""


def find_problems(df: pd.DataFrame) -> list[str]:
    """Gibt alle gefundenen Probleme als lesbare Liste zurück (leer = gültig)."""
    missing_cols = set([*INPUT_COLS, TARGET_COL]) - set(df.columns)
    if missing_cols:
        return [f"Spalten fehlen: {sorted(missing_cols)}"]

    problems = []
    if df.isna().any().any():
        problems.append(f"Fehlende Werte in: {df.columns[df.isna().any()].tolist()}")
    if not df[TARGET_COL].isin([0, 1]).all():
        problems.append(f"{TARGET_COL} ist nicht binär")
    unknown = set(df["purpose"]) - set(PURPOSE_CATEGORIES)
    if unknown:
        problems.append(f"Unbekannter Verwendungszweck: {sorted(unknown)}")
    if not df["fico"].between(300, 850).all():
        problems.append("FICO außerhalb von 300–850")
    for col in NON_NEGATIVE_COLS:
        if (df[col] < 0).any():
            problems.append(f"Negative Werte in {col}")
    return problems


def validate(df: pd.DataFrame) -> None:
    """Bricht mit einer Liste aller Probleme ab, wenn die Daten ungültig sind."""
    problems = find_problems(df)
    if problems:
        raise DataValidationError("Ungültige Trainingsdaten:\n- " + "\n- ".join(problems))
