"""Rohdaten laden."""

from pathlib import Path

import pandas as pd


def load_data(path: str | Path) -> pd.DataFrame:
    """Liest die CSV und vereinheitlicht die Spaltennamen (int.rate → int_rate)."""
    df = pd.read_csv(path)
    return df.rename(columns=lambda col: col.replace(".", "_"))
