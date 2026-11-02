import numpy as np
import pandas as pd
import pytest

from src.config import TARGET_COL
from src.data.transformation import clean, split_data
from src.data.validation import DataValidationError, find_problems, validate


def test_load_data_renames_columns(loans):
    assert "int_rate" in loans.columns
    assert not any("." in col for col in loans.columns)


def test_sample_data_is_valid(loans):
    assert find_problems(loans) == []


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("fico", 200, "FICO"),
        ("purpose", "holiday", "Verwendungszweck"),
        (TARGET_COL, 2, "nicht binär"),
        ("installment", -1.0, "Negative Werte in installment"),
        ("dti", np.nan, "Fehlende Werte"),
    ],
)
def test_find_problems_detects_invalid_values(loans, column, value, message):
    loans.loc[0, column] = value
    assert any(message in problem for problem in find_problems(loans))


def test_validate_rejects_missing_columns(loans):
    with pytest.raises(DataValidationError, match="Spalten fehlen"):
        validate(loans.drop(columns="fico"))


def test_clean_clips_revol_util_and_drops_duplicates(loans):
    loans.loc[0, "revol_util"] = 119.0
    cleaned = clean(pd.concat([loans.head(5), loans.head(1)]))
    assert cleaned["revol_util"].max() <= 100
    assert len(cleaned) == 5


def test_split_is_stratified_60_20_20(loans):
    train, val, test = split_data(loans)
    assert len(train) + len(val) + len(test) == len(loans)
    assert len(test) == pytest.approx(0.2 * len(loans), abs=1)
    for part in (train, val, test):
        assert part[TARGET_COL].mean() == pytest.approx(loans[TARGET_COL].mean(), abs=0.01)
