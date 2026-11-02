import pandas as pd

from src.config import FEATURE_COLS, INPUT_COLS, PURPOSE_DUMMY_COLS
from src.features.feature_engineering import build_features


def test_build_features_returns_fixed_columns(loans):
    features = build_features(loans[INPUT_COLS])
    assert list(features.columns) == FEATURE_COLS
    assert not features.isna().any().any()


def test_single_application_gets_all_purpose_columns(application):
    features = build_features(pd.DataFrame([application]))
    assert list(features.columns) == FEATURE_COLS
    assert features["purpose_debt_consolidation"].iloc[0] == 1


def test_reference_category_has_no_dummy(application):
    features = build_features(pd.DataFrame([{**application, "purpose": "all_other"}]))
    assert features[PURPOSE_DUMMY_COLS].sum(axis=1).iloc[0] == 0


def test_risk_flags(application):
    risky = {**application, "fico": 650, "revol_util": 90.0, "inq_last_6mths": 4, "pub_rec": 1, "dti": 28.0}
    features = build_features(pd.DataFrame([application, risky]))
    assert features["risk_score"].tolist() == [0, 5]
