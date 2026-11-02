from src.config import INPUT_COLS, TARGET_COL
from src.monitoring.drift import drift_report


def test_same_distribution_has_no_drift(loans):
    reference = loans.sample(n=3000, random_state=0)
    current = loans.drop(reference.index).sample(n=500, random_state=0)
    assert drift_report(reference[INPUT_COLS], current[INPUT_COLS])["status"] == "ok"


def test_shifted_feature_is_detected(loans):
    reference = loans.sample(n=3000, random_state=0)
    current = loans.drop(reference.index).sample(n=500, random_state=0).copy()
    current["int_rate"] += 0.05  # Zinsniveau steigt
    report = drift_report(reference, current)
    assert report["status"] == "drift"
    assert report["features"]["int_rate"]["drift"] is True
    assert report["features"]["fico"]["drift"] is False


def test_too_few_samples(loans):
    report = drift_report(loans, loans.head(5).drop(columns=TARGET_COL))
    assert report == {"status": "insufficient_data", "samples": 5, "features": {}}
