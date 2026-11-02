import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import compute_metrics, find_threshold, should_promote


@pytest.mark.parametrize(
    ("candidate", "champion", "expected"),
    [
        (0.65, None, True),  # erstes Modell über dem Minimum
        (0.55, None, False),  # erstes Modell unter dem Minimum
        (0.68, 0.66, True),  # besser als der Champion
        (0.66, 0.66, True),  # gleich gut
        (0.64, 0.66, False),  # schlechter als der Champion
        (0.58, 0.50, False),  # besser, aber unter dem Minimum
    ],
)
def test_should_promote(candidate, champion, expected):
    assert should_promote(candidate, champion) is expected


def test_find_threshold_separates_perfect_scores():
    y = pd.Series([0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
    threshold = find_threshold(y, y_prob)
    assert 0.3 < threshold <= 0.7
    assert compute_metrics(y, y_prob, threshold) == {"roc_auc": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
