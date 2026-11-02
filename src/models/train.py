"""Kandidaten-Modelle definieren und per Grid Search trainieren."""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import RANDOM_STATE
from src.features.feature_engineering import build_features

# Je Modell ein kleines Grid mit den 2–3 wirksamsten Hyperparametern.
# class_weight="balanced" gleicht die ca. 16 % Ausfälle aus.
CANDIDATES = {
    "logistic_regression": (
        LogisticRegression(class_weight="balanced", max_iter=1000),
        {"model__C": [0.01, 0.1, 1.0]},
    ),
    "random_forest": (
        RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        {"model__max_depth": [6, 10], "model__min_samples_leaf": [5, 20]},
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {"model__learning_rate": [0.05, 0.1], "model__max_depth": [2, 3]},
    ),
}


def build_pipeline(classifier) -> Pipeline:
    """Rohdaten rein, Wahrscheinlichkeit raus: Features, Skalierung und Modell als ein Objekt."""
    return Pipeline(
        [
            ("features", FunctionTransformer(build_features)),
            ("scaler", StandardScaler()),
            ("model", classifier),
        ]
    )


def train_candidate(name: str, X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    """5-fache Kreuzvalidierung auf den Trainingsdaten, optimiert auf ROC-AUC."""
    classifier, param_grid = CANDIDATES[name]
    search = GridSearchCV(
        build_pipeline(classifier),
        param_grid,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )
    return search.fit(X_train, y_train)
