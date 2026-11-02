"""Ein Kreditantrag → Ausfallwahrscheinlichkeit, Entscheidung und Log-Eintrag."""

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from src.models.predict import ChampionModel
from src.monitoring.drift import log_prediction


@dataclass
class Prediction:
    default_probability: float
    decision: Literal["approve", "decline"]
    model_version: str


def predict(model: ChampionModel, application: dict) -> Prediction:
    probability = model.predict_proba(pd.DataFrame([application]))[0]
    # Ab dem Schwellenwert aus dem Training gilt das Ausfallrisiko als zu hoch
    decision = "decline" if probability >= model.threshold else "approve"
    log_prediction(application, probability, model.version)
    return Prediction(round(probability, 4), decision, model.version)
