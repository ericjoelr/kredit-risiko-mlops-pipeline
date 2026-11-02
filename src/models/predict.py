"""Das freigegebene Modell (Alias "champion") aus der MLflow Registry laden."""

from dataclasses import dataclass
from datetime import datetime

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from sklearn.pipeline import Pipeline

from src.config import CHAMPION_ALIAS, MODEL_NAME
from src.utils.mlflow_connection import connect_mlflow


class NoChampionError(RuntimeError):
    """In der Registry ist kein Modell freigegeben."""


@dataclass
class ChampionModel:
    pipeline: Pipeline
    version: str
    run_id: str
    threshold: float
    metrics: dict[str, float]
    trained_at: datetime

    def predict_proba(self, applications: pd.DataFrame) -> list[float]:
        """Ausfallwahrscheinlichkeit je Antrag."""
        return self.pipeline.predict_proba(applications)[:, 1].tolist()


def load_champion() -> ChampionModel:
    client = connect_mlflow()
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    except MlflowException as exc:
        raise NoChampionError(
            f"Kein freigegebenes Modell '{MODEL_NAME}@{CHAMPION_ALIAS}' gefunden. Zuerst trainieren: make train"
        ) from exc

    run = client.get_run(version.run_id)
    return ChampionModel(
        pipeline=mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"),
        version=str(version.version),
        run_id=version.run_id,
        threshold=float(run.data.params["threshold"]),
        metrics={k.removeprefix("test_"): v for k, v in run.data.metrics.items() if k.startswith("test_")},
        trained_at=datetime.fromtimestamp(run.info.start_time / 1000),
    )
