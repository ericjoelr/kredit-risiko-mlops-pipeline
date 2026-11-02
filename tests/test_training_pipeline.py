import mlflow
import pandas as pd

from src.config import CHAMPION_ALIAS, MODEL_NAME
from src.models.predict import load_champion
from src.pipelines import training_pipeline


def test_first_training_creates_champion(champion):
    assert champion is True
    model = load_champion()
    assert 0 < model.threshold < 1
    assert model.metrics["roc_auc"] >= 0.6
    tags = mlflow.MlflowClient().get_model_version(MODEL_NAME, model.version).tags
    assert tags["promotion"] == "accepted"


def test_worse_model_is_rejected(champion, sample_csv, tmp_path):
    # Zufällige Labels → ROC-AUC ≈ 0.5, liegt unter dem Minimum
    df = pd.read_csv(sample_csv)
    df["not.fully.paid"] = df["not.fully.paid"].sample(frac=1, random_state=1).to_numpy()
    shuffled = tmp_path / "shuffled.csv"
    df.to_csv(shuffled, index=False)
    before = load_champion().version

    assert training_pipeline.run_training(shuffled) is False

    client = mlflow.MlflowClient()
    assert str(client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS).version) == before
    latest = max(client.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda v: int(v.version))
    assert latest.tags["promotion"] == "rejected"
