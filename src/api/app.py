"""REST-API für das freigegebene Modell: uvicorn src.api.app:app"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from src.api.schemas import DriftReport, LoanApplication, ModelInfo, PredictionResponse
from src.config import MODEL_NAME
from src.models.predict import load_champion
from src.monitoring.drift import current_report
from src.pipelines.inference_pipeline import predict
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    # Ohne freigegebenes Modell startet die API nicht (kein stiller Fallback)
    app.state.model = load_champion()
    logger.info("Modell %s v%s geladen", MODEL_NAME, app.state.model.version)
    yield


app = FastAPI(title="Credit Risk API", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s → %d (%.0f ms)", request.method, request.url.path, response.status_code, duration_ms)
    return response


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info")
def model_info(request: Request) -> ModelInfo:
    model = request.app.state.model
    return ModelInfo(
        name=MODEL_NAME,
        version=model.version,
        model_type=type(model.pipeline.named_steps["model"]).__name__,
        threshold=model.threshold,
        test_metrics=model.metrics,
        trained_at=model.trained_at,
    )


@app.post("/predict")
def predict_default(application: LoanApplication, request: Request) -> PredictionResponse:
    result = predict(request.app.state.model, application.model_dump())
    return PredictionResponse(
        default_probability=result.default_probability, decision=result.decision, model_version=result.model_version
    )


@app.get("/drift")
def drift(request: Request) -> DriftReport:
    return DriftReport(**current_report(request.app.state.model.run_id))
