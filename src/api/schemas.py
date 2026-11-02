"""Request- und Response-Modelle der API. Pydantic prüft Typen und Wertebereiche."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class LoanApplication(BaseModel):
    credit_policy: Literal[0, 1] = Field(description="1 = erfüllt die Kreditrichtlinien")
    purpose: Literal[
        "all_other",
        "credit_card",
        "debt_consolidation",
        "educational",
        "home_improvement",
        "major_purchase",
        "small_business",
    ]
    int_rate: float = Field(gt=0, lt=1, description="Zinssatz als Anteil, 0.11 = 11 %")
    installment: float = Field(ge=0, description="Monatliche Rate in USD")
    log_annual_inc: float = Field(gt=0, description="Natürlicher Log des Jahreseinkommens")
    dti: float = Field(ge=0, description="Schulden im Verhältnis zum Einkommen")
    fico: int = Field(ge=300, le=850)
    days_with_cr_line: float = Field(ge=0, description="Tage seit der ersten Kreditlinie")
    revol_bal: float = Field(ge=0, description="Offener revolvierender Saldo in USD")
    revol_util: float = Field(ge=0, le=100, description="Auslastung der Kreditlinie in %")
    inq_last_6mths: int = Field(ge=0, description="Kreditanfragen der letzten 6 Monate")
    delinq_2yrs: int = Field(ge=0, description="Zahlungsverzüge (30+ Tage) in 2 Jahren")
    pub_rec: int = Field(ge=0, description="Negative öffentliche Einträge")


class PredictionResponse(BaseModel):
    default_probability: float
    decision: Literal["approve", "decline"]
    model_version: str


class ModelInfo(BaseModel):
    name: str
    version: str
    model_type: str
    threshold: float
    test_metrics: dict[str, float]
    trained_at: datetime


class FeatureDrift(BaseModel):
    p_value: float
    drift: bool


class DriftReport(BaseModel):
    status: Literal["ok", "drift", "insufficient_data"]
    samples: int
    features: dict[str, FeatureDrift]
