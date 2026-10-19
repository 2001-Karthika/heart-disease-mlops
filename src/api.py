"""FastAPI service: /predict, /health, /metrics.

Run locally:  uvicorn src.api:app --reload --port 8000
Docs (Swagger UI): http://localhost:8000/docs
"""
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge,
                               Histogram, generate_latest)
from pydantic import BaseModel, Field, field_validator

from src import __version__
from src.logging_config import setup_logging
from src.predict import HeartDiseaseModel

logger = logging.getLogger("heart-api")


class PatientFeatures(BaseModel):
    """Raw patient measurements (UCI Heart Disease feature names)."""

    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=50, le=300, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=30, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment (1-3)")
    ca: Optional[float] = Field(None, ge=0, le=4, description="Major vessels (0-3), optional")
    thal: Optional[int] = Field(None, description="Thalassemia 3/6/7, optional")

    @field_validator("thal")
    @classmethod
    def _check_thal(cls, v):
        if v is not None and v not in (3, 6, 7):
            raise ValueError("thal must be one of 3, 6, 7")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
                "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
                "ca": 0, "thal": 6,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    probability_disease: float
    model_name: str
    model_version: str

    model_config = {"protected_namespaces": ()}


def create_app(model_path: Optional[Path] = None) -> FastAPI:
    setup_logging()
    registry = CollectorRegistry()
    req_count = Counter("api_requests_total", "Total HTTP requests",
                        ["method", "endpoint", "status"], registry=registry)
    req_latency = Histogram("api_request_latency_seconds", "Request latency in seconds",
                            ["endpoint"], registry=registry)
    pred_count = Counter("predictions_total", "Predictions by class", ["label"],
                         registry=registry)
    pred_conf = Histogram("prediction_confidence", "Confidence of predictions",
                          buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0], registry=registry)
    model_loaded = Gauge("model_loaded", "1 if the model is loaded", registry=registry)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.model = HeartDiseaseModel(model_path)
            model_loaded.set(1)
            logger.info("Model loaded: %s", app.state.model.name)
        except FileNotFoundError as exc:
            app.state.model = None
            model_loaded.set(0)
            logger.error("Model could not be loaded: %s", exc)
        yield

    app = FastAPI(title="Heart Disease Risk API", version=__version__,
                  description="Predicts heart disease risk from patient health data.",
                  lifespan=lifespan)
    app.state.model = None

    @app.middleware("http")
    async def log_and_measure(request: Request, call_next):
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            elapsed = time.perf_counter() - start
            route = request.scope.get("route")
            endpoint = getattr(route, "path", request.url.path)
            req_count.labels(request.method, endpoint, str(status)).inc()
            req_latency.labels(endpoint).observe(elapsed)
            logger.info(json.dumps({
                "event": "request", "method": request.method, "path": request.url.path,
                "status": status, "latency_ms": round(elapsed * 1000, 2),
                "client": request.client.host if request.client else None,
            }))

    @app.get("/")
    def root():
        return {"service": "heart-disease-api", "version": __version__,
                "docs": "/docs", "predict": "POST /predict", "health": "/health",
                "metrics": "/metrics"}

    @app.get("/health")
    def health():
        loaded = app.state.model is not None
        return {"status": "ok" if loaded else "degraded", "model_loaded": loaded,
                "model_name": app.state.model.name if loaded else None}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(features: PatientFeatures):
        model = app.state.model
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        record = features.model_dump()
        result = model.predict([record])[0]
        pred_count.labels(result["label"]).inc()
        pred_conf.observe(result["confidence"])
        logger.info(json.dumps({"event": "prediction", "input": record, "output": result}))
        return PredictionResponse(**result, model_name=model.name, model_version=__version__)

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
