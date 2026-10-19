"""Inference helper: load the saved pipeline and score raw patient records."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
import sklearn

from src import config

logger = logging.getLogger(__name__)

LABELS = {0: "No heart disease", 1: "Heart disease"}


class HeartDiseaseModel:
    """Thin wrapper around the persisted sklearn Pipeline (preprocessing + classifier)."""

    def __init__(self, model_path: Optional[Path] = None,
                 metadata_path: Optional[Path] = None) -> None:
        self.model_path = Path(model_path or config.MODEL_PATH)
        self.metadata_path = Path(metadata_path or config.METADATA_PATH)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}. Run: python -m src.train"
            )
        self.pipeline = joblib.load(self.model_path)
        self.metadata: Dict[str, Any] = {}
        if self.metadata_path.exists():
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            trained_with = self.metadata.get("sklearn_version")
            if trained_with and trained_with != sklearn.__version__:
                logger.warning("Model trained with scikit-learn %s but running %s",
                               trained_with, sklearn.__version__)

    @property
    def name(self) -> str:
        return self.metadata.get("model_name", "unknown")

    def predict(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score a list of raw records (dicts with the 13 feature names)."""
        frame = pd.DataFrame(records).reindex(columns=config.FEATURES)
        frame = frame.apply(pd.to_numeric, errors="coerce").astype(float)
        proba = self.pipeline.predict_proba(frame)
        classes = list(self.pipeline.classes_)
        out = []
        for row in proba:
            p_disease = float(row[classes.index(1)])
            pred = int(p_disease >= 0.5)
            out.append({
                "prediction": pred,
                "label": LABELS[pred],
                "confidence": round(p_disease if pred == 1 else 1.0 - p_disease, 4),
                "probability_disease": round(p_disease, 4),
            })
        return out
