import json

import joblib
import pandas as pd
import pytest

from src import config
from src.data import split_xy
from src.predict import HeartDiseaseModel


def test_training_writes_model_and_metadata(trained_artifacts):
    model_path, meta_path = trained_artifacts
    assert model_path.exists() and meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["model_name"] in {"logistic_regression", "random_forest"}
    assert set(meta["all_models"]) == {"logistic_regression", "random_forest"}
    assert 0.0 <= meta["cv_metrics"]["cv_roc_auc_mean"] <= 1.0


def test_model_beats_chance_on_signal_data(trained_artifacts, clean_df):
    model_path, _ = trained_artifacts
    pipe = joblib.load(model_path)
    X, y = split_xy(clean_df)
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y, pipe.predict_proba(X)[:, 1]) > 0.7


def test_saved_pipeline_is_deterministic(trained_artifacts, clean_df):
    model_path, _ = trained_artifacts
    X, _ = split_xy(clean_df)
    pipe = joblib.load(model_path)
    first = pipe.predict_proba(X.head(20))
    second = joblib.load(model_path).predict_proba(X.head(20))
    assert (first == second).all()


def test_predict_wrapper_output_contract(trained_artifacts, sample_patient):
    model = HeartDiseaseModel(*trained_artifacts)
    result = model.predict([sample_patient])[0]
    assert result["prediction"] in (0, 1)
    assert 0.5 <= result["confidence"] <= 1.0
    assert 0.0 <= result["probability_disease"] <= 1.0
    assert result["label"] in {"No heart disease", "Heart disease"}


def test_predict_wrapper_handles_missing_optional_fields(trained_artifacts, sample_patient):
    model = HeartDiseaseModel(*trained_artifacts)
    record = dict(sample_patient, ca=None, thal=None)
    assert model.predict([record])[0]["prediction"] in (0, 1)


def test_predict_wrapper_batch(trained_artifacts, clean_df):
    model = HeartDiseaseModel(*trained_artifacts)
    X, _ = split_xy(clean_df)
    records = X.head(5).to_dict(orient="records")
    assert len(model.predict(records)) == 5


def test_missing_model_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        HeartDiseaseModel(tmp_path / "missing.joblib")


def test_feature_list_in_metadata_matches_config(trained_artifacts):
    meta = json.loads(trained_artifacts[1].read_text())
    assert meta["features"] == config.FEATURES
    assert isinstance(pd.DataFrame([dict.fromkeys(meta["features"], 1)]), pd.DataFrame)
