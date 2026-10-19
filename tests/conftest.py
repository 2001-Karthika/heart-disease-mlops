import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import clean_data  # noqa: E402
from tests.synthetic import make_synthetic_raw  # noqa: E402


@pytest.fixture(scope="session")
def raw_df():
    return make_synthetic_raw(n=300, seed=1)


@pytest.fixture(scope="session")
def clean_df(raw_df):
    return clean_data(raw_df)


@pytest.fixture(scope="session")
def sample_patient():
    return {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
        "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
        "ca": 0, "thal": 6,
    }


@pytest.fixture(scope="session")
def trained_artifacts(clean_df, tmp_path_factory):
    """Train a tiny model once per session (no MLflow) and return its file paths."""
    from src import train as train_module

    out = tmp_path_factory.mktemp("artifacts")
    model_path = out / "model.joblib"
    meta_path = out / "model_metadata.json"
    train_module.train(use_mlflow=False, fig_dir=out / "figs", model_path=model_path,
                       metadata_path=meta_path, data=clean_df)
    return model_path, meta_path
