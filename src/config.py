"""Central configuration: paths, column definitions, constants."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- Paths (can be overridden with environment variables) -------------------
DATA_RAW_PATH = Path(os.getenv("HEART_RAW_PATH", ROOT / "data" / "raw" / "heart_disease_raw.csv"))
DATA_CLEAN_PATH = Path(
    os.getenv("HEART_CLEAN_PATH", ROOT / "data" / "processed" / "heart_clean.csv")
)
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "model.joblib"))
METADATA_PATH = Path(os.getenv("MODEL_METADATA_PATH", ROOT / "models" / "model_metadata.json"))
FIG_DIR = Path(os.getenv("FIG_DIR", ROOT / "reports" / "figures"))
REPORTS_DIR = ROOT / "reports"
LOG_DIR = Path(os.getenv("LOG_DIR", ROOT / "logs"))

# ---- MLflow -----------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
MLFLOW_ARTIFACT_ROOT = (ROOT / "mlartifacts").as_uri()
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "heart-disease-classification")

# ---- Dataset schema (UCI Heart Disease / Cleveland subset) ------------------
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]
TARGET = "target"
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type (1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true)",
    "restecg": "Resting ECG result (0 normal, 1 ST-T abnormality, 2 LV hypertrophy)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina (1 = yes)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of peak-exercise ST segment (1 up, 2 flat, 3 down)",
    "ca": "Number of major vessels (0-3) coloured by fluoroscopy",
    "thal": "Thalassemia (3 normal, 6 fixed defect, 7 reversible defect)",
    "target": "1 = heart disease present, 0 = absent",
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
