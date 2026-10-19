"""Feature engineering: the preprocessing pipeline (imputation, scaling, encoding)."""
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def build_preprocessor() -> ColumnTransformer:
    """Median-impute + standardise numeric columns; mode-impute + one-hot categoricals."""
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, config.NUMERIC_FEATURES),
            ("cat", categorical, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Full pipeline = preprocessing + classifier, so raw rows can be scored directly."""
    return Pipeline(steps=[("preprocess", build_preprocessor()), ("model", estimator)])
