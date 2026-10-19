import numpy as np
from sklearn.linear_model import LogisticRegression

from src import config
from src.data import split_xy
from src.features import build_pipeline, build_preprocessor


def test_preprocessor_output_has_no_nans(clean_df):
    X, _ = split_xy(clean_df)
    assert X.isna().sum().sum() > 0  # input really has missing values
    out = build_preprocessor().fit_transform(X)
    assert not np.isnan(out).any()


def test_preprocessor_scales_numeric_columns(clean_df):
    X, _ = split_xy(clean_df)
    pre = build_preprocessor().fit(X)
    out = pre.transform(X)
    n_num = len(config.NUMERIC_FEATURES)
    assert np.allclose(out[:, :n_num].mean(axis=0), 0, atol=1e-6)
    assert np.allclose(out[:, :n_num].std(axis=0), 1, atol=1e-6)


def test_preprocessor_one_hot_encodes_categoricals(clean_df):
    X, _ = split_xy(clean_df)
    pre = build_preprocessor().fit(X)
    names = pre.get_feature_names_out()
    assert any(n.startswith("cat__cp_") for n in names)
    assert len(names) > len(config.FEATURES)


def test_preprocessor_handles_unseen_category(clean_df):
    X, _ = split_xy(clean_df)
    pre = build_preprocessor().fit(X)
    row = X.iloc[[0]].copy()
    row["cp"] = 99.0
    assert pre.transform(row).shape[0] == 1


def test_pipeline_fit_predict(clean_df):
    X, y = split_xy(clean_df)
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1)
