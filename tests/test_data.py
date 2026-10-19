import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import clean_data, load_raw, prepare_clean_dataset, split_xy, validate_schema


def test_clean_data_has_expected_columns(clean_df):
    assert list(clean_df.columns) == config.COLUMNS


def test_clean_data_target_is_binary(clean_df):
    assert set(clean_df[config.TARGET].unique()) <= {0, 1}


def test_clean_data_converts_question_marks_to_nan(raw_df, clean_df):
    assert (raw_df[["ca", "thal"]] == "?").any().any()
    assert clean_df[["ca", "thal"]].isna().sum().sum() > 0
    assert all(pd.api.types.is_numeric_dtype(clean_df[c]) for c in config.COLUMNS)


def test_clean_data_drops_duplicates(raw_df):
    doubled = pd.concat([raw_df, raw_df.iloc[:10]], ignore_index=True)
    assert len(clean_data(doubled)) == len(clean_data(raw_df))


def test_multiclass_target_binarised():
    df = pd.DataFrame([[50, 1, 1, 120, 200, 0, 0, 150, 0, 1.0, 1, 0, 3, t]
                       for t in (0, 1, 2, 3, 4)], columns=config.COLUMNS)
    df["age"] = [50, 51, 52, 53, 54]  # make rows unique
    assert clean_data(df)[config.TARGET].tolist() == [0, 1, 1, 1, 1]


def test_validate_schema_raises_on_missing_column(clean_df):
    with pytest.raises(ValueError):
        validate_schema(clean_df.drop(columns=["chol"]))


def test_split_xy_shapes(clean_df):
    X, y = split_xy(clean_df)
    assert X.shape == (len(clean_df), len(config.FEATURES))
    assert config.TARGET not in X.columns
    assert len(y) == len(clean_df)


def test_prepare_clean_dataset_roundtrip(raw_df, tmp_path):
    raw_path = tmp_path / "raw.csv"
    out_path = tmp_path / "clean.csv"
    raw_df.to_csv(raw_path, index=False)
    cleaned = prepare_clean_dataset(raw_path, out_path)
    assert out_path.exists()
    assert pd.read_csv(out_path).shape == cleaned.shape


def test_load_raw_supports_headerless_uci_format(raw_df, tmp_path):
    path = tmp_path / "processed.cleveland.data"
    raw_df.to_csv(path, header=False, index=False)
    loaded = load_raw(path)
    assert list(loaded.columns) == config.COLUMNS
    assert len(loaded) == len(raw_df)


def test_load_raw_missing_file_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_raw(tmp_path / "nope.csv")


def test_no_infinite_values(clean_df):
    assert not np.isinf(clean_df.to_numpy(dtype=float)).any()
