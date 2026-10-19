"""Data loading, cleaning and validation."""
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def load_raw(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw dataset.

    Accepts both our own CSV (with header, written by scripts/download_data.py)
    and the original headerless UCI file ``processed.cleveland.data``.
    """
    path = Path(path or config.DATA_RAW_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run: python scripts/download_data.py"
        )
    with open(path, "r", encoding="utf-8") as fh:
        first_line = fh.readline().strip().lower()
    has_header = first_line.startswith("age")
    if has_header:
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, header=None, names=config.COLUMNS)
    logger.info("Loaded raw data %s with shape %s", path, df.shape)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing."""
    missing = [c for c in config.COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw frame.

    * '?' placeholders -> NaN, everything coerced to numeric
    * duplicate rows dropped
    * multi-class UCI target (0-4) -> binary (0 = no disease, 1 = disease)
    Missing feature values are kept as NaN on purpose: they are imputed inside the
    modelling pipeline (fit on training folds only) to avoid data leakage.
    """
    validate_schema(df)
    out = df[config.COLUMNS].copy()
    out = out.replace("?", np.nan)
    for col in config.COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=[config.TARGET])
    out = out.drop_duplicates().reset_index(drop=True)
    out[config.TARGET] = (out[config.TARGET] > 0).astype(int)
    return out


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned frame into features and target."""
    return df[config.FEATURES].copy(), df[config.TARGET].copy()


def prepare_clean_dataset(
    raw_path: Optional[Path] = None, out_path: Optional[Path] = None
) -> pd.DataFrame:
    """Load raw data, clean it and persist the cleaned CSV."""
    out_path = Path(out_path or config.DATA_CLEAN_PATH)
    clean = clean_data(load_raw(raw_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_path, index=False)
    logger.info("Saved cleaned data to %s with shape %s", out_path, clean.shape)
    return clean


def load_clean(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the cleaned dataset, creating it from raw data if necessary."""
    path = Path(path or config.DATA_CLEAN_PATH)
    if not path.exists():
        return prepare_clean_dataset(out_path=path)
    return pd.read_csv(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = prepare_clean_dataset()
    print(data.describe(include="all").T)
