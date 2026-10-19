"""Download the UCI Heart Disease dataset (Cleveland subset) into data/raw/.

Usage:
    python scripts/download_data.py

Strategy (first one that works wins):
  1. Official UCI zip:   https://archive.ics.uci.edu/static/public/45/heart+disease.zip
  2. Direct UCI file:    .../heart-disease/processed.cleveland.data
  3. ucimlrepo package:  fetch_ucirepo(id=45)   (pip install ucimlrepo)

Manual fallback: download https://archive.ics.uci.edu/dataset/45/heart+disease in a
browser, unzip, and copy processed.cleveland.data to data/raw/processed.cleveland.data.
This script (and the training code) will pick that file up automatically.
"""
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
FILE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
RAW_NAME = "processed.cleveland.data"


def _from_zip() -> pd.DataFrame:
    resp = requests.get(ZIP_URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        with zf.open(RAW_NAME) as fh:
            return pd.read_csv(fh, header=None, names=config.COLUMNS, na_values="?")


def _from_file_url() -> pd.DataFrame:
    resp = requests.get(FILE_URL, timeout=60)
    resp.raise_for_status()
    return pd.read_csv(
        io.StringIO(resp.text), header=None, names=config.COLUMNS, na_values="?"
    )


def _from_ucimlrepo() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo  # type: ignore

    ds = fetch_ucirepo(id=45)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    df.columns = config.COLUMNS
    return df


def _from_local_manual_copy() -> pd.DataFrame:
    manual = config.DATA_RAW_PATH.parent / RAW_NAME
    if not manual.exists():
        raise FileNotFoundError(manual)
    return pd.read_csv(manual, header=None, names=config.COLUMNS, na_values="?")


def main() -> int:
    out = config.DATA_RAW_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    df = None
    for name, fn in [
        ("UCI zip", _from_zip),
        ("UCI direct file", _from_file_url),
        ("ucimlrepo package", _from_ucimlrepo),
        ("manually placed processed.cleveland.data", _from_local_manual_copy),
    ]:
        try:
            df = fn()
            print(f"[ok] loaded dataset via {name}: shape={df.shape}")
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {name} failed: {exc}")
    if df is None:
        print("Could not obtain the dataset. See the manual fallback in this file's docstring.")
        return 1
    df.to_csv(out, index=False)
    print(f"Saved raw dataset to {out}")

    from src.data import prepare_clean_dataset

    clean = prepare_clean_dataset(out)
    print(f"Saved cleaned dataset to {config.DATA_CLEAN_PATH} shape={clean.shape}")
    print(clean[config.TARGET].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
