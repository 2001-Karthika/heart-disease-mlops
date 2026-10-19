"""Synthetic stand-in for the UCI dataset, used ONLY by unit tests / smoke checks.

It has the same schema and value ranges as the Cleveland data so tests run without
network access. Never use it for the real experiments or the report.
"""
import numpy as np
import pandas as pd

from src import config


def make_synthetic_raw(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Raw-looking frame: multi-class target 0-4 and a few '?' placeholders."""
    rng = np.random.default_rng(seed)
    age = rng.integers(29, 78, n)
    sex = rng.integers(0, 2, n)
    cp = rng.integers(1, 5, n)
    trestbps = rng.normal(131, 17, n).round()
    chol = rng.normal(246, 51, n).round()
    fbs = (rng.random(n) < 0.15).astype(int)
    restecg = rng.choice([0, 1, 2], n, p=[0.5, 0.01, 0.49])
    thalach = rng.normal(150, 22, n).round()
    exang = (rng.random(n) < 0.33).astype(int)
    oldpeak = np.clip(rng.gamma(1.2, 0.9, n), 0, 6).round(1)
    slope = rng.integers(1, 4, n)
    ca = rng.choice([0, 1, 2, 3], n, p=[0.6, 0.2, 0.13, 0.07]).astype(float)
    thal = rng.choice([3, 6, 7], n, p=[0.55, 0.06, 0.39]).astype(float)

    logit = (
        0.04 * (age - 54) + 0.9 * sex + 0.5 * (cp == 4) - 0.03 * (thalach - 150)
        + 0.8 * exang + 0.6 * oldpeak + 0.9 * ca + 0.8 * (thal == 7) - 0.3
    )
    prob = 1 / (1 + np.exp(-logit))
    disease = rng.random(n) < prob
    target = np.where(disease, rng.integers(1, 5, n), 0)

    df = pd.DataFrame({
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol, "fbs": fbs,
        "restecg": restecg, "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
        "slope": slope, "ca": ca, "thal": thal, "target": target,
    })[config.COLUMNS]
    df = df.astype(object)
    df.loc[rng.choice(n, 4, replace=False), "ca"] = "?"
    df.loc[rng.choice(n, 2, replace=False), "thal"] = "?"
    return df
