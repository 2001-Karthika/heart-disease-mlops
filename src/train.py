"""Train, tune, evaluate and track heart-disease classifiers.

Run:  python -m src.train            (logs to MLflow)
      python -m src.train --no-mlflow (skip tracking; useful for quick local checks)

Pipeline
  1. load cleaned data, stratified train/test split
  2. for each candidate model: GridSearchCV (5-fold, ROC-AUC) on the training split
  3. 5-fold cross-validation of the tuned model (accuracy/precision/recall/F1/ROC-AUC)
  4. hold-out test evaluation + plots (ROC, confusion matrix, feature importance)
  5. pick the best model by CV ROC-AUC (never by test score) and save it for serving
"""
import argparse
import json
import logging
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from src import config  # noqa: E402
from src.data import load_clean, split_xy  # noqa: E402
from src.features import build_pipeline  # noqa: E402

try:  # MLflow is optional at import time so unit tests stay light
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
except ImportError:  # pragma: no cover
    mlflow = None

logger = logging.getLogger("train")

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def get_candidates() -> Dict[str, Dict[str, Any]]:
    """Candidate models and their hyper-parameter grids."""
    return {
        "logistic_regression": {
            "estimator": LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE),
            "grid": {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__penalty": ["l2"],
                "model__class_weight": [None, "balanced"],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1),
            "grid": {
                "model__n_estimators": [100, 300],
                "model__max_depth": [None, 4, 8],
                "model__min_samples_leaf": [1, 3, 5],
            },
        },
    }


def test_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    return {
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_precision": precision_score(y_true, y_pred, zero_division=0),
        "test_recall": recall_score(y_true, y_pred, zero_division=0),
        "test_f1": f1_score(y_true, y_pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y_true, y_proba),
    }


test_metrics.__test__ = False  # stop pytest from collecting this helper as a test


def feature_names(pipe) -> list:
    return list(pipe.named_steps["preprocess"].get_feature_names_out())


def plot_confusion(pipe, X_test, y_test, name: str, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test, ax=ax, cmap="Blues",
                                          display_labels=["No disease", "Disease"])
    ax.set_title(f"Confusion matrix - {name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"model_confusion_{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_importance(pipe, name: str, out_dir: Path, top: int = 15) -> Path:
    model = pipe.named_steps["model"]
    names = feature_names(pipe)
    if hasattr(model, "feature_importances_"):
        values, title = model.feature_importances_, "Feature importance"
    else:
        values, title = model.coef_.ravel(), "Coefficient (standardised)"
    order = np.argsort(np.abs(values))[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(names)[order], values[order], color="#4C78A8")
    ax.set_title(f"{title} - {name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"model_importance_{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roc_comparison(results: Dict[str, Dict], X_test, y_test, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, res in results.items():
        RocCurveDisplay.from_estimator(res["pipeline"], X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="chance")
    ax.set_title("ROC curves (hold-out test set)")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "model_roc_comparison.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _log_model(pipe, X_example, name: str) -> None:
    """Log an MLflow model; works with MLflow 2.x (artifact_path) and 3.x (name)."""
    signature = infer_signature(X_example, pipe.predict_proba(X_example))
    kwargs = dict(sk_model=pipe, signature=signature, input_example=X_example.head(3),
                  serialization_format="cloudpickle")
    try:
        mlflow.sklearn.log_model(name="model", **kwargs)
    except TypeError:
        mlflow.sklearn.log_model(artifact_path="model", **kwargs)


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(config.MLFLOW_EXPERIMENT,
                                 artifact_location=config.MLFLOW_ARTIFACT_ROOT)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)


def train(use_mlflow: bool = True, fig_dir: Optional[Path] = None,
          model_path: Optional[Path] = None, metadata_path: Optional[Path] = None,
          data=None) -> Dict[str, Any]:
    fig_dir = Path(fig_dir or config.FIG_DIR)
    model_path = Path(model_path or config.MODEL_PATH)
    metadata_path = Path(metadata_path or config.METADATA_PATH)
    use_mlflow = use_mlflow and mlflow is not None
    if use_mlflow:
        setup_mlflow()

    df = load_clean() if data is None else data
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    logger.info("Train=%s Test=%s positives(train)=%.2f", X_train.shape, X_test.shape,
                y_train.mean())
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)

    parent = mlflow.start_run(run_name="model-comparison") if use_mlflow else nullcontext()
    results: Dict[str, Dict[str, Any]] = {}
    with parent:
        if use_mlflow:
            mlflow.log_params({
                "n_rows": len(df), "n_features": X.shape[1],
                "test_size": config.TEST_SIZE, "cv_folds": config.CV_FOLDS,
                "random_state": config.RANDOM_STATE,
                "sklearn_version": sklearn.__version__,
            })
            data_path = Path(config.DATA_CLEAN_PATH)
            if data_path.exists():
                mlflow.log_artifact(str(data_path), artifact_path="data")

        for name, spec in get_candidates().items():
            logger.info("=== %s ===", name)
            child = (mlflow.start_run(run_name=name, nested=True) if use_mlflow
                     else nullcontext())
            with child:
                pipe = build_pipeline(spec["estimator"])
                search = GridSearchCV(pipe, spec["grid"], scoring="roc_auc", cv=cv,
                                      n_jobs=-1, refit=True)
                search.fit(X_train, y_train)
                best = search.best_estimator_

                cv_res = cross_validate(best, X_train, y_train, cv=cv, scoring=SCORING,
                                        n_jobs=-1)
                cv_metrics = {}
                for key in SCORING:
                    vals = cv_res[f"test_{key}"]
                    cv_metrics[f"cv_{key}_mean"] = float(np.mean(vals))
                    cv_metrics[f"cv_{key}_std"] = float(np.std(vals))

                y_pred = best.predict(X_test)
                y_proba = best.predict_proba(X_test)[:, 1]
                t_metrics = test_metrics(y_test, y_pred, y_proba)

                cm_path = plot_confusion(best, X_test, y_test, name, fig_dir)
                imp_path = plot_importance(best, name, fig_dir)

                if use_mlflow:
                    mlflow.set_tag("model_type", name)
                    mlflow.log_params({k.replace("model__", ""): v
                                       for k, v in search.best_params_.items()})
                    mlflow.log_metric("grid_best_cv_roc_auc", float(search.best_score_))
                    mlflow.log_metrics(cv_metrics)
                    mlflow.log_metrics(t_metrics)
                    mlflow.log_artifact(str(cm_path), artifact_path="plots")
                    mlflow.log_artifact(str(imp_path), artifact_path="plots")
                    try:
                        _log_model(best, X_train, name)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("MLflow model logging failed: %s", exc)

                results[name] = {
                    "pipeline": best,
                    "best_params": {k.replace("model__", ""): v
                                    for k, v in search.best_params_.items()},
                    "grid_best_cv_roc_auc": float(search.best_score_),
                    "cv": cv_metrics,
                    "test": t_metrics,
                }
                logger.info("%s | CV AUC %.3f | test AUC %.3f | test recall %.3f", name,
                            cv_metrics["cv_roc_auc_mean"], t_metrics["test_roc_auc"],
                            t_metrics["test_recall"])

        roc_path = plot_roc_comparison(results, X_test, y_test, fig_dir)
        best_name = max(results, key=lambda n: results[n]["cv"]["cv_roc_auc_mean"])

        # ---- persist the winner for serving -------------------------------
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(results[best_name]["pipeline"], model_path)
        best = results[best_name]
        metadata = {
            "model_name": best_name,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__,
            "features": config.FEATURES,
            "best_params": {k: (v if v is None or isinstance(v, (int, float, str)) else str(v))
                            for k, v in best["best_params"].items()},
            "cv_metrics": best["cv"],
            "test_metrics": best["test"],
            "all_models": {n: {"cv": r["cv"], "test": r["test"],
                               "best_params": r["best_params"]}
                           for n, r in results.items()},
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        logger.info("Best model: %s -> %s", best_name, model_path)

        if use_mlflow:
            mlflow.log_artifact(str(roc_path), artifact_path="plots")
            mlflow.log_artifact(str(metadata_path), artifact_path="serving")
            mlflow.log_artifact(str(model_path), artifact_path="serving")
            mlflow.set_tag("best_model", best_name)
            mlflow.log_metric("best_cv_roc_auc", best["cv"]["cv_roc_auc_mean"])
            mlflow.log_metric("best_test_roc_auc", best["test"]["test_roc_auc"])

    return {"best_model": best_name, "results": results, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-mlflow", action="store_true", help="disable MLflow tracking")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = train(use_mlflow=not args.no_mlflow)
    print(json.dumps(out["metadata"], indent=2, default=str))


if __name__ == "__main__":
    main()
