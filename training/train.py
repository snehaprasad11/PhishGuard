"""Train and calibrate the URL model.

Usage (from the repo root):
    python -m training.train

Reads the pinned data, builds leakage-safe splits, fits a logistic regression
on the train split, calibrates it on the validation split, and logs everything
to MLflow. The test split is written out but never looked at here — that is
`evaluate.py`'s job.
"""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from training.calibrate import fit_platt, fold_calibration
from training.config import load_config
from training.data.known_sites import build_known_sites
from training.data.loaders import load_urls
from training.data.split import SPLITS, build_dataset
from training.features.url_features import FEATURE_NAMES, featurise

log = logging.getLogger("phishguard.train")


def make_classifier(cfg: dict[str, Any], class_weight: str | None = None) -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=cfg["model"]["C"], class_weight=class_weight, max_iter=1000,
            random_state=cfg["seed"],
        ),
    )


def fit_model(
    dataset: pd.DataFrame, cfg: dict[str, Any], known_sites: frozenset[str] = frozenset()
) -> tuple[Pipeline, dict[str, float]]:
    """Fit on train, calibrate on val. Returns the calibrated pipeline and validation metrics."""
    train, val = dataset[dataset["split"] == "train"], dataset[dataset["split"] == "val"]
    x_train = featurise(train["url"].tolist(), known_sites)
    x_val = featurise(val["url"].tolist(), known_sites)
    y_train, y_val = train["label"].to_numpy(), val["label"].to_numpy()

    raw = make_classifier(cfg).fit(x_train, y_train)

    a, b = fit_platt(raw.decision_function(x_val), y_val)
    calibrated = fold_calibration(raw, a, b)

    metrics = {
        "val_pr_auc": average_precision_score(y_val, calibrated.predict_proba(x_val)[:, 1]),
        "val_brier_uncalibrated": brier_score_loss(y_val, raw.predict_proba(x_val)[:, 1]),
        # Optimistic: measured on the rows the calibration was fitted to. The
        # honest number is the test-split Brier from evaluate.py.
        "val_brier_calibrated": brier_score_loss(y_val, calibrated.predict_proba(x_val)[:, 1]),
        "platt_a": a,
        "platt_b": b,
    }
    return calibrated, metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config()
    np.random.seed(cfg["seed"])
    artifacts = Path(cfg["paths"]["artifacts"])
    artifacts.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(load_urls(cfg), cfg)
    counts = dataset.groupby(["split", "label"]).size()
    for split in SPLITS:
        log.info("split=%s benign=%d phishing=%d", split, counts[split, 0], counts[split, 1])

    known_sites = build_known_sites(cfg)
    log.info("known_sites=%d untrusted_hosts=%d", len(known_sites),
             sum(entry.startswith("!") for entry in known_sites))
    model, metrics = fit_model(dataset, cfg, known_sites)
    log.info("metrics %s", json.dumps({k: round(v, 5) for k, v in metrics.items()}))

    dataset.to_parquet(artifacts / "dataset.parquet", index=False)
    joblib.dump(model, artifacts / "url-model.joblib")

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    with mlflow.start_run(run_name="train"):
        mlflow.log_params({
            "model_version": cfg["model"]["version"], "seed": cfg["seed"],
            "C": cfg["model"]["C"], **cfg["split"],
        })
        mlflow.log_params({f"rows_{s}_{'phishing' if y else 'benign'}": int(n)
                           for (s, y), n in counts.items()})
        mlflow.log_metrics(metrics)
        mlflow.log_dict({"features": list(FEATURE_NAMES)}, "features.json")
        mlflow.log_artifact(cfg["paths"]["manifest"])  # exact data this model came from
        mlflow.log_artifact("config.yaml")
        mlflow.log_artifact(cfg["paths"]["known_sites"])
        mlflow.log_artifact(str(artifacts / "url-model.joblib"))


if __name__ == "__main__":
    main()
