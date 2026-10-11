"""Compare the shipped model against the alternatives, on the validation split only.

Usage (from the repo root, after `python -m training.train`):
    python -m training.benchmark

Answers three questions with measurements rather than assertions:
* Does a gradient-boosted model beat the logistic regression, and by how much?
* Does class weighting help with the benign/phishing imbalance?
* Is isotonic calibration better than sigmoid (Platt) calibration?

The test split is never touched here, so these comparisons can inform design
choices without contaminating the held-out numbers.
"""

import json
import logging
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss

from training.calibrate import fit_platt
from training.config import load_config
from training.data.known_sites import load_known_sites
from training.data.split import _hash_fraction
from training.evaluate import rates_at, select_threshold
from training.features.url_features import featurise
from training.train import make_classifier

log = logging.getLogger("phishguard.benchmark")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def compare_calibration(scores: np.ndarray, labels: np.ndarray, fold: np.ndarray) -> dict:
    """Two-fold cross-fitted Brier score of each calibration method.

    Each method is fitted on one half of the validation sites and scored on the
    other, so neither is graded on rows it has already seen.
    """
    out = {"none": [], "sigmoid": [], "isotonic": []}
    for held_out in (False, True):
        fit, score = fold != held_out, fold == held_out
        a, b = fit_platt(scores[fit], labels[fit])
        isotonic = IsotonicRegression(out_of_bounds="clip").fit(_sigmoid(scores[fit]), labels[fit])
        out["none"].append(brier_score_loss(labels[score], _sigmoid(scores[score])))
        out["sigmoid"].append(brier_score_loss(labels[score], _sigmoid(a * scores[score] + b)))
        out["isotonic"].append(
            brier_score_loss(labels[score], isotonic.predict(_sigmoid(scores[score])))
        )
    return {method: float(np.mean(values)) for method, values in out.items()}


def run_benchmark(
    dataset: pd.DataFrame, cfg: dict[str, Any], known_sites: frozenset[str] = frozenset()
) -> dict[str, Any]:
    train, val = dataset[dataset["split"] == "train"], dataset[dataset["split"] == "val"]
    x_train = featurise(train["url"].tolist(), known_sites)
    x_val = featurise(val["url"].tolist(), known_sites)
    y_train, y_val = train["label"].to_numpy(), val["label"].to_numpy()
    target_fpr = cfg["threshold"]["target_fpr"]

    candidates = {
        "logistic_regression": make_classifier(cfg),
        "logistic_regression_class_weighted": make_classifier(cfg, class_weight="balanced"),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, random_state=cfg["seed"]
        ),
    }
    models = {}
    for name, model in candidates.items():
        probs = model.fit(x_train, y_train).predict_proba(x_val)[:, 1]
        models[name] = {
            "val_pr_auc": float(average_precision_score(y_val, probs)),
            "val_recall_at_target_fpr": rates_at(
                y_val, probs, select_threshold(y_val, probs, target_fpr)
            )["recall"],
        }

    fold = val["site"].map(lambda site: _hash_fraction(site, cfg["seed"] + 1) < 0.5).to_numpy()
    scores = candidates["logistic_regression"].decision_function(x_val)
    return {
        "shipped": "logistic_regression",
        "models": models,
        "calibration_brier": compare_calibration(scores, y_val, fold),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config()
    artifacts = Path(cfg["paths"]["artifacts"])
    dataset = pd.read_parquet(artifacts / "dataset.parquet")
    result = run_benchmark(dataset, cfg, load_known_sites(cfg))
    (artifacts / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    log.info("benchmark %s", json.dumps(result))

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    with mlflow.start_run(run_name="benchmark"):
        mlflow.log_param("model_version", cfg["model"]["version"])
        for name, scores in result["models"].items():
            mlflow.log_metrics({f"{name}.{k}": v for k, v in scores.items()})
        mlflow.log_metrics({f"brier.{k}": v for k, v in result["calibration_brier"].items()})


if __name__ == "__main__":
    main()
