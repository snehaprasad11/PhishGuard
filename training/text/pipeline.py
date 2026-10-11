"""Train, evaluate and export the message-text model.

Usage (from the repo root):
    python -m training.text.pipeline

The text model reads the wording of a pasted message (an SMS or an email) and
estimates whether it is trying to manipulate the reader the way scam and
phishing messages do. It complements the URL model: a message can be
suspicious because of its link, its wording, or both.

Same discipline as the URL model: leakage-safe splits, calibration on the
validation split, a threshold chosen on validation and only applied to test,
and an export that is refused unless ONNX Runtime agrees with scikit-learn.

It also trains a TF-IDF bag-of-words model purely for comparison. That model
wins on the test split and loses badly on ordinary modern messages, and both
numbers are reported so the choice can be judged.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from training.calibrate import fit_platt, fold_calibration
from training.config import load_config
from training.evaluate import conservative_threshold, rates_at, reliability_curve, select_threshold
from training.export_onnx import INPUT_NAME, PARITY_TOLERANCE, ParityError, onnx_probabilities
from training.quantise import quantise_and_measure
from training.text.data import SPLITS, build_text_dataset, load_messages
from training.text.features import (
    MAX_CHARS,
    TEXT_FEATURE_NAMES,
    TextVectoriser,
    featurise_text,
)
from training.text.lexicon import LEXICON

log = logging.getLogger("phishguard.text")


def load_curated_messages(cfg: dict[str, Any], holdout: bool = False) -> list[tuple[str, str]]:
    """(label, text) pairs written for this project; label is "legit" or "scam"."""
    key = "curated_messages_holdout" if holdout else "curated_messages"
    lines = Path(cfg["paths"][key]).read_text("utf-8").splitlines()
    pairs = [line.split("\t", 1) for line in lines if line.strip() and not line.startswith("#")]
    return [(label, text) for label, text in pairs]


def _calibrated(raw: Pipeline, x_val: Any, y_val: np.ndarray) -> Pipeline:
    return fold_calibration(raw, *fit_platt(raw.decision_function(x_val), y_val))


def fit_text_model(dataset: pd.DataFrame, cfg: dict[str, Any]) -> Pipeline:
    """Weights from train; calibration from val."""
    train, val = dataset[dataset["split"] == "train"], dataset[dataset["split"] == "val"]
    raw = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=cfg["text_model"]["C"], max_iter=2000, random_state=cfg["seed"]),
    ).fit(featurise_text(train["text"].tolist()), train["label"].to_numpy())
    return _calibrated(raw, featurise_text(val["text"].tolist()), val["label"].to_numpy())


def fit_bag_of_words(
    dataset: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[TextVectoriser, Pipeline]:
    """The comparison model: TF-IDF over the most widespread words and word pairs."""
    model_cfg = cfg["text_model"]["bag_of_words"]
    train, val = dataset[dataset["split"] == "train"], dataset[dataset["split"] == "val"]
    vectoriser = TextVectoriser.fit(
        train["text"].tolist(), model_cfg["max_terms"], model_cfg["min_documents"]
    )
    raw = make_pipeline(
        LogisticRegression(C=model_cfg["C"], max_iter=2000, random_state=cfg["seed"])
    ).fit(vectoriser.transform(train["text"].tolist()), train["label"].to_numpy())
    calibrated = _calibrated(
        raw, vectoriser.transform(val["text"].tolist()), val["label"].to_numpy()
    )
    return vectoriser, calibrated


def _threshold(dataset: pd.DataFrame, probs_val: np.ndarray, cfg: dict[str, Any]) -> float:
    val = dataset[dataset["split"] == "val"]
    return conservative_threshold(
        val["label"].to_numpy(), probs_val, val["group"].to_numpy(),
        cfg["threshold"]["target_fpr"], cfg["threshold"]["confidence"], cfg["seed"],
    )


def text_threshold(dataset: pd.DataFrame, model: Pipeline, cfg: dict[str, Any]) -> float:
    val = dataset[dataset["split"] == "val"]
    return _threshold(dataset, model.predict_proba(featurise_text(val["text"].tolist()))[:, 1], cfg)


def _curated_check(
    score: Any, threshold: float, curated: list[tuple[str, str]]
) -> dict[str, Any]:
    probs = score([text for _, text in curated])
    legit = [(text, p) for (label, text), p in zip(curated, probs, strict=True) if label == "legit"]
    scam = [(text, p) for (label, text), p in zip(curated, probs, strict=True) if label == "scam"]
    return {
        "legitimate_total": len(legit),
        "legitimate_flagged": [
            {"text": text, "probability": float(p)} for text, p in legit if p > threshold
        ],
        "scam_total": len(scam),
        "scam_caught": int(sum(p > threshold for _, p in scam)),
        "scam_missed": [
            {"text": text, "probability": float(p)} for text, p in scam if p <= threshold
        ],
    }


def _summary(
    dataset: pd.DataFrame, score: Any, threshold: float, target_fpr: float,
    curated: dict[str, list[tuple[str, str]]],
) -> dict[str, Any]:
    """Test metrics, per-kind slices and curated checks for any scoring function."""
    test = dataset[dataset["split"] == "test"]
    y = test["label"].to_numpy()
    probs = score(test["text"].tolist())
    flagged = probs > threshold

    def slice_rates(mask: np.ndarray) -> dict[str, Any]:
        benign, scam = mask & (y == 0), mask & (y == 1)
        return {
            "legitimate": int(benign.sum()),
            "scam": int(scam.sum()),
            "fpr": float(flagged[benign].mean()) if benign.any() else None,
            "recall": float(flagged[scam].mean()) if scam.any() else None,
        }

    at_threshold = rates_at(y, probs, threshold)
    return {
        "metrics": {
            "pr_auc": float(average_precision_score(y, probs)),
            "recall_at_fpr_1pct": rates_at(
                y, probs, select_threshold(y, probs, target_fpr)
            )["recall"],
            "brier": float(brier_score_loss(y, probs)),
        },
        "threshold": {
            "target_fpr": target_fpr,
            "chosen_threshold": threshold,
            "test_fpr": at_threshold["fpr"],
            "test_recall": at_threshold["recall"],
            "test_precision": at_threshold["precision"],
        },
        "slices": {
            "sms": slice_rates((test["kind"] == "sms").to_numpy()),
            "email": slice_rates((test["kind"] == "email").to_numpy()),
        },
        "curated": {
            name: _curated_check(score, threshold, items) for name, items in curated.items()
        },
        "reliability_curve": reliability_curve(y, probs),
    }


def evaluate_text(
    dataset: pd.DataFrame, model: Pipeline, threshold: float, cfg: dict[str, Any],
    curated: dict[str, list[tuple[str, str]]],
) -> dict[str, Any]:
    def score(texts: list[str]) -> np.ndarray:
        return model.predict_proba(featurise_text(texts))[:, 1]

    counts = dataset.groupby(["split", "kind", "label"]).size()
    report = _summary(dataset, score, threshold, cfg["threshold"]["target_fpr"], curated)
    report.update({
        "model_version": cfg["text_model"]["version"],
        "rows": {
            split: {
                f"{kind}_{'scam' if label else 'legitimate'}":
                    int(counts.get((split, kind, label), 0))
                for kind in ("sms", "email") for label in (0, 1)
            }
            for split in SPLITS
        },
        "coefficients": {
            name: float(c) for name, c in zip(TEXT_FEATURE_NAMES, model[-1].coef_[0], strict=True)
        },
    })
    return report


def compare_bag_of_words(
    dataset: pd.DataFrame, cfg: dict[str, Any], curated: dict[str, list[tuple[str, str]]]
) -> dict[str, Any]:
    """The same measurements for the TF-IDF model, reduced to what the comparison needs."""
    vectoriser, model = fit_bag_of_words(dataset, cfg)

    def score(texts: list[str]) -> np.ndarray:
        return model.predict_proba(vectoriser.transform(texts))[:, 1]

    val = dataset[dataset["split"] == "val"]
    threshold = _threshold(dataset, score(val["text"].tolist()), cfg)
    summary = _summary(dataset, score, threshold, cfg["threshold"]["target_fpr"], curated)
    order = np.argsort(model[-1].coef_[0])
    return {
        "vocabulary_size": len(vectoriser.vocabulary),
        "metrics": summary["metrics"],
        "threshold": summary["threshold"],
        "curated": {
            name: {
                "legitimate_total": check["legitimate_total"],
                "legitimate_flagged": len(check["legitimate_flagged"]),
                "scam_total": check["scam_total"],
                "scam_caught": check["scam_caught"],
            }
            for name, check in summary["curated"].items()
        },
        # The words it leans on show what it actually learned.
        "strongest_scam_terms": [vectoriser.vocabulary[i] for i in order[::-1][:15]],
        "strongest_legitimate_terms": [vectoriser.vocabulary[i] for i in order[:15]],
    }


def export_text(
    model: Pipeline, threshold: float, dataset: pd.DataFrame, cfg: dict[str, Any],
    fixture_messages: list[tuple[str, str]],
) -> dict[str, Any]:
    """Write models/text-model.onnx + .json and the browser parity fixture. Returns size info."""
    onnx_bytes = to_onnx(
        model,
        initial_types=[(INPUT_NAME, FloatTensorType([None, len(TEXT_FEATURE_NAMES)]))],
        options={id(model[-1]): {"zipmap": False}},
    ).SerializeToString()

    sample = dataset["text"].sample(min(3000, len(dataset)), random_state=cfg["seed"]).tolist()
    features = featurise_text(sample + [text for _, text in fixture_messages])
    gap = float(np.max(np.abs(
        model.predict_proba(features)[:, 1] - onnx_probabilities(onnx_bytes, features)
    )))
    if not gap < PARITY_TOLERANCE:
        raise ParityError(f"text model: sklearn and ONNX differ by up to {gap:.2e}")

    quantised, quantisation = quantise_and_measure(onnx_bytes, features, INPUT_NAME)
    shipped = quantised if quantisation["shipped"] == "int8" and quantised else onnx_bytes

    scaler, classifier = model[0], model[-1]
    models = Path(cfg["paths"]["models"])
    (models / "text-model.onnx").write_bytes(shipped)
    metadata = {
        "model_version": cfg["text_model"]["version"],
        "onnx_sha256": hashlib.sha256(shipped).hexdigest(),
        "input_name": INPUT_NAME,
        "max_chars": MAX_CHARS,
        "feature_names": list(TEXT_FEATURE_NAMES),
        "lexicon": {category: list(phrases) for category, phrases in LEXICON.items()},
        # contribution of feature i = coefficient[i] * (x[i] - mean[i]) / scale[i]
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        # flag a message when its probability is above this
        "threshold": threshold,
    }
    (models / "text-model.json").write_text(json.dumps(metadata, indent=2) + "\n", "utf-8")

    fixture_features = featurise_text([text for _, text in fixture_messages])
    fixture = [
        {"label": label, "text": text, "features": row.tolist(), "probability": float(p)}
        for (label, text), row, p in zip(
            fixture_messages, fixture_features,
            model.predict_proba(fixture_features)[:, 1], strict=True,
        )
    ]
    fixture_path = Path("extension/tests/fixtures/text-parity.json")
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n", "utf-8")

    log.info("exported bytes=%d max_probability_gap=%.2e quantisation=%s",
             len(shipped), gap, quantisation["shipped"])
    return {"model_size_kb": len(shipped) / 1024, "quantisation": quantisation}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config()
    artifacts = Path(cfg["paths"]["artifacts"])
    artifacts.mkdir(parents=True, exist_ok=True)

    dataset = build_text_dataset(load_messages(cfg), cfg)
    dataset.to_parquet(artifacts / "text-dataset.parquet", index=False)
    development = load_curated_messages(cfg)
    curated = {"development": development, "holdout": load_curated_messages(cfg, holdout=True)}

    model = fit_text_model(dataset, cfg)
    threshold = text_threshold(dataset, model, cfg)
    report = evaluate_text(dataset, model, threshold, cfg, curated)
    report["bag_of_words"] = compare_bag_of_words(dataset, cfg, curated)
    exported = export_text(model, threshold, dataset, cfg, development)
    report["metrics"]["model_size_kb"] = exported["model_size_kb"]
    report["quantisation"] = exported["quantisation"]
    (artifacts / "text-eval.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")

    log.info("metrics %s", json.dumps(report["metrics"]))
    log.info("threshold %s", json.dumps(report["threshold"]))
    for name, check in report["curated"].items():
        other = report["bag_of_words"]["curated"][name]
        log.info(
            "curated %s: legit_flagged=%d/%d scam_caught=%d/%d (bag of words: %d/%d, %d/%d)",
            name, len(check["legitimate_flagged"]), check["legitimate_total"],
            check["scam_caught"], check["scam_total"], other["legitimate_flagged"],
            other["legitimate_total"], other["scam_caught"], other["scam_total"],
        )

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    with mlflow.start_run(run_name="text-model"):
        mlflow.log_params(
            {"model_version": report["model_version"], "text_C": cfg["text_model"]["C"]}
        )
        mlflow.log_metrics({f"text_test_{k}": v for k, v in report["metrics"].items()})
        mlflow.log_metrics({
            f"text_{k}": v for k, v in report["threshold"].items() if isinstance(v, float)
        })
        mlflow.log_metrics({
            f"text_bag_of_words_{k}": v for k, v in report["bag_of_words"]["metrics"].items()
        })
        mlflow.log_metric(
            "text_holdout_legit_flagged", len(report["curated"]["holdout"]["legitimate_flagged"])
        )
        mlflow.log_artifact(str(artifacts / "text-eval.json"))


if __name__ == "__main__":
    main()
