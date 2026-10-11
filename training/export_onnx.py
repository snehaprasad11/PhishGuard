"""Export the trained model to ONNX and prove the export is faithful.

Usage (from the repo root, after `python -m training.train`):
    python -m training.export_onnx

Writes three files for the browser:
* `models/url-model.onnx`  — the scaler + calibrated logistic regression as one graph.
* `models/url-model.json`  — everything the extension needs besides the graph: feature order,
  decision threshold, and the scaler/coefficient values used to build the evidence panel.
  The file is always written whole, so it can never be left without a threshold.
* `extension/tests/fixtures/url-parity.json` — URLs with the features and probabilities Python
  computed for them, so the TypeScript featuriser can be tested against the same answers.

The export is refused if ONNX Runtime and scikit-learn disagree by more than
`PARITY_TOLERANCE` on any sampled URL: a model that scores differently in the
browser than it did in evaluation is worse than no model.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.pipeline import Pipeline

from training.config import load_config
from training.data.known_sites import load_known_sites
from training.data.loaders import load_curated_benign
from training.evaluate import validation_threshold
from training.features.url_features import FEATURE_NAMES, SUSPICIOUS_KEYWORDS, featurise
from training.quantise import quantise_and_measure

log = logging.getLogger("phishguard.export")

PARITY_TOLERANCE = 1e-4
INPUT_NAME = "features"

# Hand-picked to exercise every branch of the featuriser in the TypeScript parity test.
EDGE_CASE_URLS = (
    "http://192.168.0.1/login",
    "http://[2001:db8::1]:8080/a",
    "HTTPS://WWW.Secure-Login.paypa1.example.shop/verify/account-update?id=42&x=1#top",
    "http://paypal.com@evil.example/",
    "http://xn--pypal-4ve.com/",
    "http://pаypal.com/",
    "https://example.co.uk",
    "example.org/?x=1",
    "http://localhost:3000/",
    "https://sites.google.com/view/secure-login",
    "https://accounts.google.com/signin",
    "http://example.com/İ/ABC",
)


class ParityError(Exception):
    pass


def convert(model: Pipeline) -> bytes:
    """Serialise the pipeline as an ONNX graph taking float32 [n, n_features]."""
    onnx_model = to_onnx(
        model,
        initial_types=[(INPUT_NAME, FloatTensorType([None, len(FEATURE_NAMES)]))],
        # Plain probability tensor instead of a list of dicts: far cheaper to read in JS.
        options={id(model[-1]): {"zipmap": False}},
    )
    return onnx_model.SerializeToString()


def onnx_probabilities(onnx_bytes: bytes, features: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    _, probabilities = session.run(None, {INPUT_NAME: features.astype(np.float32)})
    return probabilities[:, 1]


def check_parity(model: Pipeline, onnx_bytes: bytes, features: np.ndarray) -> float:
    """Return the largest probability gap between sklearn and ONNX; raise if over tolerance."""
    gap = float(np.max(np.abs(
        model.predict_proba(features)[:, 1] - onnx_probabilities(onnx_bytes, features)
    )))
    if not gap < PARITY_TOLERANCE:
        raise ParityError(f"sklearn and ONNX probabilities differ by up to {gap:.2e}")
    return gap


def model_metadata(
    model: Pipeline, cfg: dict[str, Any], onnx_bytes: bytes, threshold: float
) -> dict[str, Any]:
    scaler, classifier = model[0], model[-1]
    return {
        "model_version": cfg["model"]["version"],
        "onnx_sha256": hashlib.sha256(onnx_bytes).hexdigest(),
        "input_name": INPUT_NAME,
        "feature_names": list(FEATURE_NAMES),
        "suspicious_keywords": list(SUSPICIOUS_KEYWORDS),
        # contribution of feature i = coefficient[i] * (x[i] - mean[i]) / scale[i]
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        # flag a URL when its probability is above this
        "threshold": threshold,
    }


def parity_fixture(model: Pipeline, urls: list[str], known_sites: frozenset[str]) -> list[dict]:
    features = featurise(urls, known_sites)
    probabilities = model.predict_proba(features)[:, 1]
    return [
        {"url": url, "features": row.tolist(), "probability": float(p)}
        for url, row, p in zip(urls, features, probabilities, strict=True)
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config()
    artifacts, models = Path(cfg["paths"]["artifacts"]), Path(cfg["paths"]["models"])
    model = joblib.load(artifacts / "url-model.joblib")
    known_sites = load_known_sites(cfg)
    dataset = pd.read_parquet(artifacts / "dataset.parquet")

    onnx_bytes = convert(model)
    sample = dataset["url"].sample(5000, random_state=cfg["seed"]).tolist()
    fixture_urls = list(EDGE_CASE_URLS) + load_curated_benign(cfg)[:20]
    gap = check_parity(model, onnx_bytes, featurise(sample + fixture_urls, known_sites))

    # Measured for the record; see quantise.py for why it does not apply to this model.
    _, quantisation = quantise_and_measure(
        onnx_bytes, featurise(fixture_urls, known_sites), INPUT_NAME
    )
    (artifacts / "url-quantisation.json").write_text(json.dumps(quantisation, indent=2), "utf-8")

    models.mkdir(exist_ok=True)
    (models / "url-model.onnx").write_bytes(onnx_bytes)
    threshold = validation_threshold(dataset, model, cfg, known_sites)
    metadata = model_metadata(model, cfg, onnx_bytes, threshold)
    (models / "url-model.json").write_text(json.dumps(metadata, indent=2) + "\n", "utf-8")
    fixture = Path("extension/tests/fixtures/url-parity.json")
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_text(
        json.dumps(parity_fixture(model, fixture_urls, known_sites), indent=1, ensure_ascii=False)
        + "\n", "utf-8",
    )
    log.info("exported bytes=%d max_probability_gap=%.2e fixture_urls=%d",
             len(onnx_bytes), gap, len(fixture_urls))


if __name__ == "__main__":
    main()
