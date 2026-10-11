"""Consistency of the committed files in models/ — runs in CI without any training data.

These catch the "stale artifact" class of mistake: someone changes the
featuriser or re-exports one file but not the others, and the browser ends up
running a model that no longer matches its metadata or its reported metrics.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from training.export_onnx import EDGE_CASE_URLS, INPUT_NAME, onnx_probabilities
from training.features.url_features import FEATURE_NAMES, SUSPICIOUS_KEYWORDS, featurise
from training.text.features import TEXT_FEATURE_NAMES, featurise_text
from training.text.lexicon import LEXICON

MODELS = Path("models")
FIXTURE = Path("extension/tests/fixtures/url-parity.json")

pytestmark = pytest.mark.skipif(
    not (MODELS / "url-model.onnx").exists(), reason="no model has been published yet"
)


@pytest.fixture(scope="module")
def metadata():
    return json.loads((MODELS / "url-model.json").read_text("utf-8"))


@pytest.fixture(scope="module")
def known_sites():
    return frozenset((MODELS / "known-sites.txt").read_text("utf-8").split())


def test_onnx_file_matches_the_hash_in_its_metadata(metadata):
    digest = hashlib.sha256((MODELS / "url-model.onnx").read_bytes()).hexdigest()
    assert digest == metadata["onnx_sha256"]


def test_metadata_matches_the_current_featuriser(metadata):
    assert metadata["feature_names"] == list(FEATURE_NAMES)
    assert metadata["suspicious_keywords"] == list(SUSPICIOUS_KEYWORDS)
    assert metadata["input_name"] == INPUT_NAME
    for key in ("scaler_mean", "scaler_scale", "coefficients"):
        assert len(metadata[key]) == len(FEATURE_NAMES)
    assert 0 < metadata["threshold"] < 1


def test_reported_metrics_describe_the_published_model(metadata):
    metrics = json.loads((MODELS / "metrics.json").read_text("utf-8"))
    assert metrics["model_version"] == metadata["model_version"]
    assert metadata["model_version"] in (MODELS / "model_card.html").read_text("utf-8")


def test_known_sites_list_is_well_formed(known_sites):
    trusted = {entry for entry in known_sites if not entry.startswith("!")}
    assert len(trusted) > 1000
    assert all("/" not in entry and " " not in entry and "." in entry for entry in known_sites)
    assert "google.com" in trusted and "!sites.google.com" in known_sites


def test_parity_fixture_still_matches_the_python_featuriser_and_model(metadata, known_sites):
    """The TypeScript tests trust this fixture; it must be what Python computes today."""
    fixture = json.loads(FIXTURE.read_text("utf-8"))
    urls = [row["url"] for row in fixture]
    assert urls[: len(EDGE_CASE_URLS)] == list(EDGE_CASE_URLS)

    features = featurise(urls, known_sites)
    np.testing.assert_allclose(features, [row["features"] for row in fixture], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        onnx_probabilities((MODELS / "url-model.onnx").read_bytes(), features),
        [row["probability"] for row in fixture],
        atol=1e-4,
    )


@pytest.fixture(scope="module")
def text_metadata():
    return json.loads((MODELS / "text-model.json").read_text("utf-8"))


def test_text_model_file_matches_its_metadata_and_the_current_code(text_metadata):
    digest = hashlib.sha256((MODELS / "text-model.onnx").read_bytes()).hexdigest()
    assert digest == text_metadata["onnx_sha256"]
    assert text_metadata["feature_names"] == list(TEXT_FEATURE_NAMES)
    assert text_metadata["lexicon"] == {name: list(phrases) for name, phrases in LEXICON.items()}
    assert 0 < text_metadata["threshold"] < 1
    assert text_metadata["model_version"] in (MODELS / "model_card.html").read_text("utf-8")


def test_text_parity_fixture_still_matches_python():
    fixture = json.loads(Path("extension/tests/fixtures/text-parity.json").read_text("utf-8"))
    assert {row["label"] for row in fixture} == {"legit", "scam"}
    features = featurise_text([row["text"] for row in fixture])
    np.testing.assert_allclose(features, [row["features"] for row in fixture], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        onnx_probabilities((MODELS / "text-model.onnx").read_bytes(), features),
        [row["probability"] for row in fixture],
        atol=1e-4,
    )
