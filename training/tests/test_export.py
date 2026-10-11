import numpy as np
import onnx
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from training.calibrate import fold_calibration
from training.export_onnx import (
    EDGE_CASE_URLS,
    INPUT_NAME,
    ParityError,
    check_parity,
    convert,
    model_metadata,
    onnx_probabilities,
    parity_fixture,
)
from training.features.url_features import FEATURE_NAMES, featurise

CFG = {"model": {"version": "url-test"}}


@pytest.fixture(scope="module")
def model():
    """A calibrated pipeline fitted on real feature vectors with a made-up but learnable label."""
    rng = np.random.default_rng(0)
    urls = [f"https://site{i}.example-{i % 7}.org/a/b{i}?q={i}" for i in range(300)]
    urls += [f"http://secure-login{i}.verify-{i % 5}.shop/" for i in range(300)]
    x = featurise(urls)
    y = np.array([0] * 300 + [1] * 300)
    y[rng.choice(600, 60, replace=False)] ^= 1
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(x, y)
    return fold_calibration(pipeline, 0.8, -0.3), x


def test_onnx_matches_sklearn(model):
    pipeline, x = model
    assert check_parity(pipeline, convert(pipeline), x) < 1e-4


def test_graph_takes_one_float_matrix_in_feature_order(model):
    pipeline, _ = model
    graph = onnx.load_from_string(convert(pipeline)).graph
    assert [i.name for i in graph.input] == [INPUT_NAME]
    dims = graph.input[0].type.tensor_type.shape.dim
    assert dims[1].dim_value == len(FEATURE_NAMES)


def test_parity_check_catches_a_feature_order_mismatch(model):
    """The classic export bug: columns reach the graph in a different order than in training."""
    pipeline, x = model
    onnx_bytes = convert(pipeline)

    class Shuffled:
        def predict_proba(self, features):
            return pipeline.predict_proba(features[:, ::-1])

    with pytest.raises(ParityError, match="differ by up to"):
        check_parity(Shuffled(), onnx_bytes, x)


def test_calibration_survives_the_export(model):
    pipeline, x = model
    scores = pipeline[-1].decision_function(pipeline[0].transform(x))
    np.testing.assert_allclose(
        onnx_probabilities(convert(pipeline), x), 1 / (1 + np.exp(-scores)), atol=1e-4
    )


def test_metadata_reproduces_the_model_score(model):
    """The evidence panel rebuilds the score from these numbers, so they must be sufficient."""
    pipeline, x = model
    meta = model_metadata(pipeline, CFG, convert(pipeline), threshold=0.4)
    assert meta["threshold"] == 0.4
    assert meta["feature_names"] == list(FEATURE_NAMES)
    standardised = (x - np.array(meta["scaler_mean"])) / np.array(meta["scaler_scale"])
    contributions = standardised * np.array(meta["coefficients"])
    probability = 1 / (1 + np.exp(-(contributions.sum(axis=1) + meta["intercept"])))
    np.testing.assert_allclose(probability, pipeline.predict_proba(x)[:, 1], atol=1e-12)


def test_parity_fixture_covers_every_edge_case(model):
    pipeline, _ = model
    fixture = parity_fixture(pipeline, list(EDGE_CASE_URLS), frozenset({"google.com"}))
    assert [row["url"] for row in fixture] == list(EDGE_CASE_URLS)
    assert all(len(row["features"]) == len(FEATURE_NAMES) for row in fixture)
    assert all(0 <= row["probability"] <= 1 for row in fixture)
