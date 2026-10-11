import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from training.calibrate import fit_platt, fold_calibration


@pytest.fixture(scope="module")
def overconfident():
    """A model whose scores are 3x too extreme, plus fresh data to calibrate on."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6000, 3))
    true_logit = 1.2 * x[:, 0] - 0.8 * x[:, 1] - 1.0
    y = (rng.random(6000) < 1 / (1 + np.exp(-true_logit))).astype(int)
    pipeline = make_pipeline(StandardScaler(), LogisticRegression()).fit(x[:3000], y[:3000])
    pipeline[-1].coef_ *= 3
    pipeline[-1].intercept_ *= 3
    return pipeline, x[3000:], y[3000:]


def test_calibration_improves_brier_and_recovers_the_distortion(overconfident):
    pipeline, x, y = overconfident
    a, b = fit_platt(pipeline.decision_function(x), y)
    calibrated = fold_calibration(pipeline, a, b)
    assert a == pytest.approx(1 / 3, abs=0.05)
    before = brier_score_loss(y, pipeline.predict_proba(x)[:, 1])
    after = brier_score_loss(y, calibrated.predict_proba(x)[:, 1])
    assert after < before


def test_calibration_is_monotonic(overconfident):
    """Calibration may change probabilities but never the order of two URLs."""
    pipeline, x, y = overconfident
    calibrated = fold_calibration(pipeline, *fit_platt(pipeline.decision_function(x), y))
    order = np.argsort(pipeline.decision_function(x), kind="stable")
    assert np.all(np.diff(calibrated.predict_proba(x)[order, 1]) >= 0)


def test_folded_model_equals_sigmoid_of_scaled_score(overconfident):
    pipeline, x, _ = overconfident
    a, b = 0.4, -0.7
    expected = 1 / (1 + np.exp(-(a * pipeline.decision_function(x) + b)))
    np.testing.assert_allclose(fold_calibration(pipeline, a, b).predict_proba(x)[:, 1], expected)


def test_folding_does_not_modify_the_original(overconfident):
    pipeline, _, _ = overconfident
    before = pipeline[-1].coef_.copy()
    fold_calibration(pipeline, 0.5, 0.1)
    np.testing.assert_array_equal(pipeline[-1].coef_, before)


def test_negative_slope_is_rejected():
    scores = np.linspace(-3, 3, 200)
    with pytest.raises(ValueError, match="not positive"):
        fit_platt(scores, (scores < 0).astype(int))  # labels anti-correlated with scores
