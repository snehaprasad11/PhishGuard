"""Probability calibration by Platt (sigmoid) scaling.

A logistic regression outputs p = sigmoid(z), where z = w.x + c is its score.
Platt scaling fits two numbers (a, b) on held-out data so that sigmoid(a*z + b)
matches observed phishing rates. Because a*z + b = (a*w).x + (a*c + b), the
calibrated model is still a single linear model: we fold (a, b) into the
coefficients. That keeps the evidence panel exact (per-feature contributions
sum to the calibrated score) and the ONNX export a plain logistic regression.
"""

import copy

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Fit sigmoid(a * score + b) to the labels; scores are uncalibrated log-odds."""
    # Effectively unregularised: with one input and thousands of rows there is nothing to overfit.
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(scores.reshape(-1, 1), labels)
    a, b = float(platt.coef_[0, 0]), float(platt.intercept_[0])
    if a <= 0:
        raise ValueError(f"calibration slope {a:.4f} is not positive: it would reverse the ranking")
    return a, b


def fold_calibration(pipeline: Pipeline, a: float, b: float) -> Pipeline:
    """Return a copy of the pipeline whose final logistic regression is calibrated."""
    calibrated = copy.deepcopy(pipeline)
    classifier = calibrated[-1]
    classifier.intercept_ = a * classifier.intercept_ + b
    classifier.coef_ = a * classifier.coef_
    return calibrated
