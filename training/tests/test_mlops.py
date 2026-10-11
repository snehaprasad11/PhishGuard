"""Registry (promote / rollback), shadow comparison and drift report."""

import json

import mlflow
import numpy as np
import pytest

from training import registry
from training.drift import (
    bin_edges,
    bin_shares,
    build_reference,
    drift_report,
    live_feed_url,
    psi,
    render_markdown,
)
from training.features.url_features import FEATURE_NAMES
from training.shadow import compare

METRICS = {"pr_auc": 0.80, "test_fpr": 0.010}
GATES = {"pr_auc": {"max_drop": 0.01}, "test_fpr": {"max": 0.015}}


@pytest.fixture
def cfg(tmp_path):
    """A private registry and models/ folder, so tests never touch the real ones."""
    models = tmp_path / "models"
    models.mkdir()
    tracking = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking)
    mlflow.create_experiment("test", artifact_location=(tmp_path / "artifacts").as_uri())
    return {
        "paths": {"models": str(models)},
        "mlflow": {"tracking_uri": tracking, "experiment": "test"},
        "gates": GATES,
    }


def publish(cfg, tag: str, **metric_overrides) -> None:
    """Write a complete, distinguishable release into models/."""
    models = registry.Path(cfg["paths"]["models"])
    for name in registry.RELEASE_FILES:
        (models / name).write_text(f"{name} from {tag}", "utf-8")
    (models / "metrics.json").write_text(
        json.dumps({"model_version": f"url-{tag}", "metrics": {**METRICS, **metric_overrides}}), "utf-8"
    )
    (models / "text-model.json").write_text(json.dumps({"model_version": f"text-{tag}"}), "utf-8")


def read(cfg, name: str) -> str:
    return (registry.Path(cfg["paths"]["models"]) / name).read_text("utf-8")


def test_register_promote_and_rollback_restore_the_previous_release_exactly(cfg):
    publish(cfg, "v1")
    v1 = registry.register(cfg)
    assert registry.promote(cfg) == v1
    assert registry.status(cfg)["champion"] == {"version": v1, "url_model": "url-v1", "text_model": "text-v1"}
    assert json.loads(read(cfg, "baseline-metrics.json"))["model_version"] == "url-v1"

    publish(cfg, "v2")
    v2 = registry.register(cfg)
    assert registry.status(cfg)["challenger"]["version"] == v2
    assert registry.status(cfg)["champion"]["version"] == v1      # registering does not ship
    registry.promote(cfg)
    state = registry.status(cfg)
    assert (state["champion"]["version"], state["previous"]["version"]) == (v2, v1)
    assert read(cfg, "url-model.onnx") == "url-model.onnx from v2"

    assert registry.rollback(cfg) == v1
    state = registry.status(cfg)
    assert state["champion"]["version"] == v1
    assert state["challenger"]["version"] == v2                   # kept for investigation
    assert state["previous"] is None
    for name in ("url-model.onnx", "known-sites.txt", "text-model.onnx", "model_card.html"):
        assert read(cfg, name) == f"{name} from v1"
    assert json.loads(read(cfg, "metrics.json"))["model_version"] == "url-v1"
    assert json.loads(read(cfg, "baseline-metrics.json"))["model_version"] == "url-v1"


def test_a_worse_challenger_is_not_promoted(cfg):
    publish(cfg, "v1")
    registry.register(cfg)
    registry.promote(cfg)
    publish(cfg, "v2", pr_auc=0.60)
    registry.register(cfg)
    with pytest.raises(SystemExit, match="gate failed"):
        registry.promote(cfg)
    assert registry.status(cfg)["champion"]["url_model"] == "url-v1"
    assert json.loads(read(cfg, "baseline-metrics.json"))["model_version"] == "url-v1"


def test_registry_refuses_impossible_requests(cfg):
    with pytest.raises(SystemExit, match="missing"):
        registry.register(cfg)
    with pytest.raises(SystemExit, match="nothing to promote"):
        registry.promote(cfg)
    publish(cfg, "v1")
    registry.register(cfg)
    registry.promote(cfg)
    with pytest.raises(SystemExit, match="already the champion"):
        registry.promote(cfg)
    with pytest.raises(SystemExit, match="nothing to roll back"):
        registry.rollback(cfg)


def test_shadow_comparison_counts_every_kind_of_verdict_change():
    labels = np.array([1, 1, 1, 0, 0, 0])
    champion = np.array([0.9, 0.9, 0.1, 0.9, 0.1, 0.1])
    challenger = np.array([0.9, 0.1, 0.9, 0.1, 0.9, 0.1])
    result = compare(labels, champion, challenger, 0.5, 0.5)
    assert result["verdict_changes"] == {
        "phishing_newly_caught": 1,
        "phishing_newly_missed": 1,
        "benign_newly_flagged": 1,
        "benign_no_longer_flagged": 1,
        "unchanged": 2,
    }
    assert result["champion"]["recall"] == result["challenger"]["recall"] == pytest.approx(2 / 3)
    assert result["urls"] == 6


def test_shadow_comparison_uses_each_models_own_threshold():
    labels = np.array([1, 0])
    same = np.array([0.6, 0.6])
    result = compare(labels, same, same, champion_threshold=0.5, challenger_threshold=0.7)
    # Identical scores cannot rank the two URLs, so PR-AUC is the base rate, 0.5.
    assert result["champion"] == {"pr_auc": 0.5, "recall": 1.0, "fpr": 1.0}
    assert result["challenger"]["recall"] == 0.0 and result["challenger"]["fpr"] == 0.0
    assert result["verdict_changes"]["phishing_newly_missed"] == 1
    assert result["verdict_changes"]["benign_no_longer_flagged"] == 1


def test_psi_hand_checked():
    assert psi(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.0
    # (0.8-0.5)*ln(0.8/0.5) + (0.2-0.5)*ln(0.2/0.5)
    expected = 0.3 * np.log(1.6) - 0.3 * np.log(0.4)
    assert psi(np.array([0.5, 0.5]), np.array([0.8, 0.2])) == pytest.approx(expected)
    assert np.isfinite(psi(np.array([1.0, 0.0]), np.array([0.0, 1.0])))  # empty bins are floored


def test_bins_handle_continuous_values_and_flags():
    values = np.arange(1000, dtype=float)
    edges = bin_edges(values)
    assert len(edges) == 9
    np.testing.assert_allclose(bin_shares(values, edges), 0.1, atol=0.002)
    flag = np.array([0.0] * 95 + [1.0] * 5)
    assert bin_edges(flag) == [0.0]                    # one distinct edge, two bins
    np.testing.assert_allclose(bin_shares(flag, bin_edges(flag)), [0.0, 1.0])
    np.testing.assert_allclose(bin_shares(np.array([0.0, 1.0]), [0.5]), [0.5, 0.5])


def test_drift_report_is_quiet_on_the_same_population_and_loud_on_a_shifted_one():
    rng = np.random.default_rng(0)
    n = len(FEATURE_NAMES)
    features = rng.normal(size=(4000, n))
    probs = rng.beta(5, 2, 4000)
    reference = build_reference(features, probs, "unit test")
    assert set(reference["columns"]) == {*FEATURE_NAMES, "model_score"}
    assert "url" not in json.dumps(reference)          # a summary only, never the URLs

    same = drift_report(reference, rng.normal(size=(4000, n)), rng.beta(5, 2, 4000), threshold=0.5)
    assert same["max_feature_psi"] < 0.05 and same["score_psi"] < 0.05

    shifted = rng.normal(size=(4000, n))
    shifted[:, 3] += 1.5
    moved = drift_report(reference, shifted, rng.beta(2, 5, 4000), threshold=0.5)
    assert next(iter(moved["feature_psi"])) == FEATURE_NAMES[3]   # sorted, worst first
    assert moved["max_feature_psi"] > 0.25 and moved["score_psi"] > 0.25
    assert moved["recall"] < same["recall"]


def test_drift_markdown_states_the_alert_and_the_catch_rate():
    report = {
        "model_version": "url-test", "urls": 1234, "since": "2026-10-01", "reference": "train",
        "reference_rows": 5000, "recall": 0.61, "score_psi": 0.31, "max_feature_psi": 0.4,
        "feature_psi": {"host_length": 0.4, "path_depth": 0.15, "has_at": 0.01},
    }
    alert = render_markdown(report, alert=True)
    assert "Drift alert" in alert and "61.0%" in alert and "1,234" in alert
    assert "| `host_length` | 0.400 | **shift** |" in alert
    assert "| `path_depth` | 0.150 | watch |" in alert
    assert "No drift alert" in render_markdown(report, alert=False)


def test_live_feed_url_uses_an_application_key_only_when_one_is_set(monkeypatch):
    url = "http://data.phishtank.com/data/online-valid.csv.gz"
    monkeypatch.delenv("PHISHTANK_APP_KEY", raising=False)
    assert live_feed_url(url) == url
    monkeypatch.setenv("PHISHTANK_APP_KEY", " abc123 ")
    assert live_feed_url(url) == "http://data.phishtank.com/data/abc123/online-valid.csv.gz"
