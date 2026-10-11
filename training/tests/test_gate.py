import pytest
import yaml

from training.gate import check, gated_metrics

BASELINE = {
    "pr_auc": 0.80, "recall_at_fpr_1pct": 0.65, "brier": 0.028, "test_fpr": 0.010,
    "model_size_kb": 1.1, "latency_ms_p50": 2.0, "curated_holdout_flagged": 0,
    "text_pr_auc": 0.80, "text_test_fpr": 0.006, "text_test_recall": 0.37,
    "text_holdout_legit_flagged": 0, "text_holdout_scam_caught": 11,
}


@pytest.fixture(scope="module")
def gates():
    with open("config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["gates"]


def test_identical_model_passes(gates):
    assert check(BASELINE, BASELINE, gates) == []


def test_small_wobble_within_tolerance_passes(gates):
    candidate = {**BASELINE, "pr_auc": 0.795, "recall_at_fpr_1pct": 0.64, "brier": 0.029}
    assert check(candidate, BASELINE, gates) == []


def test_an_improvement_passes(gates):
    candidate = {**BASELINE, "pr_auc": 0.90, "recall_at_fpr_1pct": 0.80, "brier": 0.020}
    assert check(candidate, BASELINE, gates) == []


@pytest.mark.parametrize(
    ("metric", "worse", "expected"),
    [
        ("pr_auc", 0.70, "dropped more than"),
        ("recall_at_fpr_1pct", 0.50, "dropped more than"),
        ("brier", 0.050, "rose more than"),
        ("test_fpr", 0.030, "above the limit"),
        ("model_size_kb", 5000.0, "above the limit"),
        ("latency_ms_p50", 400.0, "above the limit"),
        ("curated_holdout_flagged", 9, "above the limit"),
    ],
)
def test_a_deliberately_worse_model_fails(gates, metric, worse, expected):
    failures = check({**BASELINE, metric: worse}, BASELINE, gates)
    assert len(failures) == 1
    assert failures[0].startswith(metric) and expected in failures[0]


def test_every_broken_gate_is_reported(gates):
    candidate = {**BASELINE, "pr_auc": 0.5, "brier": 0.2, "test_fpr": 0.5}
    assert len(check(candidate, BASELINE, gates)) == 3


def test_an_unmeasured_required_metric_fails(gates):
    failures = check({**BASELINE, "pr_auc": None}, BASELINE, gates)
    assert failures == ["pr_auc: not measured"]


def test_an_optional_metric_may_be_unmeasured():
    rules = {"latency_ms_p50": {"max": 150, "required": False}}
    assert check({"latency_ms_p50": None}, {}, rules) == []
    assert check({"latency_ms_p50": 200}, {}, rules) != []


def test_absolute_limits_apply_without_a_baseline(gates):
    assert check({**BASELINE, "test_fpr": 0.5}, {}, gates) == [
        "test_fpr: 0.5 is above the limit 0.015"
    ]


def test_gated_metrics_reads_an_evaluation_report():
    report = {
        "metrics": {"pr_auc": 0.8, "recall_at_fpr_1pct": 0.6, "brier": 0.03,
                    "model_size_kb": 1.0, "latency_ms_p50": None},
        "threshold": {"test_fpr": 0.01, "test_recall": 0.6},
        "external": {"curated_benign_holdout": {"flagged": 1},
                     "openphish_unseen_sites": {"recall": 0.5}},
    }
    flat = gated_metrics(report)
    assert flat["curated_holdout_flagged"] == 1
    assert flat["latency_ms_p50"] is None
    assert flat["test_fpr"] == 0.01
