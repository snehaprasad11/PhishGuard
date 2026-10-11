import numpy as np
import pandas as pd
import pytest

from training.evaluate import (
    conservative_threshold,
    evaluate,
    external_checks,
    rates_at,
    reliability_curve,
    select_threshold,
)
from training.model_card import render_model_card
from training.train import fit_model

CFG = {
    "seed": 0,
    "model": {"version": "url-test", "C": 1.0},
    "threshold": {"target_fpr": 0.05, "confidence": 0.9},
    "split": {"val_start": "2026-07-01", "test_start": "2026-08-01"},
}


def test_select_threshold_never_exceeds_the_target_fpr():
    rng = np.random.default_rng(0)
    labels = (rng.random(5000) < 0.2).astype(int)
    probs = np.clip(rng.normal(0.3 + 0.4 * labels, 0.2), 0, 1)
    for target in (0.0, 0.01, 0.05, 0.2):
        threshold = select_threshold(labels, probs, target)
        assert rates_at(labels, probs, threshold)["fpr"] <= target


def test_select_threshold_hand_checked():
    labels = np.array([0] * 10 + [1] * 2)
    probs = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.85, 0.95])
    # 20% of 10 benign = 2 allowed false positives -> threshold is the 3rd highest benign score.
    assert select_threshold(labels, probs, 0.2) == 0.7
    assert rates_at(labels, probs, 0.7) == {"fpr": 0.2, "recall": 1.0, "precision": 0.5}


def test_select_threshold_handles_ties():
    labels = np.array([0] * 100 + [1])
    probs = np.full(101, 0.5)
    assert rates_at(labels, probs, select_threshold(labels, probs, 0.01))["fpr"] == 0.0


def test_weighted_threshold_matches_repeating_rows():
    rng = np.random.default_rng(3)
    labels = np.zeros(400, dtype=int)
    probs = rng.random(400)
    weights = rng.integers(0, 4, 400).astype(float)
    repeated = np.repeat(probs, weights.astype(int))
    expected = select_threshold(np.zeros(len(repeated), dtype=int), repeated, 0.05)
    assert select_threshold(labels, probs, 0.05, weights=weights) == expected


def test_conservative_threshold_is_stricter_and_reproducible():
    rng = np.random.default_rng(4)
    sites = np.repeat(np.arange(300), 10)           # 300 sites, 10 URLs each
    site_level = rng.random(300)[sites]             # URLs of a site score alike
    probs = np.clip(site_level + rng.normal(0, 0.02, 3000), 0, 1)
    labels = np.zeros(3000, dtype=int)
    plain = select_threshold(labels, probs, 0.05)
    cautious = conservative_threshold(labels, probs, sites, 0.05, 0.9, seed=0)
    assert cautious > plain
    assert cautious == conservative_threshold(labels, probs, sites, 0.05, 0.9, seed=0)


def test_reliability_curve_hand_checked():
    labels = np.array([0, 0, 1, 1, 1])
    probs = np.array([0.05, 0.15, 0.95, 0.95, 1.0])
    curve = {b["bin"]: b for b in reliability_curve(labels, probs)}
    assert set(curve) == {"0.0-0.1", "0.1-0.2", "0.9-1.0"}   # empty bins are omitted
    assert curve["0.9-1.0"]["count"] == 3                     # 1.0 lands in the last bin
    assert curve["0.9-1.0"]["observed_rate"] == 1.0
    assert curve["0.0-0.1"]["mean_predicted"] == pytest.approx(0.05)


def _dataset() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for split, n in (("train", 600), ("val", 300), ("test", 300)):
        for i in range(n):
            if rng.random() < 0.3:
                rows.append((f"secure-login-{split}{i}.verify-account.shop/", 1, split))
            else:
                rows.append((f"site{split}{i}.org/docs/a-long-article-{i}", 0, split))
            if rng.random() < 0.1:  # label noise so the classes are not perfectly separable
                rows[-1] = (rows[-1][0], 1 - rows[-1][1], split)
    frame = pd.DataFrame(rows, columns=["url", "label", "split"])
    frame["site"] = frame["url"].str.split("/").str[0]
    return frame


def test_threshold_comes_from_validation_not_test():
    dataset = _dataset()
    model, _ = fit_model(dataset, CFG)
    report = evaluate(dataset, model, CFG)

    flipped = dataset.copy()
    is_test = flipped["split"] == "test"
    flipped.loc[is_test, "label"] = 1 - flipped.loc[is_test, "label"]
    assert evaluate(flipped, model, CFG)["threshold"]["chosen_threshold"] == \
        report["threshold"]["chosen_threshold"]


def test_report_and_model_card_contents():
    dataset = _dataset()
    model, _ = fit_model(dataset, CFG)
    report = evaluate(dataset, model, CFG)

    assert report["rows"]["test"] == {
        "benign": int(((dataset["split"] == "test") & (dataset["label"] == 0)).sum()),
        "phishing": int(((dataset["split"] == "test") & (dataset["label"] == 1)).sum()),
    }
    assert 0 <= report["metrics"]["pr_auc"] <= 1
    assert report["metrics"]["latency_ms_p50"] is None   # never invented
    assert report["slices"]["bare_host"]["phishing"] > 0

    html = render_model_card(report)
    assert "url-test" in html
    assert f"{report['metrics']['pr_auc']:.3f}" in html
    assert html.count("not measured yet") == 2            # size and latency
    report["metrics"]["model_size_kb"] = 1.5
    assert "1.5 KB" in render_model_card(report)
    assert "<script" not in html


def test_external_checks_exclude_seen_sites_and_list_false_positives():
    dataset = _dataset()
    model, _ = fit_model(dataset, CFG)
    seen_site = dataset.loc[dataset["split"] == "train", "site"].iloc[0]
    report = external_checks(
        model, threshold=0.5, dataset=dataset,
        curated_benign=["https://secure-login-x.verify-account.shop/", "https://calm.org/docs/a"],
        openphish=[
            f"https://{seen_site}/again",                    # site seen in training: excluded
            "https://secure-login-new.verify-account.top/",  # unseen
            "HTTPS://WWW.secure-login-new.verify-account.top/",  # same URL after normalising
            "https://",                                      # unparseable: skipped
        ],
    )
    assert report["openphish_unseen_sites"]["total"] == 1
    assert report["curated_benign"]["total"] == 2
    assert [f["url"] for f in report["curated_benign"]["flagged_urls"]] == [
        "https://secure-login-x.verify-account.shop/"
    ]


def test_training_is_deterministic():
    dataset = _dataset()
    first, _ = fit_model(dataset, CFG)
    second, _ = fit_model(dataset, CFG)
    np.testing.assert_array_equal(first[-1].coef_, second[-1].coef_)
    np.testing.assert_array_equal(first[-1].intercept_, second[-1].intercept_)
