"""Evaluate the trained URL model on the held-out test split.

Usage (from the repo root, after `python -m training.train`):
    python -m training.evaluate

The decision threshold is chosen on the VALIDATION split to hit the target
false-positive rate, then applied unchanged to the test split. Choosing it on
the test split would make the reported false-positive rate true by
construction instead of a measurement.
"""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from training.config import load_config
from training.data.domains import registered_domain
from training.data.known_sites import load_known_sites
from training.data.loaders import load_curated_benign, load_openphish, load_phishing
from training.features.url_features import (
    FEATURE_NAMES,
    featurise,
    on_known_site,
    parse_url,
)
from training.model_card import render_model_card

log = logging.getLogger("phishguard.evaluate")


def select_threshold(
    labels: np.ndarray, probs: np.ndarray, target_fpr: float, weights: np.ndarray | None = None
) -> float:
    """Lowest threshold t such that flagging `prob > t` keeps the benign FPR <= target_fpr.

    With `weights`, the FPR is the weighted share of benign rows flagged.
    """
    benign = labels == 0
    order = np.argsort(-probs[benign], kind="stable")
    sorted_probs = probs[benign][order]
    w = np.ones(len(sorted_probs)) if weights is None else weights[benign][order]
    # First row whose inclusion would push the flagged weight over budget: flag only above it.
    over_budget = np.searchsorted(np.cumsum(w), target_fpr * w.sum(), side="right")
    return float(sorted_probs[min(over_budget, len(sorted_probs) - 1)])


def conservative_threshold(
    labels: np.ndarray, probs: np.ndarray, sites: np.ndarray,
    target_fpr: float, confidence: float, seed: int, resamples: int = 200,
) -> float:
    """A threshold that meets target_fpr in `confidence` of site-level bootstrap resamples.

    URLs from one site are scored alike, so the validation FPR is noisier than
    its row count suggests. Resampling whole sites captures that, and taking a
    high quantile of the resulting thresholds leaves a margin instead of
    aiming exactly at the target and overshooting on new data half the time.
    """
    rng = np.random.default_rng(seed)
    site_ids, site_index = np.unique(sites, return_inverse=True)
    thresholds = [
        select_threshold(
            labels, probs, target_fpr,
            weights=rng.multinomial(len(site_ids), np.full(len(site_ids), 1 / len(site_ids)))[
                site_index
            ].astype(float),
        )
        for _ in range(resamples)
    ]
    return float(np.quantile(thresholds, confidence))


def rates_at(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    flagged = probs > threshold
    true_pos = int((flagged & (labels == 1)).sum())
    return {
        "fpr": float(flagged[labels == 0].mean()),
        "recall": float(flagged[labels == 1].mean()),
        "precision": true_pos / int(flagged.sum()) if flagged.any() else float("nan"),
    }


def reliability_curve(labels: np.ndarray, probs: np.ndarray, bins: int = 10) -> list[dict]:
    """Per equal-width probability bin: mean predicted probability vs observed phishing rate."""
    index = np.minimum((probs * bins).astype(int), bins - 1)
    curve = []
    for b in range(bins):
        mask = index == b
        if mask.any():
            curve.append({
                "bin": f"{b / bins:.1f}-{(b + 1) / bins:.1f}",
                "count": int(mask.sum()),
                "mean_predicted": float(probs[mask].mean()),
                "observed_rate": float(labels[mask].mean()),
            })
    return curve


def _is_bare_host(url: str) -> bool:
    parsed = parse_url(url)
    return parsed.path == "/" and not parsed.query


def validation_threshold(
    dataset: pd.DataFrame, model: Any, cfg: dict[str, Any],
    known_sites: frozenset[str] = frozenset(),
) -> float:
    """The decision threshold: a function of the model and the validation split only.

    Both the export (which ships it to the browser) and the evaluation (which
    reports results at it) call this, so they cannot disagree.
    """
    val = dataset[dataset["split"] == "val"]
    probs = model.predict_proba(featurise(val["url"].tolist(), known_sites))[:, 1]
    return conservative_threshold(
        val["label"].to_numpy(), probs, val["site"].to_numpy(),
        cfg["threshold"]["target_fpr"], cfg["threshold"]["confidence"], cfg["seed"],
    )


def evaluate(
    dataset: pd.DataFrame, model: Any, cfg: dict[str, Any],
    known_sites: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    target_fpr = cfg["threshold"]["target_fpr"]
    val, test = dataset[dataset["split"] == "val"], dataset[dataset["split"] == "test"]
    y_val, y_test = val["label"].to_numpy(), test["label"].to_numpy()
    p_val = model.predict_proba(featurise(val["url"].tolist(), known_sites))[:, 1]
    p_test = model.predict_proba(featurise(test["url"].tolist(), known_sites))[:, 1]

    threshold = validation_threshold(dataset, model, cfg, known_sites)
    at_threshold = rates_at(y_test, p_test, threshold)
    bare = test["url"].map(_is_bare_host).to_numpy()
    known = test["url"].map(
        lambda url: on_known_site(parse_url(url).host, known_sites)
    ).to_numpy()

    def slice_rates(mask: np.ndarray) -> dict[str, Any]:
        benign, phishing = mask & (y_test == 0), mask & (y_test == 1)
        flagged = p_test > threshold
        return {
            "benign": int(benign.sum()),
            "phishing": int(phishing.sum()),
            "fpr": float(flagged[benign].mean()) if benign.any() else None,
            "recall": float(flagged[phishing].mean()) if phishing.any() else None,
        }

    counts = dataset.groupby(["split", "label"]).size()
    split_cfg = cfg["split"]
    return {
        "model_version": cfg["model"]["version"],
        "split": (
            f"site-disjoint; phishing by submission date "
            f"(train<{split_cfg['val_start']}, val<{split_cfg['test_start']}, test>=that)"
        ),
        "rows": {
            split: {"benign": int(counts[split, 0]), "phishing": int(counts[split, 1])}
            for split in ("train", "val", "test")
        },
        "metrics": {
            "pr_auc": float(average_precision_score(y_test, p_test)),
            # Recall at the test split's own 1%-FPR point: a property of the ranking alone.
            "recall_at_fpr_1pct": rates_at(
                y_test, p_test, select_threshold(y_test, p_test, target_fpr)
            )["recall"],
            "brier": float(brier_score_loss(y_test, p_test)),
            "latency_ms_p50": None,   # measured in the browser (Step 7 onwards)
            "model_size_kb": None,    # filled in by main() from the exported .onnx
        },
        "threshold": {
            "target_fpr": target_fpr,
            "chosen_threshold": threshold,
            "chosen_on": (
                f"validation split, {cfg['threshold']['confidence']:.0%} site-bootstrap bound"
            ),
            "val_fpr": rates_at(y_val, p_val, threshold)["fpr"],
            "test_fpr": at_threshold["fpr"],
            "test_recall": at_threshold["recall"],
            "test_precision": at_threshold["precision"],
        },
        "slices": {
            "bare_host": slice_rates(bare),
            "has_path_or_query": slice_rates(~bare),
            "on_well_known_site": slice_rates(known),
            "not_on_well_known_site": slice_rates(~known),
            **{
                f"benign_tier_{tier}": slice_rates((test["tier"] == tier).to_numpy())
                for tier in ("top100k", "top1m", "unranked")
                if "tier" in test
            },
        },
        "reliability_curve": reliability_curve(y_test, p_test),
        "coefficients": {
            name: float(c) for name, c in zip(FEATURE_NAMES, model[-1].coef_[0], strict=True)
        },
    }


def _must_not_flag(
    model: Any, threshold: float, urls: list[str], known_sites: frozenset[str]
) -> dict[str, Any]:
    probs = model.predict_proba(featurise(urls, known_sites))[:, 1]
    flagged = [
        {"url": url, "probability": float(p)}
        for url, p in sorted(zip(urls, probs, strict=True), key=lambda pair: -pair[1])
        if p > threshold
    ]
    return {"total": len(urls), "flagged": len(flagged), "flagged_urls": flagged}


def external_checks(
    model: Any, threshold: float, dataset: pd.DataFrame,
    curated_benign: list[str], openphish: list[str],
    curated_holdout: list[str] | None = None,
    known_sites: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Checks on data from outside the training distribution.

    * curated_benign: well-known legitimate pages; lists every one the model would flag.
      This list informed the model's design, so it is a regression check, not a fair test.
    * curated_holdout: a second such list that was never used for any decision.
    * openphish: phishing from a second feed, restricted to sites absent from the
      train and validation splits so it measures generalisation, not memory.
    """
    seen_sites = set(dataset.loc[dataset["split"] != "test", "site"])
    unseen: dict[str, None] = {}
    for url in openphish:
        try:
            normalised = parse_url(url).normalised
        except ValueError:
            continue
        if registered_domain(normalised, private=True) not in seen_sites:
            unseen[normalised] = None
    phish_probs = model.predict_proba(featurise(list(unseen), known_sites))[:, 1]

    result = {
        "curated_benign": _must_not_flag(model, threshold, curated_benign, known_sites),
        "openphish_unseen_sites": {
            "total": len(unseen),
            "recall": float((phish_probs > threshold).mean()),
        },
    }
    if curated_holdout is not None:
        result["curated_benign_holdout"] = _must_not_flag(
            model, threshold, curated_holdout, known_sites
        )
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config()
    artifacts = Path(cfg["paths"]["artifacts"])
    dataset = pd.read_parquet(artifacts / "dataset.parquet")
    model = joblib.load(artifacts / "url-model.joblib")

    known_sites = load_known_sites(cfg)
    report = evaluate(dataset, model, cfg, known_sites)
    onnx_path = Path(cfg["paths"]["models"]) / "url-model.onnx"
    if onnx_path.exists():
        report["metrics"]["model_size_kb"] = onnx_path.stat().st_size / 1024
        # The browser applies the threshold shipped in the metadata. If it is not the
        # one these results were measured at, the model card would describe a
        # different product from the one that ships.
        shipped = json.loads(onnx_path.with_suffix(".json").read_text("utf-8"))
        if shipped.get("threshold") != report["threshold"]["chosen_threshold"]:
            raise SystemExit(
                "models/url-model.json is stale: re-run `python -m training.export_onnx`"
            )
    report["external"] = external_checks(
        model, report["threshold"]["chosen_threshold"], dataset,
        load_curated_benign(cfg), load_openphish(cfg),
        curated_holdout=load_curated_benign(cfg, holdout=True), known_sites=known_sites,
    )
    # How much real phishing sits on well-known domains, before the per-site cap hides it.
    raw_phishing = load_phishing(cfg)
    recent = raw_phishing[raw_phishing["timestamp"] >= cfg["split"]["phishing_min_date"]]
    report["known_sites"] = {
        "count": sum(not entry.startswith("!") for entry in known_sites),
        "untrusted_hosts": sum(entry.startswith("!") for entry in known_sites),
        "tranco_top_n": cfg["data"]["tranco"]["known_sites_top_n"],
        "raw_phishing_share": float(np.mean([
            on_known_site(parse_url(url).host, known_sites)
            for url in recent["url"] if url.strip()
        ])),
    }
    # Browser latency is measured by the extension's benchmark, not here. It only
    # counts if it was measured for this exact model version.
    latency_path = Path(cfg["paths"]["models"]) / "latency.json"
    if latency_path.exists():
        latency = json.loads(latency_path.read_text("utf-8"))
        if latency["model_version"] == report["model_version"]:
            report["metrics"]["latency_ms_p50"] = latency["click_to_result_ms"]["p50"]
            report["latency"] = latency
    for name, key in (("url-quantisation.json", "quantisation"), ("text-eval.json", "text_model")):
        if (artifacts / name).exists():
            report[key] = json.loads((artifacts / name).read_text("utf-8"))
    benchmark = artifacts / "benchmark.json"
    if benchmark.exists():
        report["benchmark"] = json.loads(benchmark.read_text("utf-8"))
    manifest = json.loads(Path(cfg["paths"]["manifest"]).read_text("utf-8"))
    report["trained_on"] = {
        name: {"snapshot_date": entry["snapshot_date"], "sha256": entry["sha256"]}
        for name, entry in manifest.items()
    }

    (artifacts / "eval.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    # What phishing looked like at training time, for the drift report to compare against.
    from training.drift import build_reference

    training_phishing = dataset[(dataset["split"] == "train") & (dataset["label"] == 1)]
    reference_features = featurise(training_phishing["url"].tolist(), known_sites)
    (Path(cfg["paths"]["models"]) / "drift-reference.json").write_text(
        json.dumps(build_reference(
            reference_features, model.predict_proba(reference_features)[:, 1],
            f"training-split phishing, {report['model_version']}",
        )) + "\n", "utf-8",
    )
    # The small, committed summary the release gate compares against its baseline.
    from training.gate import gated_metrics

    summary = {"model_version": report["model_version"], "metrics": gated_metrics(report)}
    (Path(cfg["paths"]["models"]) / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", "utf-8"
    )
    models = Path(cfg["paths"]["models"])
    models.mkdir(exist_ok=True)
    (models / "model_card.html").write_text(render_model_card(report), "utf-8")
    measured = {k: v for k, v in report["metrics"].items() if v is not None}
    log.info("metrics %s", json.dumps(measured))
    log.info("threshold %s", json.dumps(report["threshold"]))
    log.info("external curated_flagged=%d/%d holdout_flagged=%d/%d openphish_recall=%.3f (n=%d)",
             report["external"]["curated_benign"]["flagged"],
             report["external"]["curated_benign"]["total"],
             report["external"]["curated_benign_holdout"]["flagged"],
             report["external"]["curated_benign_holdout"]["total"],
             report["external"]["openphish_unseen_sites"]["recall"],
             report["external"]["openphish_unseen_sites"]["total"])

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    with mlflow.start_run(run_name="evaluate"):
        mlflow.log_param("model_version", report["model_version"])
        mlflow.log_metrics({f"test_{k}": v for k, v in measured.items()})
        mlflow.log_metrics({
            k: v for k, v in report["threshold"].items() if isinstance(v, float)
        })
        mlflow.log_metric("curated_benign_flagged", report["external"]["curated_benign"]["flagged"])
        mlflow.log_metric(
            "curated_holdout_flagged", report["external"]["curated_benign_holdout"]["flagged"]
        )
        mlflow.log_metric(
            "openphish_unseen_recall", report["external"]["openphish_unseen_sites"]["recall"]
        )
        mlflow.log_artifact(str(artifacts / "eval.json"))
        mlflow.log_artifact(str(models / "model_card.html"))


if __name__ == "__main__":
    main()
