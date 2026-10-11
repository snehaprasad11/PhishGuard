"""Drift report: is the phishing the model sees now still like what it was trained on?

Usage (from the repo root):
    python -m training.drift           # newest phishing in the pinned feed vs the reference
    python -m training.drift --live    # download today's feed first (what the weekly CI job does)

There is no server collecting what users analyse, so drift is watched from
the outside: fresh phishing reports are compared with the reference saved at
training time (models/drift-reference.json). Three things are reported:

* per-feature PSI — has the shape of phishing URLs changed?
* score PSI      — has the distribution of model scores on phishing changed?
* recall         — what share of the newest phishing does the shipped model catch?

Exit status is 2 when drift exceeds the alert level in config.yaml, which the
scheduled workflow turns into a GitHub issue. Exit status 3 means the live
feed could not be downloaded (PhishTank rate-limits downloads made without an
application key; set PHISHTANK_APP_KEY to use one).

PSI (population stability index) sums (new% - ref%) * ln(new% / ref%) over
bins. Rule of thumb: under 0.1 stable, 0.1 to 0.25 worth watching, above 0.25
a real shift.
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from training.config import load_config
from training.data.fetch import _download, _session
from training.export_onnx import onnx_probabilities
from training.features.url_features import FEATURE_NAMES, featurise, parse_url

log = logging.getLogger("phishguard.drift")

BINS = 10
EPSILON = 1e-4  # floor for an empty bin, so the logarithm is defined


def bin_edges(values: np.ndarray) -> list[float]:
    """Inner decile edges of the reference values, de-duplicated (a flag has only one edge)."""
    return sorted(set(np.quantile(values, np.linspace(0, 1, BINS + 1)[1:-1]).tolist()))


def bin_shares(values: np.ndarray, edges: list[float]) -> np.ndarray:
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
    return counts / counts.sum()


def psi(reference: np.ndarray, current: np.ndarray) -> float:
    ref, cur = np.maximum(reference, EPSILON), np.maximum(current, EPSILON)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def _columns(features: np.ndarray, probs: np.ndarray) -> dict[str, np.ndarray]:
    """Every feature column, plus the model's score, by name."""
    named = {name: features[:, i] for i, name in enumerate(FEATURE_NAMES)}
    return named | {"model_score": probs}


def build_reference(features: np.ndarray, probs: np.ndarray, description: str) -> dict[str, Any]:
    """The summary saved at training time: bin edges and shares, never the URLs themselves."""
    columns = _columns(features, probs)
    return {
        "description": description,
        "rows": int(len(probs)),
        "columns": {
            name: {
                "edges": (edges := bin_edges(values)),
                "shares": bin_shares(values, edges).tolist(),
            }
            for name, values in columns.items()
        },
    }


def drift_report(
    reference: dict[str, Any], features: np.ndarray, probs: np.ndarray, threshold: float
) -> dict[str, Any]:
    columns = _columns(features, probs)
    scores = {
        name: psi(np.array(ref["shares"]), bin_shares(columns[name], ref["edges"]))
        for name, ref in reference["columns"].items()
    }
    feature_psi = {name: value for name, value in scores.items() if name != "model_score"}
    return {
        "urls": int(len(probs)),
        "recall": float((probs > threshold).mean()),
        "score_psi": scores["model_score"],
        "max_feature_psi": max(feature_psi.values()),
        "feature_psi": dict(sorted(feature_psi.items(), key=lambda kv: -kv[1])),
    }


def render_markdown(report: dict[str, Any], alert: bool) -> str:
    def level(value: float) -> str:
        return "**shift**" if value > 0.25 else "watch" if value > 0.1 else "stable"

    rows = "\n".join(
        f"| `{name}` | {value:.3f} | {level(value)} |"
        for name, value in list(report["feature_psi"].items())[:10]
    )
    return f"""# PhishGuard drift report

{'**Drift alert:** the newest phishing no longer looks like the training data.'
 if alert else 'No drift alert.'}

- Model: `{report['model_version']}`
- Compared: {report['urls']:,} phishing URLs reported from {report['since']} onward, against
  {report['reference_rows']:,} from training ({report['reference']}).
- **Caught by the shipped model: {report['recall']:.1%}**
- Score distribution PSI: {report['score_psi']:.3f}
- Largest feature PSI: {report['max_feature_psi']:.3f}

| Feature | PSI | |
| --- | --- | --- |
{rows}

PSI under 0.1 is stable, 0.1 to 0.25 is worth watching, above 0.25 is a real shift. A falling
catch rate or a shift on several features is the signal to retrain on a fresh snapshot.
"""


def live_feed_url(configured: str) -> str:
    """The feed URL, using a PhishTank application key if one is provided.

    PhishTank's documented form for regular downloads is
    `.../data/<application key>/online-valid.csv.gz`.
    """
    key = os.environ.get("PHISHTANK_APP_KEY", "").strip()
    if not key:
        return configured
    return configured.replace("/data/online-valid", f"/data/{key}/online-valid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true", help="download today's PhishTank feed")
    parser.add_argument(
        "--since", help="only phishing submitted on or after this date (YYYY-MM-DD)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    cfg = load_config()
    models = Path(cfg["paths"]["models"])
    reference = json.loads((models / "drift-reference.json").read_text("utf-8"))
    metadata = json.loads((models / "url-model.json").read_text("utf-8"))
    known_sites = frozenset((models / "known-sites.txt").read_text("utf-8").split())
    manifest = json.loads(Path(cfg["paths"]["manifest"]).read_text("utf-8"))

    with tempfile.TemporaryDirectory() as tmp:
        if args.live:
            # Fresh data is only read, never pinned: the training snapshot must not change.
            feed = Path(tmp) / "phishtank-live.csv.gz"
            try:
                _download(_session(), live_feed_url(cfg["data"]["phishtank"]["url"]), feed)
            except requests.RequestException as error:
                # The error text can contain the feed URL, and so the key: report only its type.
                print(f"could not download the live feed ({type(error).__name__}). PhishTank "
                      "rate-limits keyless downloads; try later or set PHISHTANK_APP_KEY.")
                sys.exit(3)
            since = args.since or manifest["phishtank"]["snapshot_date"]
        else:
            feed = Path(cfg["paths"]["raw_data"]) / manifest["phishtank"]["file"]
            since = args.since or cfg["split"]["test_start"]
        frame = pd.read_csv(feed, usecols=["url", "submission_time"])
    submitted = pd.to_datetime(frame["submission_time"], utc=True)
    frame = frame[submitted >= pd.Timestamp(since, tz="UTC")]

    urls = []
    for url in frame["url"].drop_duplicates():
        try:
            parse_url(url)
            urls.append(url)
        except ValueError:
            continue
    drift_cfg = cfg["drift"]
    if len(urls) < drift_cfg["min_urls"]:
        print(f"only {len(urls)} phishing URLs since {since}: too few for a drift report")
        return

    features = featurise(urls, known_sites)
    probs = onnx_probabilities((models / "url-model.onnx").read_bytes(), features)
    report = drift_report(reference, features, probs, metadata["threshold"])
    report.update(model_version=metadata["model_version"], since=since,
                  reference=reference["description"], reference_rows=reference["rows"])
    alert = (
        report["max_feature_psi"] > drift_cfg["psi_alert"]
        or report["score_psi"] > drift_cfg["psi_alert"]
    )
    report["alert"] = alert

    artifacts = Path(cfg["paths"]["artifacts"])
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "drift.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    (artifacts / "drift-report.md").write_text(render_markdown(report, alert), "utf-8")
    log.info(
        "urls=%d recall=%.3f score_psi=%.3f max_feature_psi=%.3f alert=%s", report["urls"],
        report["recall"], report["score_psi"], report["max_feature_psi"], alert,
    )
    if alert:
        sys.exit(2)


if __name__ == "__main__":
    main()
