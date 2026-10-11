"""Shadow comparison: score the champion and a challenger on the same held-out URLs.

Usage (from the repo root, after the pipeline and `registry register`):
    python -m training.shadow

Aggregate metrics can hide a lot: two models with the same recall can disagree
about which URLs are phishing. Before promoting a challenger this shows, on
the same test split, how many verdicts would change and in which direction —
newly caught phishing, newly missed phishing, new and removed false alarms.

Both models are run from their ONNX files, exactly as the browser would run
them. The champion's files come from the registry; the challenger's are the
ones in models/.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from training.config import load_config
from training.export_onnx import onnx_probabilities
from training.features.url_features import FEATURE_NAMES, featurise
from training.registry import REGISTERED_MODEL, _alias_version, _client

log = logging.getLogger("phishguard.shadow")


def compare(
    labels: np.ndarray, champion: np.ndarray, challenger: np.ndarray,
    champion_threshold: float, challenger_threshold: float,
) -> dict[str, Any]:
    """Metrics for each model at its own threshold, and how their verdicts differ."""
    was, now = champion > champion_threshold, challenger > challenger_threshold
    phishing, benign = labels == 1, labels == 0

    def summary(probs: np.ndarray, flagged: np.ndarray) -> dict[str, float]:
        return {
            "pr_auc": float(average_precision_score(labels, probs)),
            "recall": float(flagged[phishing].mean()),
            "fpr": float(flagged[benign].mean()),
        }

    return {
        "urls": int(len(labels)),
        "champion": summary(champion, was),
        "challenger": summary(challenger, now),
        "verdict_changes": {
            "phishing_newly_caught": int((phishing & ~was & now).sum()),
            "phishing_newly_missed": int((phishing & was & ~now).sum()),
            "benign_newly_flagged": int((benign & ~was & now).sum()),
            "benign_no_longer_flagged": int((benign & was & ~now).sum()),
            "unchanged": int((was == now).sum()),
        },
    }


def _score(release: Path, urls: list[str]) -> tuple[np.ndarray, float] | None:
    """Probabilities and threshold from a release's URL model, or None if it cannot be scored."""
    metadata = json.loads((release / "url-model.json").read_text("utf-8"))
    if metadata["feature_names"] != list(FEATURE_NAMES):
        return None  # built with a different featuriser: its inputs cannot be recomputed here
    known_sites = frozenset((release / "known-sites.txt").read_text("utf-8").split())
    features = featurise(urls, known_sites)
    probs = onnx_probabilities((release / "url-model.onnx").read_bytes(), features)
    return probs, metadata["threshold"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config()
    artifacts, models = Path(cfg["paths"]["artifacts"]), Path(cfg["paths"]["models"])
    champion_version = _alias_version(_client(cfg), "champion")
    if champion_version is None:
        raise SystemExit(
            f"no champion in the '{REGISTERED_MODEL}' registry yet: nothing to compare"
        )

    dataset = pd.read_parquet(artifacts / "dataset.parquet")
    test = dataset[dataset["split"] == "test"]
    urls, labels = test["url"].tolist(), test["label"].to_numpy()

    with tempfile.TemporaryDirectory() as tmp:
        champion_dir = Path(
            mlflow.artifacts.download_artifacts(artifact_uri=champion_version.source, dst_path=tmp)
        )
        champion, challenger = _score(champion_dir, urls), _score(models, urls)
    if champion is None or challenger is None:
        raise SystemExit(
            "the champion was built with a different feature list, so it cannot be re-scored "
            "here; compare models/metrics.json with models/baseline-metrics.json instead"
        )

    result = compare(labels, champion[0], challenger[0], champion[1], challenger[1])
    result["champion_version"] = champion_version.version
    (artifacts / "shadow.json").write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
