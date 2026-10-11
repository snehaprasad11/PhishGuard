"""The release gate: refuse a model that is worse than the accepted baseline.

Usage (from the repo root):
    python -m training.gate            # compare models/metrics.json with the baseline
    python -m training.gate --accept   # make the current metrics the new baseline

`models/metrics.json` is written by the evaluation run and describes the model
that would ship. `models/baseline-metrics.json` describes the last model a
person accepted. CI runs this on every change and fails if any gated metric
regresses beyond its tolerance, or breaks an absolute limit, from config.yaml.

Accepting a new baseline is deliberately a separate, explicit command: a gate
that updated its own baseline would approve any slow decline.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from training.config import load_config


def check(
    current: dict[str, float | None], baseline: dict[str, float | None], gates: dict[str, Any]
) -> list[str]:
    """Return one message per broken gate; an empty list means the model may ship."""
    failures = []
    for name, rule in gates.items():
        value, reference = current.get(name), baseline.get(name)
        if value is None:
            # A metric the gate depends on must exist; "not measured" is not a pass.
            if rule.get("required", True):
                failures.append(f"{name}: not measured")
            continue
        if "max" in rule and value > rule["max"]:
            failures.append(f"{name}: {value:.4g} is above the limit {rule['max']:.4g}")
        if "min" in rule and value < rule["min"]:
            failures.append(f"{name}: {value:.4g} is below the limit {rule['min']:.4g}")
        if reference is None:
            continue
        if "max_drop" in rule and value < reference - rule["max_drop"]:
            failures.append(
                f"{name}: {value:.4g} dropped more than {rule['max_drop']:.4g} "
                f"below the baseline {reference:.4g}"
            )
        if "max_increase" in rule and value > reference + rule["max_increase"]:
            failures.append(
                f"{name}: {value:.4g} rose more than {rule['max_increase']:.4g} "
                f"above the baseline {reference:.4g}"
            )
    return failures


def gated_metrics(report: dict[str, Any]) -> dict[str, float | None]:
    """The flat set of numbers the gate looks at, drawn from an evaluation report."""
    external = report.get("external", {})
    text = report.get("text_model", {})
    text_holdout = text.get("curated", {}).get("holdout", {})
    return {
        "pr_auc": report["metrics"]["pr_auc"],
        "recall_at_fpr_1pct": report["metrics"]["recall_at_fpr_1pct"],
        "brier": report["metrics"]["brier"],
        "test_fpr": report["threshold"]["test_fpr"],
        "test_recall": report["threshold"]["test_recall"],
        "model_size_kb": report["metrics"]["model_size_kb"],
        "latency_ms_p50": report["metrics"]["latency_ms_p50"],
        "curated_holdout_flagged": external.get("curated_benign_holdout", {}).get("flagged"),
        "openphish_unseen_recall": external.get("openphish_unseen_sites", {}).get("recall"),
        "text_pr_auc": text.get("metrics", {}).get("pr_auc"),
        "text_test_fpr": text.get("threshold", {}).get("test_fpr"),
        "text_test_recall": text.get("threshold", {}).get("test_recall"),
        "text_holdout_legit_flagged": (
            len(text_holdout["legitimate_flagged"]) if text_holdout else None
        ),
        "text_holdout_scam_caught": text_holdout.get("scam_caught"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--accept", action="store_true", help="make the current metrics the baseline"
    )
    args = parser.parse_args()

    cfg = load_config()
    models = Path(cfg["paths"]["models"])
    current_path, baseline_path = models / "metrics.json", models / "baseline-metrics.json"
    if not current_path.exists():
        sys.exit(f"{current_path} is missing: run `python -m training.pipeline` first")

    if args.accept:
        shutil.copyfile(current_path, baseline_path)
        accepted = json.loads(current_path.read_text("utf-8"))
        print(f"baseline is now {accepted['model_version']}")
        return
    if not baseline_path.exists():
        sys.exit(f"{baseline_path} is missing: there is nothing to compare against. "
                 "Review the model, then run `python -m training.gate --accept`.")

    current = json.loads(current_path.read_text("utf-8"))
    baseline = json.loads(baseline_path.read_text("utf-8"))
    failures = check(current["metrics"], baseline["metrics"], cfg["gates"])

    print(f"candidate {current['model_version']}  vs  baseline {baseline['model_version']}")
    for name in cfg["gates"]:
        print(f"  {name:26s} {current['metrics'].get(name)!s:>22}   "
              f"(baseline {baseline['metrics'].get(name)!s})")
    if failures:
        print("\nGATE FAILED")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    print("\ngate passed")


if __name__ == "__main__":
    main()
