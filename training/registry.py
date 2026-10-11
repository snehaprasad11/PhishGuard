"""Model registry: versioned releases with champion / challenger aliases and rollback.

Usage (from the repo root):
    python -m training.registry register   # record models/ as a new version, alias "challenger"
    python -m training.registry promote    # challenger -> champion (only if the gate passes)
    python -m training.registry rollback   # champion <- the previous champion; restore its files
    python -m training.registry status

A *release* is everything the extension ships: both ONNX models, their
metadata, the known-sites list, the metrics and the model card. One MLflow
registered model holds every release as a version, so a version can be
restored byte for byte.

Aliases:
    challenger  the most recently registered release, awaiting a decision
    champion    the release the extension should ship
    previous    the champion before the last promotion: what rollback restores
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from training.config import load_config
from training.gate import check

REGISTERED_MODEL = "phishguard"
RELEASE_FILES = (
    "url-model.onnx", "url-model.json", "known-sites.txt",
    "text-model.onnx", "text-model.json", "metrics.json", "model_card.html",
)


def _client(cfg: dict[str, Any]) -> MlflowClient:
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    return MlflowClient()


def _alias_version(client: MlflowClient, alias: str) -> Any | None:
    try:
        return client.get_model_version_by_alias(REGISTERED_MODEL, alias)
    except MlflowException:
        return None


def register(cfg: dict[str, Any]) -> str:
    """Record the files in models/ as a new release and point "challenger" at it."""
    models = Path(cfg["paths"]["models"])
    missing = [name for name in RELEASE_FILES if not (models / name).exists()]
    if missing:
        raise SystemExit(f"cannot register: models/ is missing {missing}; run the pipeline first")
    metrics = json.loads((models / "metrics.json").read_text("utf-8"))
    text_version = json.loads((models / "text-model.json").read_text("utf-8"))["model_version"]

    client = _client(cfg)
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    with mlflow.start_run(run_name="release") as run:
        for name in RELEASE_FILES:
            mlflow.log_artifact(str(models / name), artifact_path="release")
        mlflow.log_metrics({k: v for k, v in metrics["metrics"].items() if v is not None})
    try:
        client.create_registered_model(REGISTERED_MODEL)
    except MlflowException:
        pass  # already exists
    version = client.create_model_version(
        REGISTERED_MODEL, source=f"{run.info.artifact_uri}/release", run_id=run.info.run_id,
        tags={"url_model": metrics["model_version"], "text_model": text_version},
    )
    client.set_registered_model_alias(REGISTERED_MODEL, "challenger", version.version)
    return version.version


def promote(cfg: dict[str, Any]) -> str:
    """Make the challenger the champion, if it passes the gate against the current baseline."""
    client = _client(cfg)
    challenger = _alias_version(client, "challenger")
    if challenger is None:
        raise SystemExit("nothing to promote: register a release first")
    champion = _alias_version(client, "champion")
    if champion is not None and champion.version == challenger.version:
        raise SystemExit(f"version {champion.version} is already the champion")

    models = Path(cfg["paths"]["models"])
    current = json.loads((models / "metrics.json").read_text("utf-8"))
    baseline_path = models / "baseline-metrics.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text("utf-8"))
        failures = check(current["metrics"], baseline["metrics"], cfg["gates"])
        if failures:
            raise SystemExit("not promoted, the gate failed:\n  - " + "\n  - ".join(failures))

    if champion is not None:
        client.set_registered_model_alias(REGISTERED_MODEL, "previous", champion.version)
    client.set_registered_model_alias(REGISTERED_MODEL, "champion", challenger.version)
    shutil.copyfile(models / "metrics.json", baseline_path)  # the gate now defends this release
    return challenger.version


def restore(cfg: dict[str, Any], version: Any) -> None:
    """Copy a registered release's files back into models/."""
    models = Path(cfg["paths"]["models"])
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(mlflow.artifacts.download_artifacts(artifact_uri=version.source, dst_path=tmp))
        for name in RELEASE_FILES:
            shutil.copyfile(local / name, models / name)
    shutil.copyfile(models / "metrics.json", models / "baseline-metrics.json")


def rollback(cfg: dict[str, Any]) -> str:
    """Point "champion" back at the previous release and restore its files to models/."""
    client = _client(cfg)
    champion, previous = _alias_version(client, "champion"), _alias_version(client, "previous")
    if previous is None:
        raise SystemExit("nothing to roll back to: there is no previous champion")
    client.set_registered_model_alias(REGISTERED_MODEL, "champion", previous.version)
    if champion is not None:
        # The release being withdrawn stays registered, as the challenger, for investigation.
        client.set_registered_model_alias(REGISTERED_MODEL, "challenger", champion.version)
    client.delete_registered_model_alias(REGISTERED_MODEL, "previous")
    restore(cfg, previous)
    return previous.version


def status(cfg: dict[str, Any]) -> dict[str, dict[str, str] | None]:
    client = _client(cfg)
    out = {}
    for alias in ("champion", "challenger", "previous"):
        version = _alias_version(client, alias)
        out[alias] = None if version is None else {"version": version.version, **version.tags}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["register", "promote", "rollback", "status"])
    args = parser.parse_args()
    cfg = load_config()
    if args.command == "status":
        for alias, info in status(cfg).items():
            print(f"{alias:11s} {'-' if info is None else json.dumps(info)}")
        return
    version = {"register": register, "promote": promote, "rollback": rollback}[args.command](cfg)
    done = {"register": "registered as challenger", "promote": "is now the champion",
            "rollback": "is the champion again; its files are restored in models/"}[args.command]
    print(f"version {version} {done}")
    if args.command != "register":
        print("Rebuild the extension to ship it: cd extension && pnpm build", file=sys.stderr)


if __name__ == "__main__":
    main()
