"""Load the pinned dataset snapshots into DataFrames.

Every loader verifies the file's sha256 and row count against the manifest and
checks the schema, and raises `DataError` on any mismatch — the pipeline fails
loudly rather than silently training on the wrong or malformed data (spec §13).

Usage (from the repo root):
    python -m training.data.loaders      # prints class balance
"""

import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from training.config import load_config
from training.data.fetch import sha256_of

PHISHING, BENIGN = 1, 0


class DataError(Exception):
    pass


def _pinned_file(cfg: dict[str, Any], name: str) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(cfg["paths"]["manifest"])
    if not manifest_path.exists():
        raise DataError(f"no manifest at {manifest_path}; run `python -m training.data.fetch`")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if name not in manifest:
        raise DataError(f"dataset '{name}' is not in the manifest; fetch it first")
    entry = manifest[name]
    path = Path(cfg["paths"]["raw_data"]) / entry["file"]
    if not path.exists():
        raise DataError(f"{path} is pinned in the manifest but missing on disk")
    if sha256_of(path) != entry["sha256"]:
        raise DataError(f"{path} does not match the sha256 pinned in the manifest")
    return path, entry


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise DataError(f"{name}: missing columns {sorted(missing)}")


def _check_rows(frame: pd.DataFrame, entry: dict[str, Any], name: str) -> None:
    if len(frame) != entry["rows"]:
        raise DataError(f"{name}: {len(frame)} rows on disk, manifest says {entry['rows']}")


def load_phishing(cfg: dict[str, Any]) -> pd.DataFrame:
    path, entry = _pinned_file(cfg, "phishtank")
    frame = pd.read_csv(path)
    _require_columns(frame, {"url", "submission_time", "verified", "online"}, "phishtank")
    _check_rows(frame, entry, "phishtank")
    unexpected = set(frame["verified"].unique()) - {"yes"}
    if unexpected:
        raise DataError(f"phishtank: unexpected 'verified' values {sorted(unexpected)}")
    return pd.DataFrame({
        "url": frame["url"],
        "timestamp": pd.to_datetime(frame["submission_time"], utc=True),
        "source": "phishtank",
        "label": PHISHING,
        "tier": "phishing",
    })


def load_benign(cfg: dict[str, Any]) -> pd.DataFrame:
    path, entry = _pinned_file(cfg, "commoncrawl")
    frame = pd.read_csv(path, dtype=str)
    _require_columns(frame, {"url", "timestamp", "crawl", "tier"}, "commoncrawl")
    _check_rows(frame, entry, "commoncrawl")
    return pd.DataFrame({
        "url": frame["url"],
        "timestamp": pd.to_datetime(frame["timestamp"], format="%Y%m%d%H%M%S", utc=True),
        "source": frame["crawl"],
        "label": BENIGN,
        "tier": frame["tier"],
    })


def load_tranco(cfg: dict[str, Any]) -> pd.DataFrame:
    path, entry = _pinned_file(cfg, "tranco")
    with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as f:
        frame = pd.read_csv(f, names=["rank", "domain"], dtype={"rank": int, "domain": str})
    _check_rows(frame, entry, "tranco")
    return frame


def _url_lines(text: str) -> list[str]:
    lines = (line.strip() for line in text.splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def load_openphish(cfg: dict[str, Any]) -> list[str]:
    """Phishing URLs from a second, independent feed. For external checks only."""
    path, entry = _pinned_file(cfg, "openphish")
    urls = _url_lines(path.read_text("utf-8"))
    if len(urls) != entry["rows"]:
        raise DataError(f"openphish: {len(urls)} rows on disk, manifest says {entry['rows']}")
    return urls


def load_curated_benign(cfg: dict[str, Any], holdout: bool = False) -> list[str]:
    """Hand-picked legitimate pages (banks, sign-in, government) that must not be flagged.

    `holdout=True` loads the second list, which is reported but never used for decisions.
    """
    key = "curated_benign_holdout" if holdout else "curated_benign"
    urls = _url_lines(Path(cfg["paths"][key]).read_text("utf-8"))
    if not urls:
        raise DataError("curated benign list is empty")
    return urls


def load_urls(cfg: dict[str, Any]) -> pd.DataFrame:
    """All labelled URLs: columns url, timestamp (UTC), source, label (1 = phishing), tier.

    `tier` is the popularity band of a benign URL's domain (top100k / top1m / unranked).
    """
    frame = pd.concat([load_phishing(cfg), load_benign(cfg)], ignore_index=True)
    if frame["url"].isna().any():
        raise DataError("null URLs in the labelled data")
    if set(frame["label"].unique()) != {PHISHING, BENIGN}:
        raise DataError("both classes must be present")
    return frame


if __name__ == "__main__":
    urls = load_urls(load_config())
    summary = urls.groupby("label").agg(
        rows=("url", "size"), earliest=("timestamp", "min"), latest=("timestamp", "max")
    )
    print(summary.rename(index={PHISHING: "phishing", BENIGN: "benign"}).to_string())
