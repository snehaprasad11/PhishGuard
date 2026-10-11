"""Assemble the labelled URLs into leakage-safe train / validation / test splits.

Two kinds of leakage are guarded against:

* Site leakage — every URL of a site (registered domain) lands in exactly one
  split, so the test set only contains sites the model has never seen.
* Time leakage — phishing is split by submission date, and a split never
  contains phishing submitted after its own cut-off.

Benign URLs come from a ten-week crawl window, too short to split by time, so
benign-only sites are assigned by a stable hash of the site name instead.
"""

import hashlib
import logging
from typing import Any

import numpy as np
import pandas as pd

from training.data.domains import registered_domain
from training.data.loaders import BENIGN, PHISHING
from training.features.url_features import parse_url

log = logging.getLogger("phishguard.split")

SPLITS = ("train", "val", "test")


class LeakageError(Exception):
    pass


def _normalise(url: str) -> str | None:
    try:
        return parse_url(url).normalised
    except ValueError:
        return None


def _hash_fraction(site: str, seed: int) -> float:
    """A stable number in [0, 1) for a site: same site + seed => same split, on any machine."""
    digest = hashlib.sha256(f"{seed}:{site}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def build_dataset(urls: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Return columns url (normalised), label, timestamp, site, split, tier."""
    split_cfg = cfg["split"]
    val_start = pd.Timestamp(split_cfg["val_start"], tz="UTC")
    test_start = pd.Timestamp(split_cfg["test_start"], tz="UTC")
    min_date = pd.Timestamp(split_cfg["phishing_min_date"], tz="UTC")

    frame = urls[(urls["label"] == BENIGN) | (urls["timestamp"] >= min_date)].copy()
    frame["url"] = frame["url"].map(_normalise)
    frame = frame.dropna(subset=["url"])
    frame["is_homepage"] = False
    if "tier" not in frame:
        frame["tier"] = np.where(frame["label"] == BENIGN, "top100k", "phishing")

    # Crawled benign URLs are almost all deep pages; without homepages the model
    # would learn "no path => phishing" and flag every site's front page.
    benign = frame[frame["label"] == BENIGN]
    domains = benign["url"].map(registered_domain)
    homepages = pd.DataFrame({
        "url": domains + "/",
        "timestamp": benign["timestamp"],
        "source": "homepage",
        "label": BENIGN,
        "tier": benign["tier"],
        "is_homepage": True,
    }).dropna(subset=["url"]).groupby("url", as_index=False).min()
    frame = pd.concat([frame, homepages], ignore_index=True)

    # A URL carrying both labels has no usable ground truth: drop every copy.
    ambiguous = frame.groupby("url")["label"].nunique()
    frame = frame[~frame["url"].isin(ambiguous[ambiguous > 1].index)]
    frame = frame.sort_values(["url", "timestamp"]).drop_duplicates("url", keep="first")

    frame["site"] = frame["url"].map(lambda u: registered_domain(u, private=True))
    frame = frame.dropna(subset=["site"])

    # Cap URLs per (site, class) so a few huge sites cannot dominate; homepages always survive.
    rng = np.random.default_rng(cfg["seed"])
    frame["_order"] = rng.random(len(frame))
    frame = (
        frame.sort_values(["is_homepage", "_order"], ascending=[False, True])
        .groupby(["site", "label"], sort=False)
        .head(split_cfg["max_urls_per_site"])
    )

    # A site with any phishing is placed by its earliest submission date; the rest by hash.
    first_phish = frame[frame["label"] == PHISHING].groupby("site")["timestamp"].min()
    train_frac, val_frac, _ = split_cfg["benign_fractions"]

    def assign(site: str) -> str:
        if site in first_phish.index:
            seen = first_phish[site]
            return "train" if seen < val_start else "val" if seen < test_start else "test"
        fraction = _hash_fraction(site, cfg["seed"])
        if fraction < train_frac:
            return "train"
        return "val" if fraction < train_frac + val_frac else "test"

    site_split = {site: assign(site) for site in frame["site"].unique()}
    frame["split"] = frame["site"].map(site_split)

    # Phishing submitted after a split's cut-off would be information from its future.
    cutoff = frame["split"].map({"train": val_start, "val": test_start})
    from_future = (frame["label"] == PHISHING) & cutoff.notna() & (frame["timestamp"] >= cutoff)
    frame = frame[~from_future]

    frame = frame.sort_values("url").reset_index(drop=True)
    frame = frame[["url", "label", "timestamp", "site", "split", "tier"]]
    assert_no_leakage(frame, val_start, test_start)
    return frame


def assert_no_leakage(
    frame: pd.DataFrame, val_start: pd.Timestamp, test_start: pd.Timestamp
) -> None:
    """Raise LeakageError if a site spans splits or a split holds phishing from its future."""
    spans = frame.groupby("site")["split"].nunique()
    if (spans > 1).any():
        raise LeakageError(f"sites in more than one split: {sorted(spans[spans > 1].index)[:5]}")

    phishing = frame[frame["label"] == PHISHING]
    latest = phishing.groupby("split")["timestamp"].max()
    earliest = phishing.groupby("split")["timestamp"].min()
    if latest.get("train", val_start - pd.Timedelta(1)) >= val_start:
        raise LeakageError("train contains phishing submitted on or after val_start")
    if latest.get("val", test_start - pd.Timedelta(1)) >= test_start:
        raise LeakageError("val contains phishing submitted on or after test_start")
    if earliest.get("val", val_start) < val_start or earliest.get("test", test_start) < test_start:
        raise LeakageError("a later split contains phishing from before its start date")
