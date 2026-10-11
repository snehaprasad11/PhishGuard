"""Download dataset snapshots and pin them in the manifest.

This is the only module that touches the network. Everything downstream reads
the pinned files through `loaders.py`, which checks their sha256 against the
manifest — so a model is always traceable to the exact bytes it was trained on.

Usage (from the repo root):
    python -m training.data.fetch phishtank openphish tranco commoncrawl
    python -m training.data.fetch sms_spam nazario spamassassin
    python -m training.data.fetch --from-snapshot   # re-download the exact pinned files
"""

import argparse
import gzip
import hashlib
import io
import json
import logging
import random
import tarfile
import tempfile
import time
import zipfile
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from training.config import load_config
from training.data.domains import registered_domain

log = logging.getLogger("phishguard.fetch")

USER_AGENT = "phishguard-research/0.1 (student portfolio project)"
CC_CLUSTER_IDX = "https://data.commoncrawl.org/cc-index/collections/{crawl}/indexes/cluster.idx"


def _session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    # No transparent compression: Content-Length must equal the bytes we write,
    # or the resume logic in _download cannot tell when a file is complete.
    session.headers["Accept-Encoding"] = "identity"
    retry = Retry(total=5, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504))
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pin(cfg: dict[str, Any], name: str, source: str, path: Path, rows: int, **extra: Any) -> None:
    manifest_path = Path(cfg["paths"]["manifest"])
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}
    manifest[name] = {
        "source": source,
        "snapshot_date": datetime.now(UTC).date().isoformat(),
        "file": path.name,
        "sha256": sha256_of(path),
        "rows": rows,
        **extra,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8")
    log.info("pinned dataset=%s rows=%d file=%s", name, rows, path.name)


def _download(session: requests.Session, url: str, dest: Path) -> None:
    """Download url to dest, resuming from the last byte if the connection drops mid-body."""
    size = int(session.head(url, allow_redirects=True, timeout=60).headers.get("Content-Length", 0))
    with open(dest, "wb") as f:
        for attempt in range(8):
            try:
                headers = {"Range": f"bytes={f.tell()}-"} if f.tell() else {}
                with session.get(url, headers=headers, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as exc:
                log.warning("download interrupted attempt=%d at_byte=%d error=%s",
                            attempt + 1, f.tell(), type(exc).__name__)
                time.sleep(2**attempt)
                continue
            if not size or f.tell() == size:
                return
    raise OSError(f"could not download {url}")


def fetch_phishtank(cfg: dict[str, Any], session: requests.Session, raw: Path) -> None:
    url = cfg["data"]["phishtank"]["url"]
    dest = raw / f"phishtank-{datetime.now(UTC).date().isoformat()}.csv.gz"
    _download(session, url, dest)
    _pin(cfg, "phishtank", url, dest, rows=len(pd.read_csv(dest, usecols=["url"])))


def fetch_openphish(cfg: dict[str, Any], session: requests.Session, raw: Path) -> None:
    url = cfg["data"]["openphish"]["url"]
    dest = raw / f"openphish-{datetime.now(UTC).date().isoformat()}.txt"
    _download(session, url, dest)
    rows = sum(1 for line in dest.read_text("utf-8").splitlines() if line.strip())
    _pin(cfg, "openphish", url, dest, rows=rows)


def fetch_tranco(cfg: dict[str, Any], session: requests.Session, raw: Path) -> None:
    url = cfg["data"]["tranco"]["url"]
    dest = raw / f"tranco-{datetime.now(UTC).date().isoformat()}.csv.zip"
    _download(session, url, dest)
    with zipfile.ZipFile(dest) as z, z.open(z.namelist()[0]) as f:
        rows = sum(1 for _ in f)
    _pin(cfg, "tranco", url, dest, rows=rows)


def fetch_sms_spam(cfg: dict[str, Any], session: requests.Session, raw: Path) -> None:
    url = cfg["data"]["sms_spam"]["url"]
    dest = raw / "sms-spam-collection.zip"
    _download(session, url, dest)
    with zipfile.ZipFile(dest) as z, z.open("SMSSpamCollection") as f:
        rows = sum(1 for _ in f)
    _pin(cfg, "sms_spam", url, dest, rows=rows)


def fetch_nazario(cfg: dict[str, Any], session: requests.Session, raw: Path) -> None:
    source = cfg["data"]["nazario"]
    for year in source["years"]:
        url = f"{source['base_url']}phishing-{year}"
        dest = raw / f"nazario-phishing-{year}.mbox"
        _download(session, url, dest)
        # Each message in an mbox file starts with a line beginning "From ".
        with open(dest, "rb") as f:
            rows = sum(1 for line in f if line.startswith(b"From "))
        _pin(cfg, f"nazario_{year}", url, dest, rows=rows)


def fetch_spamassassin(cfg: dict[str, Any], session: requests.Session, raw: Path) -> None:
    source = cfg["data"]["spamassassin"]
    for archive in source["archives"]:
        url = f"{source['base_url']}{archive}.tar.bz2"
        dest = raw / f"spamassassin-{archive}.tar.bz2"
        _download(session, url, dest)
        with tarfile.open(dest, "r:bz2") as tar:
            rows = sum(1 for member in tar if member.isfile() and "cmds" not in member.name)
        _pin(cfg, f"spamassassin_{archive}", url, dest, rows=rows)


def surt_to_url(surt: str) -> str | None:
    """`uk,co,example,shop)/a/b?x=1` -> `shop.example.co.uk/a/b?x=1`.

    Common Crawl index keys are SURTs: host labels reversed, everything
    lowercased, scheme and a leading `www.` dropped. The result is therefore a
    normalised, scheme-less URL; the featuriser normalises phishing URLs the
    same way so the two classes stay comparable.
    """
    host_part, sep, path = surt.partition(")")
    if not sep or not host_part:
        return None
    reversed_host, _, port = host_part.partition(":")
    host = ".".join(reversed(reversed_host.split(",")))
    return host + (f":{port}" if port else "") + path


def _cluster_idx_pages(path: Path) -> Iterator[tuple[str, str]]:
    """Yield (url, timestamp) from a crawl's cluster.idx.

    cluster.idx holds the first entry of every ~3000-URL block of the crawl's
    sorted URL index, one per line: `<surt> <timestamp>\\t<file>\\t<offset>\\t...`.
    That makes it an evenly spaced 1-in-3000 sample of everything crawled.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            surt, _, timestamp = line.split("\t", 1)[0].rpartition(" ")
            url = surt_to_url(surt)
            if url and not url.endswith("/robots.txt"):
                yield url, timestamp


TIERS = ("top100k", "top1m", "unranked")


def popularity_tier(rank: int | None, top_n: int) -> str:
    """Which popularity band a domain falls in, given its Tranco rank (None = not listed)."""
    if rank is None:
        return "unranked"
    return "top100k" if rank <= top_n else "top1m"


def fetch_commoncrawl(cfg: dict[str, Any], session: requests.Session, raw: Path) -> None:
    """Sample benign URLs from Common Crawl in three popularity tiers.

    * top100k  — up to `max_urls_per_domain` URLs for every popular domain seen.
    * top1m    — one URL each from a random sample of less popular ranked domains.
    * unranked — one URL each from a random sample of domains Tranco does not list.

    Without the last two the model never sees a small legitimate site, and its
    false-positive rate on them cannot be measured.
    """
    cc = cfg["data"]["commoncrawl"]
    top_n = cfg["data"]["tranco"]["top_n"]
    ranks = _tranco_ranks(cfg)
    rng = random.Random(cfg["seed"])

    rows: list[tuple[str, str, str, str]] = []
    for crawl in cc["crawls"]:
        by_tier: dict[str, dict[str, set[tuple[str, str]]]] = {tier: {} for tier in TIERS}
        # ~100 MB of index we only need once: keep it out of the repo folder.
        with tempfile.TemporaryDirectory() as tmp:
            idx = Path(tmp) / "cluster.idx"
            _download(session, CC_CLUSTER_IDX.format(crawl=crawl), idx)
            for url, timestamp in _cluster_idx_pages(idx):
                domain = registered_domain(url)
                if domain:
                    tier = popularity_tier(ranks.get(domain), top_n)
                    by_tier[tier].setdefault(domain, set()).add((url, timestamp))

        # Sorted iteration + seeded rng => the same sample on every run of the same crawl.
        for tier in TIERS:
            domains = sorted(by_tier[tier])
            per_domain = cc["max_urls_per_domain"]
            if tier != "top100k":
                domains = sorted(rng.sample(domains, min(cc["long_tail_domains"], len(domains))))
                per_domain = 1
            kept = 0
            for domain in domains:
                pages = sorted(by_tier[tier][domain])
                keep = rng.sample(pages, min(per_domain, len(pages)))
                rows.extend((url, timestamp, crawl, tier) for url, timestamp in keep)
                kept += len(keep)
            log.info("crawl=%s tier=%s domains_seen=%d urls_kept=%d",
                     crawl, tier, len(by_tier[tier]), kept)

    frame = pd.DataFrame(rows, columns=["url", "timestamp", "crawl", "tier"])
    dest = raw / f"commoncrawl-{'_'.join(c.removeprefix('CC-MAIN-') for c in cc['crawls'])}.csv.gz"
    buffer = io.BytesIO()
    # mtime=0 keeps the gzip header constant so identical rows give an identical sha256.
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as gz:
        gz.write(frame.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    dest.write_bytes(buffer.getvalue())
    _pin(
        cfg, "commoncrawl", "https://data.commoncrawl.org/cc-index/collections/", dest,
        rows=len(frame), crawls=cc["crawls"], max_urls_per_domain=cc["max_urls_per_domain"],
        long_tail_domains=cc["long_tail_domains"], tranco_top_n=top_n, seed=cfg["seed"],
    )


def _tranco_ranks(cfg: dict[str, Any]) -> dict[str, int]:
    # Imported here: loaders verifies the pinned Tranco file before we rank against it.
    from training.data.loaders import load_tranco

    tranco = load_tranco(cfg)
    return dict(zip(tranco["domain"], tranco["rank"], strict=True))


def restore_snapshot(cfg: dict[str, Any], session: requests.Session, raw: Path) -> None:
    """Download every file pinned in the manifest from the published snapshot and verify it.

    Live feeds change hourly, so re-fetching them can never reproduce a pinned
    hash; this is how another machine (or CI) gets the exact training data.
    """
    base = cfg["data"].get("snapshot_base_url")
    if not base:
        raise SystemExit("data.snapshot_base_url is not set in config.yaml")
    manifest = json.loads(Path(cfg["paths"]["manifest"]).read_text("utf-8"))
    for name, entry in manifest.items():
        dest = raw / entry["file"]
        if not (dest.exists() and sha256_of(dest) == entry["sha256"]):
            _download(session, f"{base.rstrip('/')}/{entry['file']}", dest)
        if sha256_of(dest) != entry["sha256"]:
            raise SystemExit(f"{dest} does not match the sha256 pinned for '{name}'")
        log.info("restored dataset=%s file=%s", name, entry["file"])


FETCHERS = {
    "phishtank": fetch_phishtank,
    "openphish": fetch_openphish,
    "tranco": fetch_tranco,
    "commoncrawl": fetch_commoncrawl,
    "sms_spam": fetch_sms_spam,
    "nazario": fetch_nazario,
    "spamassassin": fetch_spamassassin,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="*", choices=list(FETCHERS))
    parser.add_argument("--from-snapshot", action="store_true")
    args = parser.parse_args()
    if bool(args.datasets) == args.from_snapshot:
        parser.error("name datasets to fetch live, or pass --from-snapshot (not both)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    cfg = load_config()
    raw = Path(cfg["paths"]["raw_data"])
    raw.mkdir(parents=True, exist_ok=True)
    session = _session()
    if args.from_snapshot:
        restore_snapshot(cfg, session, raw)
    for name in args.datasets:
        FETCHERS[name](cfg, session, raw)


if __name__ == "__main__":
    main()
