import gzip
import json

import pandas as pd
import pytest

from training.data.domains import is_public_suffix, registered_domain
from training.data.fetch import _cluster_idx_pages, popularity_tier, sha256_of, surt_to_url
from training.data.known_sites import find_open_hosts, select_known_sites
from training.data.loaders import BENIGN, PHISHING, DataError, load_phishing, load_urls

PHISHTANK_CSV = (
    "phish_id,url,phish_detail_url,submission_time,verified,verification_time,online,target\n"
    "1,http://192.168.0.1/login,x,2026-08-01T10:00:00+00:00,yes,x,yes,Other\n"
    '2,"https://paypa1.example/a,b",x,2026-09-01T10:00:00+00:00,yes,x,yes,PayPal\n'
)
COMMONCRAWL_CSV = (
    "url,timestamp,crawl,tier\n"
    "example.org/docs/intro,20260905120000,CC-MAIN-2026-39,top1m\n"
)


@pytest.fixture
def cfg(tmp_path):
    """A config pointing at a tiny pinned snapshot of each labelled dataset."""
    raw = tmp_path / "raw"
    raw.mkdir()
    manifest = {}
    for name, text, rows in [("phishtank", PHISHTANK_CSV, 2), ("commoncrawl", COMMONCRAWL_CSV, 1)]:
        path = raw / f"{name}.csv.gz"
        path.write_bytes(gzip.compress(text.encode()))
        manifest[name] = {"file": path.name, "sha256": sha256_of(path), "rows": rows}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return {"paths": {"raw_data": str(raw), "manifest": str(manifest_path)}}


def _rewrite(cfg, name, text):
    """Replace a dataset's file and re-pin its hash, leaving the manifest row count alone."""
    path = f"{cfg['paths']['raw_data']}/{name}.csv.gz"
    with open(path, "wb") as f:
        f.write(gzip.compress(text.encode()))
    with open(cfg["paths"]["manifest"]) as f:
        manifest = json.load(f)
    manifest[name]["sha256"] = sha256_of(path)
    with open(cfg["paths"]["manifest"], "w") as f:
        json.dump(manifest, f)


def test_load_urls_has_both_classes_and_utc_timestamps(cfg):
    urls = load_urls(cfg)
    assert urls["label"].value_counts().to_dict() == {PHISHING: 2, BENIGN: 1}
    assert str(urls["timestamp"].dt.tz) == "UTC"
    assert "https://paypa1.example/a,b" in set(urls["url"])  # quoted comma survives parsing
    assert set(urls["tier"]) == {"phishing", "top1m"}


def test_file_changed_after_pinning_is_rejected(cfg):
    path = f"{cfg['paths']['raw_data']}/phishtank.csv.gz"
    with open(path, "wb") as f:
        f.write(gzip.compress((PHISHTANK_CSV + "3,http://x.test,x,2026-01-01,yes,x,yes,Other\n").encode()))
    with pytest.raises(DataError, match="sha256"):
        load_phishing(cfg)


def test_missing_column_is_rejected(cfg):
    _rewrite(cfg, "phishtank", PHISHTANK_CSV.replace("submission_time", "submitted"))
    with pytest.raises(DataError, match="missing columns"):
        load_phishing(cfg)


def test_row_count_mismatch_is_rejected(cfg):
    _rewrite(cfg, "phishtank", PHISHTANK_CSV.rsplit("\n", 2)[0] + "\n")
    with pytest.raises(DataError, match="manifest says 2"):
        load_phishing(cfg)


def test_unverified_phish_is_rejected(cfg):
    _rewrite(cfg, "phishtank", PHISHTANK_CSV.replace(",yes,x,yes,Other", ",no,x,yes,Other"))
    with pytest.raises(DataError, match="unexpected 'verified'"):
        load_phishing(cfg)


def test_dataset_missing_from_manifest_is_rejected(cfg):
    with open(cfg["paths"]["manifest"], "w") as f:
        json.dump({}, f)
    with pytest.raises(DataError, match="not in the manifest"):
        load_phishing(cfg)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://login.pay.example.co.uk/x?y=1", "example.co.uk"),
        ("http://192.168.0.1/login", "192.168.0.1"),
        ("https://WWW.Example.COM/", "example.com"),
    ],
)
def test_registered_domain(url, expected):
    assert registered_domain(url) == expected


@pytest.mark.parametrize(
    ("surt", "expected"),
    [
        ("uk,co,example,shop)/a/b?x=1", "shop.example.co.uk/a/b?x=1"),
        ("com,example)/", "example.com/"),
        ("com,example:8080)/admin", "example.com:8080/admin"),
        ("not a surt", None),
    ],
)
def test_surt_to_url(surt, expected):
    assert surt_to_url(surt) == expected


def test_cluster_idx_pages_skips_robots_and_keeps_timestamps(tmp_path):
    idx = tmp_path / "cluster.idx"
    idx.write_text(
        "com,example)/docs/a b?x=1 20260905120000\tcdx-00001.gz\t0\t100\t1\n"
        "com,example)/robots.txt 20260905120001\tcdx-00001.gz\t100\t100\t2\n"
    )
    assert list(_cluster_idx_pages(idx)) == [("example.com/docs/a b?x=1", "20260905120000")]


def test_private_suffixes_make_shared_hosts_separate_sites():
    assert registered_domain("https://shop1.web.app/x") == "web.app"
    assert registered_domain("https://shop1.web.app/x", private=True) == "shop1.web.app"
    assert is_public_suffix("web.app") and is_public_suffix("co.uk")
    assert not is_public_suffix("google.com")


def test_known_sites_are_top_ranked_and_exclude_hosting_platforms():
    tranco = pd.DataFrame({
        "rank": [1, 2, 3, 4, 5],
        "domain": ["google.com", "web.app", "zebra.org", "apple.com", "toolow.net"],
    })
    assert select_known_sites(tranco, top_n=4) == ["apple.com", "google.com", "zebra.org"]


def test_popularity_tier():
    assert popularity_tier(1, top_n=100) == "top100k"
    assert popularity_tier(100, top_n=100) == "top100k"
    assert popularity_tier(101, top_n=100) == "top1m"
    assert popularity_tier(None, top_n=100) == "unranked"


def test_open_hosts_need_enough_phishing_inside_the_training_period():
    def day(d):
        return pd.Timestamp(d, tz="UTC")

    phishing = pd.DataFrame({
        "url": (
            [f"https://sites.google.com/view/p{i}" for i in range(3)]      # 3 in period
            + ["https://docs.google.com/forms/x"] * 2                      # only 2: below threshold
            + [f"https://forms.office.com/r/{i}" for i in range(3)]        # too late
            + [f"https://evil{i}.shop/" for i in range(5)]                 # not on a known domain
            + ["https://"]                                                 # unparseable
        ),
        "timestamp": [day("2026-03-01")] * 5 + [day("2026-08-01")] * 3 + [day("2026-03-01")] * 6,
    })
    known = frozenset({"google.com", "office.com"})
    assert find_open_hosts(
        phishing, known, start=day("2026-01-01"), end=day("2026-07-01"), min_phishing=3
    ) == ["sites.google.com"]
