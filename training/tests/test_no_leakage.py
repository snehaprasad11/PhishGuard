import pandas as pd
import pytest

from training.data.split import LeakageError, assert_no_leakage, build_dataset

VAL_START = pd.Timestamp("2026-07-01", tz="UTC")
TEST_START = pd.Timestamp("2026-08-01", tz="UTC")
CFG = {
    "seed": 42,
    "split": {
        "phishing_min_date": "2026-01-01",
        "val_start": "2026-07-01",
        "test_start": "2026-08-01",
        "benign_fractions": [0.6, 0.2, 0.2],
        "max_urls_per_site": 3,
    },
}


def _raw_urls() -> pd.DataFrame:
    rows = []
    # 40 phishing sites, one new site every 6 days from March, two URLs each a day apart.
    for i in range(40):
        seen = pd.Timestamp("2026-03-01", tz="UTC") + pd.Timedelta(days=6 * i)
        later = seen + pd.Timedelta(days=1)
        rows.append((f"http://phish{i}-bad.shop/login", seen, 1))
        rows.append((f"https://www.phish{i}-bad.shop/verify?id={i}", later, 1))
    # A shared host abused before AND after both cut-offs.
    for day in ["2026-04-01", "2026-07-15", "2026-08-15"]:
        rows.append((f"https://sites.sharedhost.com/view/{day}", pd.Timestamp(day, tz="UTC"), 1))
    # 60 benign sites, five deep pages each (more than the per-site cap).
    for i in range(60):
        for page in range(5):
            rows.append((f"good{i}.org/docs/page-{page}", pd.Timestamp("2026-08-10", tz="UTC"), 0))
    rows.append(("sharedhost.com/about", pd.Timestamp("2026-08-10", tz="UTC"), 0))
    # Stale phishing, an unparseable URL, and one URL carrying both labels.
    rows.append(("http://ancient-bad.shop/", pd.Timestamp("2020-01-01", tz="UTC"), 1))
    rows.append(("https://", pd.Timestamp("2026-08-10", tz="UTC"), 0))
    rows.append(("http://both.example.net/x", pd.Timestamp("2026-05-01", tz="UTC"), 1))
    rows.append(("both.example.net/x", pd.Timestamp("2026-08-10", tz="UTC"), 0))
    frame = pd.DataFrame(rows, columns=["url", "timestamp", "label"])
    frame["source"] = "fixture"
    return frame


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return build_dataset(_raw_urls(), CFG)


def test_no_site_appears_in_more_than_one_split(dataset):
    assert dataset.groupby("site")["split"].nunique().max() == 1
    assert set(dataset["split"]) == {"train", "val", "test"}


def test_phishing_respects_time_boundaries(dataset):
    phishing = dataset[dataset["label"] == 1]
    by_split = {name: group["timestamp"] for name, group in phishing.groupby("split")}
    assert by_split["train"].max() < VAL_START
    assert VAL_START <= by_split["val"].min() and by_split["val"].max() < TEST_START
    assert by_split["test"].min() >= TEST_START


def test_shared_host_stays_in_one_split_without_future_phishing(dataset):
    shared = dataset[dataset["site"] == "sharedhost.com"]
    assert set(shared["split"]) == {"train"}           # first abused in April
    assert set(shared[shared["label"] == 1]["url"]) == {"sites.sharedhost.com/view/2026-04-01"}
    assert "sharedhost.com/about" in set(shared["url"])  # its benign page follows the site


def test_every_benign_site_gets_a_homepage_and_respects_the_cap(dataset):
    benign = dataset[dataset["label"] == 0]
    assert "good0.org/" in set(benign["url"])
    per_site = benign.groupby("site").size()
    assert per_site.max() <= CFG["split"]["max_urls_per_site"]
    assert benign[benign["url"].str.endswith(".org/")]["site"].nunique() == 60


def test_urls_are_normalised_and_bad_rows_dropped(dataset):
    urls = set(dataset["url"])
    assert "phish0-bad.shop/verify?id=0" in urls   # scheme and www. removed
    assert not any(u.startswith(("http://", "https://", "www.")) for u in urls)
    assert "ancient-bad.shop/" not in urls         # older than phishing_min_date
    assert "both.example.net/x" not in urls                # conflicting labels
    assert dataset["url"].is_unique


def test_build_is_deterministic():
    first, second = build_dataset(_raw_urls(), CFG), build_dataset(_raw_urls(), CFG)
    pd.testing.assert_frame_equal(first, second)


def test_leakage_check_catches_a_site_in_two_splits(dataset):
    leaked = dataset.copy()
    victim = leaked[leaked["split"] == "train"].iloc[0]
    leaked = pd.concat([leaked, victim.to_frame().T.assign(split="test")], ignore_index=True)
    with pytest.raises(LeakageError, match="more than one split"):
        assert_no_leakage(leaked, VAL_START, TEST_START)


def test_leakage_check_catches_future_phishing_in_train(dataset):
    leaked = dataset.copy()
    row = leaked[(leaked["split"] == "train") & (leaked["label"] == 1)].index[0]
    leaked.loc[row, "timestamp"] = TEST_START
    with pytest.raises(LeakageError, match="train contains phishing"):
        assert_no_leakage(leaked, VAL_START, TEST_START)
