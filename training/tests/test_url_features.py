import math

import numpy as np
import pytest

from training.features.url_features import (
    BASE_FEATURE_NAMES,
    FEATURE_NAMES,
    extract_features,
    featurise,
    on_known_site,
    parse_url,
    shannon_entropy,
)


def test_hand_checked_ip_login_url():
    # normalised: "192.168.0.1/login" -> host "192.168.0.1" (11 chars, 8 digits), path "/login"
    features = extract_features("http://192.168.0.1/login")
    assert {name: features[name] for name in BASE_FEATURE_NAMES} == {
        "host_length": math.log1p(11),
        "host_label_count": 4.0,
        "host_digit_ratio": 8 / 11,
        # '1' x3, '.' x3, five characters once each, 11 in total
        "host_entropy": pytest.approx(
            2 * (3 / 11) * math.log2(11 / 3) + 5 * (1 / 11) * math.log2(11)
        ),
        "host_is_ip": 1.0,
        "host_hyphen_count": 0.0,
        "host_is_idn": 0.0,
        "keyword_hits_host": 0.0,
        "tld_legacy": 0.0,
        "tld_country": 0.0,
        "tld_other": 0.0,           # an IP has no TLD
        "on_known_site": 0.0,
        "has_at": 0.0,
        "is_bare_host": 0.0,
        "path_length": math.log1p(6),
        "path_depth": math.log1p(1),
        "path_digit_ratio": 0.0,
        "path_hyphen_count": 0.0,
        "keyword_hits_path": 1.0,   # "login"
        "query_length": 0.0,
    }


def test_hand_checked_lookalike_url():
    # host "secure-login.paypa1.example.shop" (32 chars, 1 digit)
    # tail "/verify/account-update?id=42&x=1" (32 chars, 3 digits), path is 22 chars
    f = extract_features(
        "HTTPS://www.Secure-Login.paypa1.example.shop/verify/account-update?id=42&x=1#top"
    )
    assert f["host_length"] == math.log1p(32)
    assert f["host_label_count"] == 4.0
    assert f["host_digit_ratio"] == 1 / 32
    assert f["host_hyphen_count"] == math.log1p(1)
    assert f["keyword_hits_host"] == 2.0   # secure, login
    assert (f["tld_legacy"], f["tld_country"], f["tld_other"]) == (0.0, 0.0, 1.0)
    assert f["is_bare_host"] == 0.0
    assert f["path_length"] == math.log1p(22)
    assert f["path_depth"] == math.log1p(2)
    assert f["path_digit_ratio"] == 3 / 32
    assert f["path_hyphen_count"] == math.log1p(1)
    assert f["keyword_hits_path"] == 3.0   # verify, account, update
    assert f["query_length"] == math.log1p(9)


@pytest.mark.parametrize(
    ("url", "bare"),
    [("https://example.com", 1.0), ("example.com/", 1.0), ("example.com/?x=1", 0.0),
     ("example.com/a", 0.0)],
)
def test_is_bare_host(url, bare):
    assert extract_features(url)["is_bare_host"] == bare


@pytest.mark.parametrize(
    "variant",
    [
        "example.com/a/b?x=1",
        "https://example.com/a/b?x=1",
        "http://www.example.com/a/b?x=1",
        "HTTPS://WWW.EXAMPLE.COM:443/a/b?x=1#section",
        "  http://example.com:80/a/b?x=1  ",
    ],
)
def test_formatting_differences_do_not_change_features(variant):
    """Benign data is stored normalised; raw URLs must land on identical features."""
    assert extract_features(variant) == extract_features("example.com/a/b?x=1")


@pytest.mark.parametrize(
    ("url", "normalised", "host", "path", "query"),
    [
        ("https://example.com", "example.com/", "example.com", "/", ""),
        ("https://example.com?x=1", "example.com/?x=1", "example.com", "/", "x=1"),
        ("http://example.com:8080/a", "example.com:8080/a", "example.com", "/a", ""),
        ("http://user@www.example.com/a", "user@example.com/a", "example.com", "/a", ""),
        ("http://[2001:db8::1]:8080/a", "[2001:db8::1]:8080/a", "[2001:db8::1]", "/a", ""),
        ("http://example.com/a@b", "example.com/a@b", "example.com", "/a@b", ""),
    ],
)
def test_parse_url(url, normalised, host, path, query):
    parsed = parse_url(url)
    assert parsed.normalised == normalised
    assert (parsed.host, parsed.path, parsed.query) == (host, path, query)


@pytest.mark.parametrize(
    ("url", "flag", "expected"),
    [
        ("http://[2001:db8::1]/a", "host_is_ip", 1.0),
        ("http://1.2.3.4.example.com/", "host_is_ip", 0.0),
        ("http://paypal.com@evil.example/", "has_at", 1.0),
        ("http://xn--pypal-4ve.com/", "host_is_idn", 1.0),
        ("http://pаypal.com/", "host_is_idn", 1.0),  # Cyrillic "а"
        ("http://paypal.com/", "host_is_idn", 0.0),
        ("http://example.co.uk/", "tld_country", 1.0),
        ("http://example.org/", "tld_legacy", 1.0),
        ("http://localhost/", "tld_other", 0.0),
    ],
)
def test_flags(url, flag, expected):
    assert extract_features(url)[flag] == expected


def test_tld_buckets_are_mutually_exclusive():
    for url in ["a.com/", "a.de/", "a.shop/", "1.2.3.4/", "localhost/"]:
        f = extract_features(url)
        assert f["tld_legacy"] + f["tld_country"] + f["tld_other"] <= 1.0


def test_only_ascii_is_lowercased():
    # Dotted capital I lowercases differently in Python and JavaScript, so it is left alone.
    assert parse_url("http://example.com/İ").path == "/İ"
    assert parse_url("http://EXAMPLE.com/ABC").normalised == "example.com/abc"


def test_shannon_entropy():
    assert shannon_entropy("") == 0.0
    assert shannon_entropy("aaaa") == 0.0
    assert shannon_entropy("ab") == 1.0
    assert shannon_entropy("abcd") == 2.0


def test_empty_url_is_rejected():
    for empty in ["", "   ", "https://", "#fragment"]:
        with pytest.raises(ValueError, match="empty URL"):
            extract_features(empty)


def test_featurise_shape_order_and_determinism():
    urls = ["http://192.168.0.1/login", "https://example.com/a?x=1", "weird///??@@"]
    matrix = featurise(urls)
    assert matrix.shape == (3, len(FEATURE_NAMES))
    assert np.isfinite(matrix).all()
    assert np.array_equal(matrix, featurise(urls))
    assert list(extract_features(urls[0])) == list(FEATURE_NAMES)
    assert matrix[0, FEATURE_NAMES.index("host_is_ip")] == 1.0
    assert math.isclose(matrix[1, FEATURE_NAMES.index("query_length")], math.log1p(3))


KNOWN = frozenset({"google.com", "example.co.uk"})


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("google.com", True),
        ("accounts.google.com", True),
        ("a.b.example.co.uk", True),
        ("google.com.evil.shop", False),    # known name used as a subdomain of another site
        ("notgoogle.com", False),           # shares a suffix of characters, not of labels
        ("co.uk", False),
        ("com", False),
        ("192.168.0.1", False),
    ],
)
def test_on_known_site(host, expected):
    assert on_known_site(host, KNOWN) is expected


def test_keywords_are_not_counted_on_known_sites():
    real = extract_features("https://accounts.google.com/signin", KNOWN)
    fake = extract_features("https://accounts.google.com.signin-verify.shop/signin", KNOWN)
    names = ("on_known_site", "keyword_hits_host", "keyword_hits_path")
    assert tuple(real[n] for n in names) == (1, 0, 0)
    assert tuple(fake[n] for n in names) == (0, 3, 1)   # account, signin, verify / signin


def test_without_a_known_sites_list_nothing_is_known():
    f = extract_features("https://accounts.google.com/signin")
    assert (f["on_known_site"], f["keyword_hits_host"], f["keyword_hits_path"]) == (0, 1, 1)


def test_featurise_of_nothing_keeps_its_shape():
    assert featurise([]).shape == (0, len(FEATURE_NAMES))


def test_untrusted_host_on_a_known_domain():
    known = frozenset({"google.com", "!sites.google.com", "!google.com"})
    assert on_known_site("accounts.google.com", known) is True
    assert on_known_site("sites.google.com", known) is False
    assert on_known_site("google.com", known) is False      # exact host only
    f = extract_features("https://sites.google.com/view/secure-login", known)
    assert (f["on_known_site"], f["keyword_hits_path"]) == (0, 2)


def test_bare_host_columns_mirror_host_features_only_on_bare_hosts():
    bare = extract_features("https://secure-login.example.shop/")
    deep = extract_features("https://secure-login.example.shop/a")
    assert bare["host_length:bare_host"] == bare["host_length"] == math.log1p(25)
    assert bare["keyword_hits_host:bare_host"] == 2.0
    assert all(value == 0.0 for name, value in deep.items() if name.endswith(":bare_host"))
    assert "path_length:bare_host" not in bare        # path features do not vary on a bare host
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == len(bare)
