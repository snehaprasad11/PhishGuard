"""Turn a URL into a fixed-order vector of hand-engineered features.

Design rules (they exist because this file is reimplemented in TypeScript for
the browser, and the two must agree exactly):

* Pure string logic only — no Public Suffix List, no locale-dependent calls.
* Only ASCII A-Z is lowercased; lengths count Unicode code points.
* `FEATURE_NAMES` is the single source of truth for column order.
* The only outside knowledge is the list of well-known sites, which is shipped
  next to the model so both sides read the same file.

Every URL is normalised first (scheme, leading `www.`, fragment and default
port removed). The benign training data only exists in that form, so applying
it to every URL stops the model learning formatting differences between the
two classes instead of real signal.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

BASE_FEATURE_NAMES = (
    "host_length",
    "host_label_count",
    "host_digit_ratio",
    "host_entropy",
    "host_is_ip",
    "host_hyphen_count",
    "host_is_idn",
    "keyword_hits_host",
    "tld_legacy",
    "tld_country",
    "tld_other",
    "on_known_site",
    "has_at",
    "is_bare_host",
    "path_length",
    "path_depth",
    "path_digit_ratio",
    "path_hyphen_count",
    "keyword_hits_path",
    "query_length",
)

# A bare-host URL (`evil-login.shop/`) and a URL with a path are different
# populations: on a bare host the host is all there is to judge. Each host-side
# feature therefore gets a second column that is non-zero only on bare hosts,
# which lets the linear model learn a separate weight for that case. The model
# stays linear, so a feature's contribution is still exact: the sum of its two
# columns' contributions.
BARE_HOST_SUFFIX = ":bare_host"
HOST_SIDE_FEATURES = BASE_FEATURE_NAMES[: BASE_FEATURE_NAMES.index("is_bare_host")]
FEATURE_NAMES = BASE_FEATURE_NAMES + tuple(
    name + BARE_HOST_SUFFIX for name in HOST_SIDE_FEATURES
)

# Lengths and counts are heavily right-skewed (a few URLs are thousands of
# characters long), so these are stored as log(1 + x).
LOG_SCALED = frozenset({
    "host_length", "host_hyphen_count", "path_length", "path_depth",
    "path_hyphen_count", "query_length",
})

# Action words phishing URLs use to look official. Counted separately in the
# host and in the path: `secure-login.example.shop` is far more suspicious
# than `example.com/login`. Not counted at all on a well-known site, where
# `accounts.` or `/signin` is simply what a real sign-in page looks like.
SUSPICIOUS_KEYWORDS = (
    "account", "bank", "billing", "confirm", "login", "password",
    "recover", "secure", "signin", "suspend", "unlock", "update", "verify", "wallet",
)
LEGACY_TLDS = frozenset({"com", "org", "net", "edu", "gov", "mil", "int"})

_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
_SCHEME = re.compile(r"^[a-z][a-z0-9+.\-]*://")
# [0-9], not \d: in Python \d also matches non-ASCII digits, in JavaScript it does not.
_IPV4 = re.compile(r"^[0-9]{1,3}(\.[0-9]{1,3}){3}$")
_ASCII_DIGITS = frozenset("0123456789")


@dataclass(frozen=True)
class ParsedUrl:
    normalised: str
    host: str   # no userinfo, no port
    path: str   # always starts with "/"
    query: str  # without the leading "?"


def parse_url(url: str) -> ParsedUrl:
    text = _SCHEME.sub("", url.strip().translate(_ASCII_LOWER), count=1).split("#", 1)[0]
    if not text:
        raise ValueError("empty URL")

    cut = min(i for i in (text.find("/"), text.find("?"), len(text)) if i >= 0)
    authority, rest = text[:cut], text[cut:]
    userinfo, at, hostport = authority.rpartition("@")

    if hostport.startswith("["):  # IPv6 literal: the colons inside are not a port separator
        host, _, port = hostport.partition("]")
        host, port = host + "]", port.removeprefix(":")
    else:
        host, colon, port = hostport.rpartition(":")
        if not colon or not port.isascii() or not port.isdigit():
            host, port = hostport, ""
    host = host.removeprefix("www.")
    if port in ("80", "443"):
        port = ""

    if not rest.startswith("/"):
        rest = "/" + rest
    path, _, query = rest.partition("?")
    normalised = userinfo + at + host + (":" + port if port else "") + rest
    return ParsedUrl(normalised, host, path, query)


def shannon_entropy(text: str) -> float:
    """Bits per character: 0 for `aaaa`, high for random-looking strings."""
    if not text:
        return 0.0
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in Counter(text).values())


def _keyword_hits(text: str) -> int:
    return sum(keyword in text for keyword in SUSPICIOUS_KEYWORDS)


def _digit_ratio(text: str) -> float:
    return sum(ch in _ASCII_DIGITS for ch in text) / len(text) if text else 0.0


def on_known_site(host: str, known_sites: frozenset[str]) -> bool:
    """True if the host is a known site or a subdomain of one (`mail.google.com`).

    An entry `!host` in the list marks an exact host as untrusted even though its
    domain is well known (`!sites.google.com`): anyone can publish there.
    """
    if "!" + host in known_sites:
        return False
    labels = host.split(".")
    return any(".".join(labels[i:]) in known_sites for i in range(len(labels) - 1))


def extract_features(url: str, known_sites: frozenset[str] = frozenset()) -> dict[str, float]:
    parsed = parse_url(url)
    host, full = parsed.host, parsed.normalised
    # Everything after the host: host and path are measured separately so that
    # a long benign path cannot offset a suspicious host.
    tail = parsed.path + ("?" + parsed.query if parsed.query else "")
    is_ip = bool(_IPV4.match(host)) or host.startswith("[")
    labels = [label for label in host.split(".") if label]
    tld = "" if is_ip or len(labels) < 2 else labels[-1]
    tld_country = len(tld) == 2 and tld.isascii() and tld.isalpha()
    known = on_known_site(host, known_sites)

    features = {
        "host_length": len(host),
        "host_label_count": len(labels),
        "host_digit_ratio": _digit_ratio(host),
        "host_entropy": shannon_entropy(host),
        "host_is_ip": is_ip,
        "host_hyphen_count": host.count("-"),
        "host_is_idn": not host.isascii() or any(label.startswith("xn--") for label in labels),
        "keyword_hits_host": 0 if known else _keyword_hits(host),
        "tld_legacy": tld in LEGACY_TLDS,
        "tld_country": tld_country,
        "tld_other": bool(tld) and tld not in LEGACY_TLDS and not tld_country,
        "on_known_site": known,
        "has_at": "@" in full,
        "is_bare_host": tail == "/",
        "path_length": len(parsed.path),
        "path_depth": sum(1 for segment in parsed.path.split("/") if segment),
        "path_digit_ratio": _digit_ratio(tail),
        "path_hyphen_count": parsed.path.count("-"),
        "keyword_hits_path": 0 if known else _keyword_hits(tail),
        "query_length": len(parsed.query),
    }
    values = {
        name: math.log1p(features[name]) if name in LOG_SCALED else float(features[name])
        for name in BASE_FEATURE_NAMES
    }
    for name in HOST_SIDE_FEATURES:
        values[name + BARE_HOST_SUFFIX] = values[name] * values["is_bare_host"]
    return values


def featurise(urls: list[str], known_sites: frozenset[str] = frozenset()) -> np.ndarray:
    """Feature matrix of shape (len(urls), len(FEATURE_NAMES)), columns in FEATURE_NAMES order."""
    rows = [list(extract_features(url, known_sites).values()) for url in urls]
    return np.array(rows, dtype=np.float64).reshape(len(urls), len(FEATURE_NAMES))
