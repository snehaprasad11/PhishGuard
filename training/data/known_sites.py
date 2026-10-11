"""The list of well-known sites the model uses as a reputation signal.

A model that sees only URL text cannot tell `accounts.google.com/signin` from
an imitation of it: both contain "account" and "signin". The list gives it one
extra fact — whether the host belongs to a widely visited domain.

File format (`models/known-sites.txt`), one entry per line:

    google.com            a well-known domain; its subdomains count too
    !sites.google.com     an exact host that is NOT trusted despite its domain

It is derived deterministically from the pinned data and shipped next to the
model, because the browser needs the identical list to compute the identical
feature.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from training.data.domains import is_public_suffix
from training.data.loaders import load_phishing, load_tranco
from training.features.url_features import on_known_site, parse_url


def select_known_sites(tranco: pd.DataFrame, top_n: int) -> list[str]:
    """Top-ranked domains, minus shared-hosting platforms.

    `weebly.com` or `web.app` rank highly, but anyone can publish a page under
    them, so the platform's popularity says nothing about an individual page.
    Such domains are public suffixes in the Public Suffix List, which is how
    they are recognised here.
    """
    top = tranco.loc[tranco["rank"] <= top_n, "domain"]
    return sorted(domain for domain in top if not is_public_suffix(domain))


def find_open_hosts(
    phishing: pd.DataFrame, known_sites: frozenset[str],
    start: pd.Timestamp, end: pd.Timestamp, min_phishing: int,
) -> list[str]:
    """Hosts on well-known domains that were used for phishing at least `min_phishing` times.

    `sites.google.com` or a link shortener sits on a trusted domain but lets
    anyone publish, so trusting it would wave phishing through. Only phishing
    submitted in [start, end) is counted — the training period — so the list
    carries no information from the validation or test periods.
    """
    in_period = phishing[(phishing["timestamp"] >= start) & (phishing["timestamp"] < end)]
    hosts = []
    for url in in_period["url"]:
        try:
            hosts.append(parse_url(url).host)
        except ValueError:
            continue
    counts = pd.Series(hosts).value_counts()
    return sorted(
        host for host, n in counts.items() if n >= min_phishing and on_known_site(host, known_sites)
    )


def build_known_sites(cfg: dict[str, Any]) -> frozenset[str]:
    """Write the list to its published path and return it."""
    tranco_cfg = cfg["data"]["tranco"]
    sites = select_known_sites(load_tranco(cfg), tranco_cfg["known_sites_top_n"])
    open_hosts = find_open_hosts(
        load_phishing(cfg), frozenset(sites),
        start=pd.Timestamp(cfg["split"]["phishing_min_date"], tz="UTC"),
        end=pd.Timestamp(cfg["split"]["val_start"], tz="UTC"),
        min_phishing=tranco_cfg["open_host_min_phishing"],
    )
    entries = sites + [f"!{host}" for host in open_hosts]
    path = Path(cfg["paths"]["known_sites"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(entries) + "\n", "utf-8", newline="\n")
    return frozenset(entries)


def load_known_sites(cfg: dict[str, Any]) -> frozenset[str]:
    return frozenset(Path(cfg["paths"]["known_sites"]).read_text("utf-8").split())
