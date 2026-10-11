import tldextract

# suffix_list_urls=() pins tldextract to the Public Suffix List snapshot shipped
# inside the package: no network call, and the same answer on every machine.
_icann = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)
_with_private = tldextract.TLDExtract(
    suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True
)


def is_public_suffix(domain: str) -> bool:
    """True for a suffix other people register sites under: `co.uk`, `web.app`, `weebly.com`."""
    return _with_private(domain).suffix == domain


def registered_domain(url_or_host: str, private: bool = False) -> str | None:
    """`https://login.pay.example.co.uk/x` -> `example.co.uk`; an IP host is returned as-is.

    With `private=True`, shared-hosting suffixes count as public suffixes, so
    `shop1.web.app` and `shop2.web.app` are two different sites rather than
    both being `web.app`.
    """
    parts = (_with_private if private else _icann)(url_or_host)
    return (parts.top_domain_under_public_suffix or parts.domain).lower() or None
