// TypeScript mirror of training/features/url_features.py.
//
// The two files must agree exactly: the model was trained on Python's numbers
// and scores the numbers computed here. tests/featurise.test.ts checks this
// file against values exported by the training pipeline, so change both
// together and re-export the fixture.
//
// Rules shared with the Python side: only ASCII A-Z is lowercased, and every
// length counts Unicode code points (not UTF-16 units, hence the spreads).

export const BASE_FEATURE_NAMES = [
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
] as const;

export type BaseFeature = (typeof BASE_FEATURE_NAMES)[number];

export const BARE_HOST_SUFFIX = ":bare_host";
const HOST_SIDE_FEATURES = BASE_FEATURE_NAMES.slice(0, BASE_FEATURE_NAMES.indexOf("is_bare_host"));
export const FEATURE_NAMES: readonly string[] = [
  ...BASE_FEATURE_NAMES,
  ...HOST_SIDE_FEATURES.map((name) => name + BARE_HOST_SUFFIX),
];

const LOG_SCALED = new Set<string>([
  "host_length",
  "host_hyphen_count",
  "path_length",
  "path_depth",
  "path_hyphen_count",
  "query_length",
]);

export const SUSPICIOUS_KEYWORDS = [
  "account", "bank", "billing", "confirm", "login", "password",
  "recover", "secure", "signin", "suspend", "unlock", "update", "verify", "wallet",
] as const;

const LEGACY_TLDS = new Set(["com", "org", "net", "edu", "gov", "mil", "int"]);
const SCHEME = /^[a-z][a-z0-9+.-]*:\/\//;
const IPV4 = /^[0-9]{1,3}(\.[0-9]{1,3}){3}$/;

export interface ParsedUrl {
  normalised: string;
  host: string; // no userinfo, no port
  path: string; // always starts with "/"
  query: string; // without the leading "?"
}

const codePoints = (text: string): number => [...text].length;

export function isAscii(text: string): boolean {
  for (let i = 0; i < text.length; i++) if (text.charCodeAt(i) > 127) return false;
  return true;
}

function count(text: string, needle: string): number {
  return text.split(needle).length - 1;
}

export function parseUrl(url: string): ParsedUrl {
  const lowered = url.trim().replace(/[A-Z]/g, (ch) => ch.toLowerCase());
  const text = lowered.replace(SCHEME, "").split("#", 1)[0] ?? "";
  if (!text) throw new Error("empty URL");

  const cut = Math.min(...[text.indexOf("/"), text.indexOf("?"), text.length].filter((i) => i >= 0));
  const authority = text.slice(0, cut);
  let rest = text.slice(cut);

  const at = authority.lastIndexOf("@");
  const userinfo = at >= 0 ? authority.slice(0, at + 1) : ""; // includes the "@"
  const hostport = authority.slice(at + 1);

  let host: string;
  let port: string;
  if (hostport.startsWith("[")) {
    // IPv6 literal: the colons inside are not a port separator.
    const close = hostport.indexOf("]");
    host = close >= 0 ? hostport.slice(0, close) + "]" : hostport + "]";
    port = close >= 0 ? hostport.slice(close + 1).replace(/^:/, "") : "";
  } else {
    const colon = hostport.lastIndexOf(":");
    const candidate = colon >= 0 ? hostport.slice(colon + 1) : "";
    if (colon >= 0 && /^[0-9]+$/.test(candidate)) {
      host = hostport.slice(0, colon);
      port = candidate;
    } else {
      host = hostport;
      port = "";
    }
  }
  if (host.startsWith("www.")) host = host.slice(4);
  if (port === "80" || port === "443") port = "";

  if (!rest.startsWith("/")) rest = "/" + rest;
  const q = rest.indexOf("?");
  const path = q >= 0 ? rest.slice(0, q) : rest;
  const query = q >= 0 ? rest.slice(q + 1) : "";
  return { normalised: userinfo + host + (port ? ":" + port : "") + rest, host, path, query };
}

export function shannonEntropy(text: string): number {
  const chars = [...text];
  if (chars.length === 0) return 0;
  const counts = new Map<string, number>();
  for (const ch of chars) counts.set(ch, (counts.get(ch) ?? 0) + 1);
  let entropy = 0;
  for (const c of counts.values()) entropy -= (c / chars.length) * Math.log2(c / chars.length);
  return entropy;
}

function digitRatio(text: string): number {
  const chars = [...text];
  if (chars.length === 0) return 0;
  return chars.filter((ch) => ch >= "0" && ch <= "9").length / chars.length;
}

export function keywordsIn(text: string): string[] {
  return SUSPICIOUS_KEYWORDS.filter((keyword) => text.includes(keyword));
}

/**
 * True if the host is a known site or a subdomain of one (`mail.google.com`).
 * An entry `!host` marks an exact host as untrusted even though its domain is
 * well known (`!sites.google.com`): anyone can publish there.
 */
export function onKnownSite(host: string, knownSites: ReadonlySet<string>): boolean {
  if (knownSites.has("!" + host)) return false;
  const labels = host.split(".");
  for (let i = 0; i < labels.length - 1; i++) {
    if (knownSites.has(labels.slice(i).join("."))) return true;
  }
  return false;
}

export function isIpHost(host: string): boolean {
  return IPV4.test(host) || host.startsWith("[");
}

export function tldOf(host: string): string {
  const labels = host.split(".").filter(Boolean);
  return isIpHost(host) || labels.length < 2 ? "" : (labels[labels.length - 1] ?? "");
}

export function isCountryTld(tld: string): boolean {
  return /^[a-z]{2}$/.test(tld);
}

export function isOtherTld(tld: string): boolean {
  return tld !== "" && !LEGACY_TLDS.has(tld) && !isCountryTld(tld);
}

export function extractFeatures(
  url: string,
  knownSites: ReadonlySet<string> = new Set(),
): Record<string, number> {
  const parsed = parseUrl(url);
  const { host, path, query, normalised } = parsed;
  // Everything after the host: host and path are measured separately so that
  // a long benign path cannot offset a suspicious host.
  const tail = path + (query ? "?" + query : "");
  const labels = host.split(".").filter(Boolean);
  const tld = tldOf(host);
  const known = onKnownSite(host, knownSites);

  const raw: Record<BaseFeature, number | boolean> = {
    host_length: codePoints(host),
    host_label_count: labels.length,
    host_digit_ratio: digitRatio(host),
    host_entropy: shannonEntropy(host),
    host_is_ip: isIpHost(host),
    host_hyphen_count: count(host, "-"),
    host_is_idn: !isAscii(host) || labels.some((label) => label.startsWith("xn--")),
    keyword_hits_host: known ? 0 : keywordsIn(host).length,
    tld_legacy: LEGACY_TLDS.has(tld),
    tld_country: isCountryTld(tld),
    tld_other: isOtherTld(tld),
    on_known_site: known,
    has_at: normalised.includes("@"),
    is_bare_host: tail === "/",
    path_length: codePoints(path),
    path_depth: path.split("/").filter(Boolean).length,
    path_digit_ratio: digitRatio(tail),
    path_hyphen_count: count(path, "-"),
    keyword_hits_path: known ? 0 : keywordsIn(tail).length,
    query_length: codePoints(query),
  };

  const values: Record<string, number> = {};
  for (const name of BASE_FEATURE_NAMES) {
    const value = Number(raw[name]);
    values[name] = LOG_SCALED.has(name) ? Math.log1p(value) : value;
  }
  for (const name of HOST_SIDE_FEATURES) {
    values[name + BARE_HOST_SUFFIX] = (values[name] ?? 0) * (values["is_bare_host"] ?? 0);
  }
  return values;
}

/** Feature vector in the order the model was trained on. */
export function featurise(url: string, knownSites: ReadonlySet<string>): Float32Array {
  const values = extractFeatures(url, knownSites);
  return Float32Array.from(FEATURE_NAMES, (name) => values[name] ?? 0);
}
