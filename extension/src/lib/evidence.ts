import {
  BARE_HOST_SUFFIX,
  BASE_FEATURE_NAMES,
  type BaseFeature,
  type ParsedUrl,
  isAscii,
  isIpHost,
  isOtherTld,
  keywordsIn,
  onKnownSite,
  tldOf,
} from "./featurise";
import type { EvidenceItem, Highlight } from "./messages";
import type { ModelMetadata } from "./model";

/**
 * Per-feature contribution to the model's log-odds.
 *
 * The model is linear on standardised features, so
 *   log-odds = intercept + sum_i coefficient_i * (x_i - mean_i) / scale_i
 * and each term is exactly how much that feature moved the score away from an
 * average URL. This is not an approximation of the model: it is the model.
 * A host-side feature has two columns (plain and bare-host); their terms are
 * added so the user sees one line per feature.
 */
export function contributions(
  features: ArrayLike<number>,
  metadata: ModelMetadata,
): Map<BaseFeature, number> {
  const totals = new Map<BaseFeature, number>(BASE_FEATURE_NAMES.map((name) => [name, 0]));
  metadata.feature_names.forEach((name, i) => {
    const base = name.replace(BARE_HOST_SUFFIX, "") as BaseFeature;
    const standardised = ((features[i] ?? 0) - metadata.scaler_mean[i]!) / metadata.scaler_scale[i]!;
    totals.set(base, (totals.get(base) ?? 0) + metadata.coefficients[i]! * standardised);
  });
  return totals;
}

/** Log-odds rebuilt from the contributions; equals the ONNX model's output. */
export function logOdds(features: ArrayLike<number>, metadata: ModelMetadata): number {
  let total = metadata.intercept;
  for (const value of contributions(features, metadata).values()) total += value;
  return total;
}

// Two phrasings per feature: a contribution can come from a value being
// unusually high or unusually low, and the sentence must say which.
const EXPLANATIONS: Record<BaseFeature, { high: string; low: string }> = {
  host_length: {
    high: "The site name is longer than average",
    low: "The site name is shorter than average",
  },
  host_label_count: {
    high: "The site name has many parts separated by dots",
    low: "The site name has few parts",
  },
  host_digit_ratio: {
    high: "The site name contains a lot of digits",
    low: "The site name contains few or no digits",
  },
  host_entropy: {
    high: "The site name looks random rather than like a word",
    low: "The site name uses few distinct characters",
  },
  host_is_ip: {
    high: "The address uses a raw IP number instead of a site name",
    low: "The address uses a site name, not a raw IP number",
  },
  host_hyphen_count: {
    high: "The site name contains hyphens, often used to imitate a brand",
    low: "The site name has no hyphens",
  },
  host_is_idn: {
    high: "The site name uses non-English characters that can imitate other letters",
    low: "The site name uses only plain letters",
  },
  keyword_hits_host: {
    high: "The site name contains official-sounding words on a site that is not well known",
    low: "The site name contains no official-sounding words",
  },
  tld_legacy: {
    high: "The address ends in a long-established ending such as .com or .org",
    low: "The address does not end in .com, .org or similar",
  },
  tld_country: {
    high: "The address ends in a country code",
    low: "The address does not end in a country code",
  },
  tld_other: {
    high: "The address has an uncommon ending that is cheap to register",
    low: "The address has a common ending",
  },
  on_known_site: {
    high: "The site is on a widely visited domain",
    low: "The site is not on a widely visited domain",
  },
  has_at: {
    high: "The address contains an @ sign, which can hide the real site name",
    low: "The address contains no @ sign",
  },
  is_bare_host: {
    high: "The address is just a site name with no page path",
    low: "The address points to a specific page",
  },
  path_length: { high: "The page path is long", low: "The page path is short" },
  path_depth: {
    high: "The page is several folders deep",
    low: "The page is at or near the top of the site",
  },
  path_digit_ratio: {
    high: "The page path contains a lot of digits",
    low: "The page path contains few digits",
  },
  path_hyphen_count: {
    high: "The page path contains hyphens, as article-style pages do",
    low: "The page path has few or no hyphens",
  },
  keyword_hits_path: {
    high: "The page path contains official-sounding words on a site that is not well known",
    low: "The page path contains no official-sounding words",
  },
  query_length: {
    high: "The address carries a long string of parameters",
    low: "The address carries few or no parameters",
  },
};

/** Evidence for every feature, the ones raising risk most first. */
export function buildEvidence(features: ArrayLike<number>, metadata: ModelMetadata): EvidenceItem[] {
  const index = new Map(metadata.feature_names.map((name, i) => [name, i]));
  return [...contributions(features, metadata)]
    .map(([feature, contribution]) => {
      const i = index.get(feature)!;
      const aboveAverage = (features[i] ?? 0) > metadata.scaler_mean[i]!;
      return {
        feature,
        contribution,
        explanation: EXPLANATIONS[feature][aboveAverage ? "high" : "low"],
      };
    })
    .sort((a, b) => b.contribution - a.contribution);
}

/** Parts of the normalised URL worth pointing at, in order, never overlapping. */
export function findHighlights(parsed: ParsedUrl, knownSites: ReadonlySet<string>): Highlight[] {
  const text = parsed.normalised;
  // The host starts after the last "@" of the authority (the part before the first "/").
  const at = text.lastIndexOf("@", text.indexOf("/"));
  const hostStart = at + 1;
  const hostEnd = hostStart + parsed.host.length;
  const found: Highlight[] = [];

  if (isIpHost(parsed.host)) {
    found.push({ start: hostStart, end: hostEnd, reason: "A raw IP number instead of a site name" });
  }
  if (at >= 0) {
    found.push({ start: 0, end: at + 1, reason: "Everything before the @ is ignored by the browser" });
  }
  let offset = hostStart;
  for (const label of parsed.host.split(".")) {
    if (label.startsWith("xn--") || !isAscii(label)) {
      found.push({
        start: offset,
        end: offset + label.length,
        reason: "Non-English characters that can imitate other letters",
      });
    }
    offset += label.length + 1;
  }
  const tld = tldOf(parsed.host);
  if (isOtherTld(tld)) {
    found.push({ start: hostEnd - tld.length, end: hostEnd, reason: "An uncommon address ending" });
  }
  if (!onKnownSite(parsed.host, knownSites)) {
    for (const keyword of keywordsIn(text)) {
      for (let i = text.indexOf(keyword); i >= 0; i = text.indexOf(keyword, i + 1)) {
        found.push({
          start: i,
          end: i + keyword.length,
          // One shared reason, so many marked words still read as a single point.
          reason: "Official-sounding words on a site that is not well known",
        });
      }
    }
  }

  found.sort((a, b) => a.start - b.start || b.end - a.end);
  const result: Highlight[] = [];
  for (const highlight of found) {
    const last = result[result.length - 1];
    if (!last || highlight.start >= last.end) result.push(highlight);
  }
  return result;
}
