import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  FEATURE_NAMES,
  SUSPICIOUS_KEYWORDS,
  extractFeatures,
  featurise,
  onKnownSite,
  parseUrl,
} from "../src/lib/featurise";

const root = resolve(import.meta.dirname, "../..");
const fixture: { url: string; features: number[]; probability: number }[] = JSON.parse(
  readFileSync(resolve(import.meta.dirname, "fixtures/url-parity.json"), "utf-8"),
);
const metadata = JSON.parse(readFileSync(resolve(root, "models/url-model.json"), "utf-8"));
const knownSites = new Set(
  readFileSync(resolve(root, "models/known-sites.txt"), "utf-8").split(/\s+/).filter(Boolean),
);

describe("parity with the Python featuriser", () => {
  it("uses the same feature order and keyword list as the exported model", () => {
    expect([...FEATURE_NAMES]).toEqual(metadata.feature_names);
    expect([...SUSPICIOUS_KEYWORDS]).toEqual(metadata.suspicious_keywords);
  });

  it("covers a meaningful number of URLs", () => {
    expect(fixture.length).toBeGreaterThanOrEqual(30);
  });

  it.each(fixture)("matches Python for $url", ({ url, features }) => {
    const values = extractFeatures(url, knownSites);
    FEATURE_NAMES.forEach((name, i) => {
      expect(values[name], name).toBeCloseTo(features[i]!, 12);
    });
  });
});

describe("parseUrl", () => {
  it.each([
    ["https://example.com", "example.com/", "example.com", "/", ""],
    ["https://example.com?x=1", "example.com/?x=1", "example.com", "/", "x=1"],
    ["http://example.com:8080/a", "example.com:8080/a", "example.com", "/a", ""],
    ["http://user@www.example.com/a", "user@example.com/a", "example.com", "/a", ""],
    ["http://[2001:db8::1]:8080/a", "[2001:db8::1]:8080/a", "[2001:db8::1]", "/a", ""],
    ["HTTPS://WWW.EXAMPLE.COM:443/a#frag", "example.com/a", "example.com", "/a", ""],
  ])("%s", (url, normalised, host, path, query) => {
    expect(parseUrl(url)).toEqual({ normalised, host, path, query });
  });

  it("rejects an empty URL", () => {
    for (const empty of ["", "   ", "https://", "#fragment"]) {
      expect(() => parseUrl(empty)).toThrow("empty URL");
    }
  });
});

describe("onKnownSite", () => {
  const known = new Set(["google.com", "!sites.google.com"]);
  it.each([
    ["accounts.google.com", true],
    ["google.com", true],
    ["sites.google.com", false],
    ["google.com.evil.shop", false],
    ["notgoogle.com", false],
    ["com", false],
  ])("%s -> %s", (host, expected) => {
    expect(onKnownSite(host, known)).toBe(expected);
  });
});

describe("featurise", () => {
  it("returns one float per model feature", () => {
    const vector = featurise("https://example.com/a?x=1", knownSites);
    expect(vector).toBeInstanceOf(Float32Array);
    expect(vector.length).toBe(metadata.feature_names.length);
    expect([...vector].every(Number.isFinite)).toBe(true);
  });
});
