import { describe, expect, it } from "vitest";
import { analyse, analyseAddress } from "../src/lib/analyse";
import { buildEvidence, contributions, findHighlights, logOdds } from "../src/lib/evidence";
import { featurise, parseUrl } from "../src/lib/featurise";
import { assertCompatible } from "../src/lib/model";
import { confidenceFor, riskScore, verdictFor } from "../src/lib/verdict";
import { knownSites, metadata, models, parityFixture, sigmoid } from "./helpers";

describe("evidence", () => {
  it.each(parityFixture)("contributions add up to the model's score for $url", ({ url, probability }) => {
    const features = featurise(url, knownSites);
    expect(sigmoid(logOdds(features, metadata))).toBeCloseTo(probability, 5);
  });

  it("merges each bare-host column into its base feature", () => {
    const totals = contributions(featurise("https://secure-login.example.shop/", knownSites), metadata);
    expect([...totals.keys()].some((name) => name.includes(":"))).toBe(false);
    expect(totals.size).toBe(20);
  });

  it("ranks the factors that raise risk first, in plain language", () => {
    const evidence = buildEvidence(featurise("http://secure-login.verify-account.shop/", knownSites), metadata);
    const sorted = [...evidence].sort((a, b) => b.contribution - a.contribution);
    expect(evidence).toEqual(sorted);
    expect(evidence[0]!.contribution).toBeGreaterThan(0);
    expect(evidence.every((item) => item.explanation.length > 10)).toBe(true);
    expect(evidence.some((item) => /official-sounding/.test(item.explanation))).toBe(true);
  });
});

describe("highlights", () => {
  const texts = (url: string) => {
    const parsed = parseUrl(url);
    return findHighlights(parsed, knownSites).map((h) => parsed.normalised.slice(h.start, h.end));
  };

  it("marks keywords and an uncommon ending on an unknown site", () => {
    expect(texts("https://secure-login.example.shop/verify")).toEqual(["secure", "login", "shop", "verify"]);
  });

  it("marks nothing on a well-known sign-in page", () => {
    expect(texts("https://accounts.google.com/signin")).toEqual([]);
  });

  it("marks a raw IP host, the part before an @, and punycode", () => {
    expect(texts("http://192.168.0.1/a")).toEqual(["192.168.0.1"]);
    expect(texts("http://paypal.com@evil.com/")).toEqual(["paypal.com@"]);
    // The host is found after the "@", even when the same text appears before it.
    expect(texts("http://evil.top@evil.top/")).toEqual(["evil.top@", "top"]);
    expect(texts("http://xn--pypal-4ve.com/")).toEqual(["xn--pypal-4ve"]);
  });

  it("never overlaps and stays inside the text", () => {
    const parsed = parseUrl("http://account-login.bank-secure.top/login/account?login=1");
    const highlights = findHighlights(parsed, knownSites);
    highlights.reduce((end, h) => {
      expect(h.start).toBeGreaterThanOrEqual(end);
      expect(h.end).toBeLessThanOrEqual(parsed.normalised.length);
      return h.end;
    }, 0);
  });
});

describe("verdict", () => {
  it("has three buckets around the threshold", () => {
    expect(verdictFor(0.9, 0.36)).toBe("phishing");
    expect(verdictFor(0.36, 0.36)).toBe("suspicious"); // exactly at the threshold is not flagged
    expect(verdictFor(0.2, 0.36)).toBe("suspicious");
    expect(verdictFor(0.18, 0.36)).toBe("clear");
  });

  it("is least confident near the threshold", () => {
    expect(confidenceFor(0.36, 0.36)).toBe("low");
    expect(confidenceFor(0.4, 0.36)).toBe("low");
    expect(confidenceFor(0.6, 0.36)).toBe("medium");
    expect(confidenceFor(0.99, 0.36)).toBe("high");
    expect(confidenceFor(0.01, 0.36)).toBe("high");
    expect(confidenceFor(0, 0.36)).toBe("high"); // no division by zero at the extremes
    expect(confidenceFor(1, 0.36)).toBe("high");
  });

  it("scores 0 to 100", () => {
    expect([riskScore(0), riskScore(0.456), riskScore(0.9996), riskScore(1)]).toEqual([0, 46, 99, 100]);
  });
});

describe("analyseAddress", () => {
  it("flags an obvious imitation and clears a well-known sign-in page", async () => {
    const fake = await analyseAddress("http://secure-login.account-verify.paypa1-update.top/", models.url);
    const real = await analyseAddress("https://accounts.google.com/signin", models.url);
    expect(fake.verdict).toBe("phishing");
    expect(real.verdict).toBe("clear");
    expect(fake.score).toBeGreaterThan(real.score);
    expect(fake.modelVersion).toBe(metadata.model_version);
    expect(fake.displayText).toBe("secure-login.account-verify.paypa1-update.top/");
  });

  it("rejects an empty address", async () => {
    await expect(analyseAddress("https://", models.url)).rejects.toThrow("empty URL");
  });

  it("an address-only request has no message part", async () => {
    const result = await analyse({ type: "ANALYSE", url: "https://example.org/a" }, models);
    expect(result.message).toBeUndefined();
    expect(result.verdict).toBe(result.address!.verdict);
    expect(result.score).toBe(result.address!.score);
  });
});

describe("model compatibility", () => {
  it("accepts the published model", () => {
    expect(() => assertCompatible(metadata)).not.toThrow();
  });

  it("refuses a model with a different feature order", () => {
    const shuffled = { ...metadata, feature_names: [...metadata.feature_names].reverse() };
    expect(() => assertCompatible(shuffled)).toThrow("feature list");
  });

  it("refuses a model without a threshold", () => {
    expect(() => assertCompatible({ ...metadata, threshold: Number.NaN })).toThrow("threshold");
  });
});
