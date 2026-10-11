import { describe, expect, it } from "vitest";
import { analyse } from "../src/lib/analyse";
import { decidingPart, worstVerdict } from "../src/lib/messages";
import { buildTextEvidence, findTextHighlights, textLogOdds } from "../src/lib/text-evidence";
import {
  assertTextCompatible,
  clip,
  extractTextFeatures,
  hasWording,
  lexiconMatches,
  tokenise,
} from "../src/lib/text-features";
import { models, sigmoid, textFixture, textMetadata } from "./helpers";

describe("parity with the Python text featuriser", () => {
  it("covers both kinds of message", () => {
    expect(textFixture.filter((row) => row.label === "legit").length).toBeGreaterThanOrEqual(20);
    expect(textFixture.filter((row) => row.label === "scam").length).toBeGreaterThanOrEqual(10);
  });

  it.each(textFixture)("matches Python for: $text", ({ text, features, probability }) => {
    const extracted = extractTextFeatures(text, textMetadata);
    textMetadata.feature_names.forEach((name, i) => {
      expect(extracted.values[name], name).toBeCloseTo(features[i]!, 12);
    });
    expect(sigmoid(textLogOdds(extracted.vector, textMetadata))).toBeCloseTo(probability, 5);
  });
});

describe("tokenise", () => {
  const texts = (text: string) => tokenise(text).map((token) => token.text);

  it("produces lowercase words and placeholders, like Python", () => {
    expect(texts("URGENT: pay £2.99 NOW at http://evil.top/pay?id=1 or email Help@Bank.com, ref 4821. I'm ok")).toEqual([
      "urgent", "pay", "_cur_", "_num_", "now", "at", "_url_", "or", "email", "_email_", "ref", "_num_", "i'm", "ok",
    ]);
    expect(texts("a I x héllo wörld naïve")).toEqual(["llo", "rld", "na", "ve"]);
  });

  it("records where each token sits in the original text", () => {
    const text = "Verify NOW at www.evil.top";
    for (const token of tokenise(text)) {
      if (!token.text.startsWith("_")) expect(text.slice(token.start, token.end).toLowerCase()).toBe(token.text);
    }
    const url = tokenise(text).find((token) => token.text === "_url_")!;
    expect(text.slice(url.start, url.end)).toBe("www.evil.top");
  });

  it("clips by code points, not UTF-16 units", () => {
    expect(clip("😀😀😀", 2)).toBe("😀😀");
    expect(clip("abc", 5)).toBe("abc");
  });
});

describe("lexicon matches", () => {
  it("prefer the longest phrase and report character spans", () => {
    const text = "This is your final notice. Respond within 24 hours.";
    const found = lexiconMatches(tokenise(text), textMetadata.lexicon)
      .filter((match) => match.category === "urgency")
      .map((match) => text.slice(match.start, match.end));
    expect(found).toEqual(["final notice", "within 24 hours"]);
  });
});

describe("text evidence and highlights", () => {
  const scam = "URGENT: your account has been suspended. Verify your identity now or it will be closed.";

  it("contributions add up to the model's score", () => {
    const { vector } = extractTextFeatures(scam, textMetadata);
    const evidence = buildTextEvidence(vector, textMetadata);
    const total = evidence.reduce((sum, item) => sum + item.contribution, textMetadata.intercept);
    expect(total).toBeCloseTo(textLogOdds(vector, textMetadata), 10);
    expect(evidence.length).toBe(textMetadata.feature_names.length);
    expect(evidence[0]!.contribution).toBeGreaterThan(0);
  });

  it("highlights the manipulative phrases, in order and without overlap", () => {
    const { matches, clipped } = extractTextFeatures(scam, textMetadata);
    const highlights = findTextHighlights(matches);
    const marked = highlights.map((h) => clipped.slice(h.start, h.end).toLowerCase());
    expect(marked).toEqual(expect.arrayContaining(["urgent", "suspended", "verify your"]));
    highlights.reduce((end, h) => {
      expect(h.start).toBeGreaterThanOrEqual(end);
      return h.end;
    }, 0);
  });

  it("does not highlight first-person words: they are not manipulation", () => {
    const { matches } = extractTextFeatures("We will call you about my order", textMetadata);
    expect(matches.some((match) => match.category === "first_person")).toBe(true);
    expect(findTextHighlights(matches)).toEqual([]);
  });
});

describe("hasWording", () => {
  it.each([
    ["https://example.com/login", false],
    ["  www.example.com  ", false],
    ["see https://example.com", false],
    ["Your parcel is on hold: https://example.com", true],
    ["call me at 5pm about the meeting", true],
    ["", false],
  ])("%j -> %s", (text, expected) => expect(hasWording(text, 5000)).toBe(expected));
});

describe("analyse", () => {
  it("judges a pasted message twice: its link and its wording", async () => {
    const result = await analyse(
      {
        type: "ANALYSE",
        url: "http://secure-login.account-verify.top/confirm",
        text: "URGENT: your account has been suspended. Verify your identity now at http://secure-login.account-verify.top/confirm or it will be closed.",
      },
      models,
    );
    expect(result.address!.verdict).toBe("phishing");
    expect(result.message!.verdict).toBe("phishing");
    expect(result.verdict).toBe("phishing");
    expect(result.score).toBe(Math.max(result.address!.score, result.message!.score));
    expect(result.message!.modelVersion).toBe(textMetadata.model_version);
  });

  it("warns about a scam wording even when the link is to a well-known site", async () => {
    const result = await analyse(
      {
        type: "ANALYSE",
        url: "https://accounts.google.com/signin",
        text: "Dear customer, your account will be suspended. Verify your password immediately by clicking here: https://accounts.google.com/signin",
      },
      models,
    );
    expect(result.address!.verdict).toBe("clear");
    expect(result.message!.verdict).not.toBe("clear");
    expect(result.verdict).toBe(result.message!.verdict);
    expect(decidingPart(result)).toBe(result.message);
  });

  it("does not flag an ordinary message", async () => {
    const result = await analyse(
      { type: "ANALYSE", text: "Hi, your parcel will be delivered today between 2pm and 4pm. Track it in the app." },
      models,
    );
    expect(result.address).toBeUndefined();
    expect(result.verdict).toBe("clear");
  });

  it("shows only the start of a very long message, with highlights inside it", async () => {
    const long = "Please verify your account now. " + "Nothing else to say here. ".repeat(100);
    const { message } = await analyse({ type: "ANALYSE", text: long }, models);
    expect(message!.displayText.length).toBeLessThanOrEqual(601);
    expect(message!.displayText.endsWith("…")).toBe(true);
    expect(message!.highlights.every((h) => h.end <= 600)).toBe(true);
  });
});

describe("combining verdicts", () => {
  it("takes the most serious", () => {
    expect(worstVerdict(["clear", "suspicious"])).toBe("suspicious");
    expect(worstVerdict(["phishing", "clear"])).toBe("phishing");
    expect(worstVerdict(["clear"])).toBe("clear");
  });
});

describe("text model compatibility", () => {
  it("accepts the published model", () => {
    expect(() => assertTextCompatible(textMetadata)).not.toThrow();
  });

  it("refuses a feature this build cannot compute", () => {
    const extra = { ...textMetadata, feature_names: [...textMetadata.feature_names, "sentiment"] };
    expect(() => assertTextCompatible(extra)).toThrow("sentiment");
  });

  it("refuses inconsistent metadata or a missing threshold", () => {
    expect(() => assertTextCompatible({ ...textMetadata, coefficients: [1] })).toThrow("inconsistent");
    expect(() => assertTextCompatible({ ...textMetadata, threshold: 0 })).toThrow("threshold");
  });
});
