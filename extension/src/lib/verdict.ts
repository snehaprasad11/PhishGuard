import type { Confidence, Verdict } from "./messages";

/**
 * Three outcomes, deliberately none of them called "safe": a URL model can
 * find warning signs, but their absence is not proof of safety.
 *
 * - phishing:   probability above the model's decision threshold (set for a 1% false-positive
 *               rate on validation data).
 * - suspicious: above half the threshold — worth a second look, not an alarm.
 * - clear:      no warning signs found.
 */
export function verdictFor(probability: number, threshold: number): Verdict {
  if (probability > threshold) return "phishing";
  if (probability > threshold / 2) return "suspicious";
  return "clear";
}

/**
 * How firmly the model holds its verdict, from how far the calibrated
 * probability sits from the decision threshold. Near the threshold a small
 * change in the URL would flip the verdict, so confidence is low there.
 */
export function confidenceFor(probability: number, threshold: number): Confidence {
  const odds = (p: number) => Math.log(p / (1 - p));
  const clamped = Math.min(Math.max(probability, 1e-6), 1 - 1e-6);
  const distance = Math.abs(odds(clamped) - odds(threshold));
  if (distance >= 2) return "high";
  if (distance >= 0.75) return "medium";
  return "low";
}

/** 0..100, but 100 only for a probability of exactly 1: a rounded-up 100 would claim certainty. */
export const riskScore = (probability: number): number =>
  probability >= 1 ? 100 : Math.min(Math.round(probability * 100), 99);

export const VERDICT_COPY: Record<Verdict, { title: string; summary: string; steps: string[] }> = {
  phishing: {
    title: "Likely phishing",
    summary: "This has several signs commonly seen in phishing.",
    steps: [
      "Do not enter a password, card number or one-time code, and do not reply with personal details.",
      "Close the page. If it claims to be from a company you use, open that company's site by typing its address yourself.",
      "If you already entered details, change that password now and contact your bank if payment details were involved.",
    ],
  },
  suspicious: {
    title: "Be careful",
    summary: "This has some signs seen in phishing, but not enough to be sure.",
    steps: [
      "Check the site name letter by letter before entering anything.",
      "If you arrived from a message or email, go to the organisation's site directly instead of using the link.",
    ],
  },
  clear: {
    title: "No warning signs found",
    summary: "Nothing here looks like typical phishing.",
    steps: [
      "This is not a guarantee: PhishGuard reads the address and the wording you paste, not the page itself.",
      "Still check the site name before entering a password or payment details.",
    ],
  },
};
