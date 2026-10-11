import type { EvidenceItem, Highlight } from "./messages";
import type { LexiconMatch, TextModelMetadata } from "./text-features";

// As for the URL model, each term below is exactly how far one signal moved
// the log-odds away from an average message:
//   coefficient * (value - mean) / scale
// and the terms plus the intercept sum to the model's output.

const EXPLANATIONS: Record<string, { high: string; low: string }> = {
  urgency: { high: "Pushes you to act quickly", low: "Puts no pressure on you to act quickly" },
  threat: {
    high: "Warns that something bad has happened or will happen to your account",
    low: "Makes no threats",
  },
  credential_request: {
    high: "Asks you to confirm or enter personal or security details",
    low: "Does not ask for personal or security details",
  },
  call_to_action: {
    high: "Tells you to click a link or contact a number",
    low: "Does not push you towards a link or a number",
  },
  reward: { high: "Promises a prize, reward or windfall", low: "Offers no prize or reward" },
  generic_greeting: {
    high: "Greets you generically instead of by name",
    low: "Has no generic greeting",
  },
  first_person: {
    high: "Is written in a personal voice (we, my, I'm)",
    low: "Is not written in a personal voice",
  },
  signal_kinds: {
    high: "Combines several manipulation tactics at once",
    low: "Uses few or no manipulation tactics",
  },
  number_count: { high: "Contains several numbers", low: "Contains few or no numbers" },
  currency_count: { high: "Mentions amounts of money", low: "Mentions no amounts of money" },
  token_count: { high: "Is longer than a typical message", low: "Is shorter than a typical message" },
  uppercase_ratio: { high: "Uses a lot of capital letters", low: "Uses few capital letters" },
  exclamation_count: { high: "Uses exclamation marks", low: "Uses no exclamation marks" },
};

const HIGHLIGHT_REASONS: Record<string, string> = {
  urgency: "Pressure to act quickly",
  threat: "A threat about your account",
  credential_request: "A request for personal or security details",
  call_to_action: "An instruction to click a link or contact a number",
  reward: "A promise of a prize or reward",
  generic_greeting: "A generic greeting",
};

export function textContributions(vector: ArrayLike<number>, metadata: TextModelMetadata): number[] {
  return metadata.feature_names.map(
    (_, i) => (metadata.coefficients[i]! * ((vector[i] ?? 0) - metadata.scaler_mean[i]!)) / metadata.scaler_scale[i]!,
  );
}

export function textLogOdds(vector: ArrayLike<number>, metadata: TextModelMetadata): number {
  return textContributions(vector, metadata).reduce((sum, value) => sum + value, metadata.intercept);
}

/** Evidence for every signal, the ones raising risk most first. */
export function buildTextEvidence(vector: ArrayLike<number>, metadata: TextModelMetadata): EvidenceItem[] {
  const contributions = textContributions(vector, metadata);
  return metadata.feature_names
    .map((feature, i) => {
      const copy = EXPLANATIONS[feature];
      const aboveAverage = (vector[i] ?? 0) > metadata.scaler_mean[i]!;
      return {
        feature,
        contribution: contributions[i]!,
        explanation: copy ? copy[aboveAverage ? "high" : "low"] : feature.replaceAll("_", " "),
      };
    })
    .sort((a, b) => b.contribution - a.contribution);
}

/** The manipulative phrases found in the message, in order, never overlapping. */
export function findTextHighlights(matches: LexiconMatch[]): Highlight[] {
  const found = matches
    .filter((match) => HIGHLIGHT_REASONS[match.category] !== undefined)
    .map((match) => ({ start: match.start, end: match.end, reason: HIGHLIGHT_REASONS[match.category]! }))
    .sort((a, b) => a.start - b.start || b.end - a.end);
  const result: Highlight[] = [];
  for (const highlight of found) {
    const last = result[result.length - 1];
    if (!last || highlight.start >= last.end) result.push(highlight);
  }
  return result;
}
