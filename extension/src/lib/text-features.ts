// TypeScript mirror of training/text/features.py (the shipped featuriser).
//
// The phrase lists and the feature order are not duplicated here: they come
// from models/text-model.json, written by the training pipeline, so the
// browser cannot drift from the model. What is mirrored is the tokeniser and
// the arithmetic, and tests/text.test.ts checks both against values exported
// from Python.

export interface TextModelMetadata {
  model_version: string;
  onnx_sha256: string;
  input_name: string;
  max_chars: number;
  feature_names: string[];
  lexicon: Record<string, string[]>;
  scaler_mean: number[];
  scaler_scale: number[];
  coefficients: number[];
  intercept: number;
  threshold: number;
}

export interface Token {
  /** A lowercase word, or one of the placeholders _url_, _email_, _num_, _cur_. */
  text: string;
  /** Character offsets into the clipped message. */
  start: number;
  end: number;
}

export interface LexiconMatch {
  category: string;
  /** Character offsets into the clipped message. */
  start: number;
  end: number;
}

// Same alternatives, in the same order, as _TOKEN in features.py. Only ASCII
// classes, so the two engines agree.
const TOKEN =
  /(?<url>(?:https?:\/\/|www\.)[^ \t\r\n]+)|(?<email>[a-z0-9._%+-]+@[a-z0-9.-]+)|(?<num>[0-9]+(?:[.,][0-9]+)*)|(?<word>[a-z]+(?:'[a-z]+)?)|(?<cur>[$£€₹])/g;
const PLACEHOLDER: Record<string, string> = { url: "_url_", email: "_email_", num: "_num_", cur: "_cur_" };

// Features that are not lexicon categories; each must be computed below.
const SURFACE_FEATURES = [
  "signal_kinds",
  "number_count",
  "currency_count",
  "token_count",
  "uppercase_ratio",
  "exclamation_count",
];
// The one lexicon category that is not a way of manipulating the reader.
const NOT_MANIPULATION = "first_person";

/** The first `maxChars` characters, counted as Unicode code points like Python's slicing. */
export function clip(text: string, maxChars: number): string {
  return text.length <= maxChars ? text : [...text].slice(0, maxChars).join("");
}

export function tokenise(clipped: string): Token[] {
  // Lowercasing only A-Z keeps every offset identical to the original text.
  const lowered = clipped.replace(/[A-Z]/g, (ch) => ch.toLowerCase());
  const tokens: Token[] = [];
  for (const match of lowered.matchAll(TOKEN)) {
    const groups = match.groups!;
    const start = match.index;
    const end = start + match[0].length;
    if (groups["word"] !== undefined) {
      if (match[0].length >= 2) tokens.push({ text: match[0], start, end });
    } else {
      const kind = Object.keys(PLACEHOLDER).find((name) => groups[name] !== undefined)!;
      tokens.push({ text: PLACEHOLDER[kind]!, start, end });
    }
  }
  return tokens;
}

/**
 * Every lexicon phrase found. At each position the longest matching phrase of
 * a category wins, so "final notice" counts once.
 */
export function lexiconMatches(tokens: Token[], lexicon: Record<string, string[]>): LexiconMatch[] {
  const found: LexiconMatch[] = [];
  for (const [category, phrases] of Object.entries(lexicon)) {
    const set = new Set(phrases);
    const longest = Math.max(...phrases.map((phrase) => phrase.split(" ").length));
    for (let start = 0; start < tokens.length; start++) {
      for (let length = Math.min(longest, tokens.length - start); length > 0; length--) {
        const phrase = tokens.slice(start, start + length).map((token) => token.text).join(" ");
        if (set.has(phrase)) {
          found.push({ category, start: tokens[start]!.start, end: tokens[start + length - 1]!.end });
          break;
        }
      }
    }
  }
  return found;
}

export interface TextFeatures {
  /** In the order of `metadata.feature_names`. */
  vector: Float32Array;
  values: Record<string, number>;
  matches: LexiconMatch[];
  clipped: string;
}

export function extractTextFeatures(text: string, metadata: TextModelMetadata): TextFeatures {
  const clipped = clip(text, metadata.max_chars);
  const tokens = tokenise(clipped);
  const matches = lexiconMatches(tokens, metadata.lexicon);
  const categories = Object.keys(metadata.lexicon);

  const counts: Record<string, number> = Object.fromEntries(categories.map((category) => [category, 0]));
  for (const match of matches) counts[match.category]!++;
  const countTokens = (text: string) => tokens.filter((token) => token.text === text).length;
  let letters = 0;
  let capitals = 0;
  let exclamations = 0;
  for (const ch of clipped) {
    if (ch >= "A" && ch <= "Z") capitals++;
    if ((ch >= "a" && ch <= "z") || (ch >= "A" && ch <= "Z")) letters++;
    if (ch === "!") exclamations++;
  }

  const values: Record<string, number> = {};
  for (const category of categories) values[category] = Math.log1p(counts[category]!);
  values["signal_kinds"] = categories.filter((c) => c !== NOT_MANIPULATION && counts[c]! > 0).length;
  values["number_count"] = Math.log1p(countTokens("_num_"));
  values["currency_count"] = Math.log1p(countTokens("_cur_"));
  values["token_count"] = Math.log1p(tokens.length);
  values["uppercase_ratio"] = letters ? capitals / letters : 0;
  values["exclamation_count"] = Math.log1p(exclamations);

  return {
    vector: Float32Array.from(metadata.feature_names, (name) => values[name]!),
    values,
    matches,
    clipped,
  };
}

/** Refuse a text model whose features this build does not know how to compute. */
export function assertTextCompatible(metadata: TextModelMetadata): void {
  const computable = new Set([...Object.keys(metadata.lexicon), ...SURFACE_FEATURES]);
  const unknown = metadata.feature_names.filter((name) => !computable.has(name));
  if (unknown.length > 0) {
    throw new Error(`text model has features this extension build cannot compute: ${unknown.join(", ")}`);
  }
  const n = metadata.feature_names.length;
  if ([metadata.scaler_mean, metadata.scaler_scale, metadata.coefficients].some((a) => a.length !== n)) {
    throw new Error("text model metadata is inconsistent");
  }
  if (!(metadata.threshold > 0 && metadata.threshold < 1)) {
    throw new Error("text model has no usable decision threshold");
  }
}

/** True when pasted text is more than just a link: only then is its wording worth analysing. */
export function hasWording(text: string, maxChars: number): boolean {
  const words = tokenise(clip(text, maxChars)).filter((token) => !token.text.startsWith("_"));
  return words.length >= 3;
}
