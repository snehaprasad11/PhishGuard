// The contract between the popup and the inference worker (spec §11).

export const MAX_URL_LENGTH = 8192;
export const MAX_TEXT_LENGTH = 20_000;

export type Confidence = "low" | "medium" | "high";
export type Verdict = "phishing" | "suspicious" | "clear";

/** At least one of `url` and `text` must be present. */
export interface AnalyseRequest {
  type: "ANALYSE";
  /** A web address to score with the URL model. */
  url?: string;
  /** The wording of a pasted message to score with the text model. */
  text?: string;
}

export interface EvidenceItem {
  feature: string;
  /** Change in the model's log-odds caused by this feature; positive raises risk. */
  contribution: number;
  explanation: string;
}

export interface Highlight {
  /** Character offsets into `PartResult.displayText`. */
  start: number;
  end: number;
  reason: string;
}

/** One model's judgement of one thing: the address, or the wording of the message. */
export interface PartResult {
  /** What was scored, as shown to the user; highlights index into it. */
  displayText: string;
  score: number; // 0..100
  probability: number; // 0..1, calibrated
  verdict: Verdict;
  confidence: Confidence;
  modelVersion: string;
  evidence: EvidenceItem[];
  highlights: Highlight[];
}

export interface AnalyseResponse {
  type: "ANALYSE_RESULT";
  /** The more serious of the parts' verdicts, and the higher of their scores. */
  verdict: Verdict;
  score: number;
  address?: PartResult;
  message?: PartResult;
  timing: { loadMs: number; analyseMs: number };
}

export interface ErrorResponse {
  type: "ERROR";
  code: "INVALID_INPUT" | "MODEL_LOAD_FAILED" | "INFERENCE_FAILED";
  message: string;
}

export type WorkerResponse = AnalyseResponse | ErrorResponse;

/** Returns the reason a request is unusable, or null if it is fine. */
export function validateRequest(message: unknown): string | null {
  if (typeof message !== "object" || message === null) return "request must be an object";
  const { type, url, text } = message as Record<string, unknown>;
  if (type !== "ANALYSE") return "unknown request type";
  for (const [name, value, limit] of [
    ["url", url, MAX_URL_LENGTH],
    ["text", text, MAX_TEXT_LENGTH],
  ] as const) {
    if (value === undefined) continue;
    if (typeof value !== "string" || value.trim() === "") return `${name} must be a non-empty string`;
    if (value.length > limit) return `${name} is longer than ${limit} characters`;
  }
  if (url === undefined && text === undefined) return "url or text must be a non-empty string";
  return null;
}

const URL_IN_TEXT = /\b((?:https?:\/\/|www\.)[^\s<>"'`]+|[a-z0-9-]+(?:\.[a-z0-9-]+)+\/[^\s<>"'`]*)/i;

// Endings accepted for a bare name such as `evil-site.top`. Without this, a sentence with a
// missing space ("see you today.Reply soon") would be mistaken for an address.
const BARE_NAME_ENDINGS = new Set(
  (
    "com org net edu gov info biz io co me ly app dev xyz top shop site online store club live " +
    "link click icu vip work cloud tech page pro cc tv ai uk us in de fr es it nl ru cn jp br au " +
    "ca ch se no pl id ph vn za mx"
  ).split(" "),
);
const BARE_NAME = /\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.([a-z]{2,})\b/gi;

/** Pull the first web address out of pasted text such as an SMS or email body. */
export function firstUrlIn(text: string): string | null {
  const match = URL_IN_TEXT.exec(text);
  if (match) return match[0].replace(/[.,;:!?)\]]+$/, "");
  for (const bare of text.matchAll(BARE_NAME)) {
    if (BARE_NAME_ENDINGS.has(bare[1]!.toLowerCase())) return bare[0];
  }
  return null;
}

const SEVERITY: Record<Verdict, number> = { clear: 0, suspicious: 1, phishing: 2 };

export function worstVerdict(verdicts: Verdict[]): Verdict {
  return verdicts.reduce((worst, v) => (SEVERITY[v] > SEVERITY[worst] ? v : worst), "clear" as Verdict);
}

/** The part that decided the overall verdict: the most severe, then the most probable. */
export function decidingPart(result: AnalyseResponse): PartResult {
  const parts = [result.address, result.message].filter((part): part is PartResult => part !== undefined);
  return parts.reduce((a, b) =>
    SEVERITY[b.verdict] > SEVERITY[a.verdict] ||
    (SEVERITY[b.verdict] === SEVERITY[a.verdict] && b.probability > a.probability)
      ? b
      : a,
  );
}
