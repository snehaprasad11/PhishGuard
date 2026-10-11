import { buildEvidence, findHighlights } from "./evidence";
import { featurise, parseUrl } from "./featurise";
import { type AnalyseRequest, type AnalyseResponse, type PartResult, worstVerdict } from "./messages";
import type { ModelMetadata } from "./model";
import { buildTextEvidence, findTextHighlights } from "./text-evidence";
import { type TextModelMetadata, extractTextFeatures } from "./text-features";
import { confidenceFor, riskScore, verdictFor } from "./verdict";

/** Runs a model on a feature vector and returns the phishing probability. */
export type Scorer = (features: Float32Array) => Promise<number>;

export interface LoadedModels {
  url: { metadata: ModelMetadata; knownSites: ReadonlySet<string>; score: Scorer };
  text: { metadata: TextModelMetadata; score: Scorer };
  loadMs: number;
}

// How much of a long message is shown back in the popup.
const MAX_DISPLAYED_MESSAGE = 600;

function part(
  probability: number,
  threshold: number,
  rest: Pick<PartResult, "displayText" | "modelVersion" | "evidence" | "highlights">,
): PartResult {
  return {
    ...rest,
    probability,
    score: riskScore(probability),
    verdict: verdictFor(probability, threshold),
    confidence: confidenceFor(probability, threshold),
  };
}

export async function analyseAddress(url: string, model: LoadedModels["url"]): Promise<PartResult> {
  const parsed = parseUrl(url);
  const features = featurise(url, model.knownSites);
  return part(await model.score(features), model.metadata.threshold, {
    displayText: parsed.normalised,
    modelVersion: model.metadata.model_version,
    evidence: buildEvidence(features, model.metadata),
    highlights: findHighlights(parsed, model.knownSites),
  });
}

export async function analyseMessage(text: string, model: LoadedModels["text"]): Promise<PartResult> {
  const features = extractTextFeatures(text, model.metadata);
  const shown = features.clipped.slice(0, MAX_DISPLAYED_MESSAGE);
  return part(await model.score(features.vector), model.metadata.threshold, {
    displayText: shown.length < features.clipped.length ? shown + "…" : shown,
    modelVersion: model.metadata.model_version,
    evidence: buildTextEvidence(features.vector, model.metadata),
    highlights: findTextHighlights(features.matches).filter((h) => h.end <= shown.length),
  });
}

/**
 * Everything between "here is what the user gave us" and "here is what to
 * show": featurise, score, explain. Kept free of worker and ONNX specifics so
 * it can be tested with any scorer.
 */
export async function analyse(request: AnalyseRequest, models: LoadedModels): Promise<AnalyseResponse> {
  const started = performance.now();
  const address = request.url === undefined ? undefined : await analyseAddress(request.url, models.url);
  const message = request.text === undefined ? undefined : await analyseMessage(request.text, models.text);
  const parts = [address, message].filter((p): p is PartResult => p !== undefined);
  return {
    type: "ANALYSE_RESULT",
    verdict: worstVerdict(parts.map((p) => p.verdict)),
    score: Math.max(...parts.map((p) => p.score)),
    ...(address && { address }),
    ...(message && { message }),
    timing: { loadMs: models.loadMs, analyseMs: performance.now() - started },
  };
}
