import { FEATURE_NAMES, SUSPICIOUS_KEYWORDS } from "./featurise";

/** Contents of models/url-model.json, written by the training pipeline. */
export interface ModelMetadata {
  model_version: string;
  onnx_sha256: string;
  input_name: string;
  feature_names: string[];
  suspicious_keywords: string[];
  scaler_mean: number[];
  scaler_scale: number[];
  coefficients: number[];
  intercept: number;
  threshold: number;
}

/**
 * Refuse a model this build of the extension cannot featurise for. Feeding a
 * model features in a different order than it was trained on produces
 * confident nonsense, so a mismatch must be a hard failure.
 */
export function assertCompatible(metadata: ModelMetadata): void {
  const same = (a: readonly string[], b: readonly string[]) =>
    a.length === b.length && a.every((value, i) => value === b[i]);
  if (!same(metadata.feature_names, FEATURE_NAMES)) {
    throw new Error("model feature list does not match this extension build");
  }
  if (!same(metadata.suspicious_keywords, SUSPICIOUS_KEYWORDS)) {
    throw new Error("model keyword list does not match this extension build");
  }
  if (!(metadata.threshold > 0 && metadata.threshold < 1)) {
    throw new Error("model has no usable decision threshold");
  }
}

export function parseKnownSites(text: string): Set<string> {
  return new Set(text.split(/\s+/).filter(Boolean));
}

export async function sha256Hex(data: ArrayBuffer | string): Promise<string> {
  const bytes = typeof data === "string" ? new TextEncoder().encode(data) : data;
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
