/// <reference lib="webworker" />
// Runs the models off the UI thread. Everything it needs is packaged inside the
// extension, so loading and scoring make no network request to any server.

import * as ort from "onnxruntime-web/wasm";
import { type LoadedModels, type Scorer, analyse } from "../lib/analyse";
import { type AnalyseRequest, type ErrorResponse, type WorkerResponse, validateRequest } from "../lib/messages";
import { type ModelMetadata, assertCompatible, parseKnownSites, sha256Hex } from "../lib/model";
import { type TextModelMetadata, assertTextCompatible } from "../lib/text-features";

const asset = (path: string) => new URL(path, self.location.origin).href;

async function fetchOk(path: string): Promise<Response> {
  const response = await fetch(asset(path));
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response;
}

/** Load one packaged ONNX model, refusing it if the file is not the one its metadata describes. */
async function loadScorer(name: string, metadata: { onnx_sha256: string; input_name: string }): Promise<Scorer> {
  const bytes = await fetchOk(`/models/${name}.onnx`).then((r) => r.arrayBuffer());
  if ((await sha256Hex(bytes)) !== metadata.onnx_sha256) {
    throw new Error(`${name}: model file does not match the hash in its metadata`);
  }
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  const probabilityOutput = session.outputNames[1]!;
  return async (features) => {
    const input = new ort.Tensor("float32", features, [1, features.length]);
    const output = await session.run({ [metadata.input_name]: input });
    // Shape [1, 2]: probabilities of benign and phishing.
    return Number((output[probabilityOutput]!.data as Float32Array)[1]);
  };
}

async function loadModels(): Promise<LoadedModels> {
  const started = performance.now();
  // Single-threaded: multi-threaded WASM needs cross-origin isolation, and
  // models this small gain nothing from it.
  ort.env.wasm.numThreads = 1;

  const [urlMetadata, textMetadata, knownSitesText] = await Promise.all([
    fetchOk("/models/url-model.json").then((r) => r.json() as Promise<ModelMetadata>),
    fetchOk("/models/text-model.json").then((r) => r.json() as Promise<TextModelMetadata>),
    fetchOk("/models/known-sites.txt").then((r) => r.text()),
  ]);
  assertCompatible(urlMetadata);
  assertTextCompatible(textMetadata);
  const [urlScore, textScore] = [
    await loadScorer("url-model", urlMetadata),
    await loadScorer("text-model", textMetadata),
  ];
  return {
    url: { metadata: urlMetadata, knownSites: parseKnownSites(knownSitesText), score: urlScore },
    text: { metadata: textMetadata, score: textScore },
    loadMs: performance.now() - started,
  };
}

// Loaded once per worker; a failed load is retried on the next request.
let models: Promise<LoadedModels> | null = null;

function errorResponse(code: ErrorResponse["code"], error: unknown): ErrorResponse {
  // Only the error text: a URL or message must never end up in an error or a log.
  return { type: "ERROR", code, message: error instanceof Error ? error.message : String(error) };
}

async function handle(message: unknown): Promise<WorkerResponse> {
  const invalid = validateRequest(message);
  if (invalid) return { type: "ERROR", code: "INVALID_INPUT", message: invalid };

  let loaded: LoadedModels;
  try {
    models ??= loadModels();
    loaded = await models;
  } catch (error) {
    models = null;
    return errorResponse("MODEL_LOAD_FAILED", error);
  }
  try {
    return await analyse(message as AnalyseRequest, loaded);
  } catch (error) {
    const code = error instanceof Error && error.message === "empty URL" ? "INVALID_INPUT" : "INFERENCE_FAILED";
    return errorResponse(code, error);
  }
}

self.onmessage = async (event: MessageEvent) => {
  self.postMessage(await handle(event.data));
};
