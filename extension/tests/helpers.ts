import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { LoadedModels } from "../src/lib/analyse";
import { logOdds } from "../src/lib/evidence";
import type { ModelMetadata } from "../src/lib/model";
import { parseKnownSites } from "../src/lib/model";
import { textLogOdds } from "../src/lib/text-evidence";
import type { TextModelMetadata } from "../src/lib/text-features";

const root = resolve(import.meta.dirname, "../..");
const readJson = (path: string) => JSON.parse(readFileSync(path, "utf-8"));

export const modelPath = (name: string) => resolve(root, "models", name);

export const metadata: ModelMetadata = readJson(modelPath("url-model.json"));
export const textMetadata: TextModelMetadata = readJson(modelPath("text-model.json"));

export const knownSites = parseKnownSites(readFileSync(modelPath("known-sites.txt"), "utf-8"));

export const parityFixture: { url: string; features: number[]; probability: number }[] = readJson(
  resolve(import.meta.dirname, "fixtures/url-parity.json"),
);
export const textFixture: { label: "legit" | "scam"; text: string; features: number[]; probability: number }[] =
  readJson(resolve(import.meta.dirname, "fixtures/text-parity.json"));

export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

// Both models scored from their published coefficients instead of ONNX Runtime;
// model.test.ts proves the two agree.
export const models: LoadedModels = {
  url: { metadata, knownSites, score: async (features) => sigmoid(logOdds(features, metadata)) },
  text: { metadata: textMetadata, score: async (features) => sigmoid(textLogOdds(features, textMetadata)) },
  loadMs: 0,
};
