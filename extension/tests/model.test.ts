// Runs the published ONNX file through ONNX Runtime Web (the same runtime the
// extension's worker uses) and checks it against what Python computed.
import { readFileSync } from "node:fs";
import * as ort from "onnxruntime-web/wasm";
import { beforeAll, describe, expect, it } from "vitest";
import { logOdds } from "../src/lib/evidence";
import { featurise } from "../src/lib/featurise";
import { sha256Hex } from "../src/lib/model";
import { extractTextFeatures } from "../src/lib/text-features";
import { knownSites, metadata, modelPath, parityFixture, sigmoid, textFixture, textMetadata } from "./helpers";

describe("published ONNX model in ONNX Runtime Web", () => {
  const bytes = readFileSync(modelPath("url-model.onnx"));
  let session: ort.InferenceSession;

  beforeAll(async () => {
    ort.env.wasm.numThreads = 1;
    session = await ort.InferenceSession.create(new Uint8Array(bytes), { executionProviders: ["wasm"] });
  });

  async function probability(url: string): Promise<number> {
    const features = featurise(url, knownSites);
    const input = new ort.Tensor("float32", features, [1, features.length]);
    const output = await session.run({ [metadata.input_name]: input });
    return Number((output[session.outputNames[1]!]!.data as Float32Array)[1]);
  }

  it("matches the hash recorded in its metadata", async () => {
    const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    expect(await sha256Hex(buffer as ArrayBuffer)).toBe(metadata.onnx_sha256);
  });

  it.each(parityFixture)("gives Python's probability for $url", async ({ url, probability: expected }) => {
    expect(await probability(url)).toBeCloseTo(expected, 4);
  });

  it("agrees with the score rebuilt from the evidence contributions", async () => {
    for (const { url } of parityFixture) {
      const rebuilt = sigmoid(logOdds(featurise(url, knownSites), metadata));
      expect(await probability(url)).toBeCloseTo(rebuilt, 4);
    }
  });
});

describe("published text model in ONNX Runtime Web", () => {
  const bytes = readFileSync(modelPath("text-model.onnx"));
  let session: ort.InferenceSession;

  beforeAll(async () => {
    ort.env.wasm.numThreads = 1;
    session = await ort.InferenceSession.create(new Uint8Array(bytes), { executionProviders: ["wasm"] });
  });

  it("matches the hash recorded in its metadata", async () => {
    const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    expect(await sha256Hex(buffer as ArrayBuffer)).toBe(textMetadata.onnx_sha256);
  });

  it.each(textFixture)("gives Python's probability for: $text", async ({ text, probability }) => {
    const { vector } = extractTextFeatures(text, textMetadata);
    const input = new ort.Tensor("float32", vector, [1, vector.length]);
    const output = await session.run({ [textMetadata.input_name]: input });
    expect(Number((output[session.outputNames[1]!]!.data as Float32Array)[1])).toBeCloseTo(probability, 4);
  });
});
