import type { AnalyseRequest, WorkerResponse } from "./messages";

const TIMEOUT_MS = 20_000;

/** Owns the inference worker and turns its messages into promises, one request at a time. */
export class InferenceClient {
  private worker: Worker | null = null;
  private queue: Promise<unknown> = Promise.resolve();

  private start(): Worker {
    this.worker ??= new Worker(new URL("../worker/inference-worker.ts", import.meta.url), {
      type: "module",
    });
    return this.worker;
  }

  analyse(input: { url?: string; text?: string }): Promise<WorkerResponse> {
    const run = () =>
      new Promise<WorkerResponse>((resolve) => {
        const fail = (message: string) => {
          // A broken worker is discarded so the next click starts a fresh one.
          this.worker?.terminate();
          this.worker = null;
          resolve({ type: "ERROR", code: "INFERENCE_FAILED", message });
        };
        let worker: Worker;
        try {
          worker = this.start();
        } catch {
          return fail("could not start the analysis worker");
        }
        const timer = setTimeout(() => fail("analysis timed out"), TIMEOUT_MS);
        worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
          clearTimeout(timer);
          resolve(event.data);
        };
        worker.onerror = () => {
          clearTimeout(timer);
          fail("the analysis worker crashed");
        };
        worker.postMessage({ type: "ANALYSE", ...input } satisfies AnalyseRequest);
      });
    const result = this.queue.then(run, run);
    this.queue = result;
    return result;
  }
}
