import { defineConfig } from "@playwright/test";

// `pnpm bench`: the latency benchmark. Kept out of the normal end-to-end run
// because its result depends on the machine and it rewrites committed files.
export default defineConfig({
  testDir: "e2e",
  testMatch: "*.bench.ts",
  workers: 1,
  timeout: 300_000,
  reporter: [["list"]],
  outputDir: "test-results",
});
