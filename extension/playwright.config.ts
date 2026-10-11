import { defineConfig } from "@playwright/test";

// End-to-end tests load the built extension (dist/) into a real Chromium, so
// run `pnpm build` first. One worker: every test drives its own browser profile.
export default defineConfig({
  testDir: "e2e",
  testMatch: "*.e2e.ts",
  workers: 1,
  timeout: 60_000,
  reporter: [["list"]],
  outputDir: "test-results",
});
