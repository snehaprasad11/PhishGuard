import { defineConfig } from "@playwright/test";

// `pnpm screenshots`: regenerates docs/screenshots/ from the built extension, so
// the pictures in the README always show the real interface.
export default defineConfig({
  testDir: "e2e",
  testMatch: "*.shots.ts",
  workers: 1,
  timeout: 120_000,
  reporter: [["list"]],
  outputDir: "test-results",
});
