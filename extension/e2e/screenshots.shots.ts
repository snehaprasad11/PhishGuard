// Captures the screenshots used in the README and the store listing.
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";
import type { Page } from "@playwright/test";
import { analysePasted, expect, finishOnboarding, test } from "./fixtures";

const out = resolve(import.meta.dirname, "../../docs/screenshots");
mkdirSync(out, { recursive: true });

const SCAM_MESSAGE =
  "URGENT: your account has been suspended. Verify your identity now at http://secure-login.account-verify.top/confirm or it will be closed.";

const shot = (page: Page, name: string) => page.locator("main").screenshot({ path: resolve(out, `${name}.png`) });

test("screenshots", async ({ popup, context, extensionId }) => {
  await popup.setViewportSize({ width: 412, height: 900 });
  await expect(popup.getByText("Welcome to PhishGuard")).toBeVisible();
  await shot(popup, "onboarding");
  await finishOnboarding(popup);

  await analysePasted(popup, SCAM_MESSAGE);
  await shot(popup, "phishing-message");

  await analysePasted(popup, "https://accounts.google.com/signin");
  await shot(popup, "no-warning-signs");

  await analysePasted(
    popup,
    "Dear customer, your account will be suspended. Verify your password immediately by clicking here: https://accounts.google.com/signin",
  );
  await shot(popup, "scam-wording-real-link");

  // Dark theme: the palette must stay readable, and verdicts distinguishable, in both.
  await popup.emulateMedia({ colorScheme: "dark" });
  await analysePasted(popup, SCAM_MESSAGE);
  await shot(popup, "phishing-message-dark");
  await popup.emulateMedia({ colorScheme: "light" });

  const options = await context.newPage();
  await options.setViewportSize({ width: 720, height: 700 });
  await options.goto(`chrome-extension://${extensionId}/src/options/index.html`);
  await expect(options.getByText("PhishGuard options")).toBeVisible();
  await shot(options, "options");

  const card = await context.newPage();
  await card.setViewportSize({ width: 900, height: 1000 });
  await card.goto(`chrome-extension://${extensionId}/models/model_card.html`);
  await expect(card.getByRole("heading", { name: "PhishGuard text model" })).toBeVisible();
  await card.screenshot({ path: resolve(out, "model-card.png"), fullPage: true });
});
