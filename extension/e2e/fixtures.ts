import { createHash } from "node:crypto";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { type BrowserContext, type Page, test as base, chromium } from "@playwright/test";

const dist = resolve(import.meta.dirname, "../dist");

// Which browser runs the extension. Playwright's bundled Chromium is the
// default; on Windows machines whose Application Control policy blocks that
// unsigned build, installed Edge is used instead. Override with PW_CHANNEL.
export const browserChannel = process.env.PW_CHANNEL ?? (process.platform === "win32" ? "msedge" : "chromium");

/**
 * Chromium derives an unpacked extension's id from its folder path: the first
 * 16 bytes of the SHA-256 of the path (UTF-16 on Windows, UTF-8 elsewhere),
 * written with the letters a-p. The extension has no background page to ask,
 * so the tests compute it the same way.
 */
function unpackedExtensionId(path: string): string {
  const bytes = Buffer.from(path, process.platform === "win32" ? "utf16le" : "utf8");
  return [...createHash("sha256").update(bytes).digest().subarray(0, 16)]
    .map((b) => String.fromCharCode(97 + (b >> 4), 97 + (b & 15)))
    .join("");
}

interface Fixtures {
  context: BrowserContext;
  extensionId: string;
  /** Requests that left the browser for anywhere other than the extension itself. */
  outsideRequests: string[];
  popup: Page;
}

export const test = base.extend<Fixtures>({
  // eslint-disable-next-line no-empty-pattern
  context: async ({}, use) => {
    const context = await chromium.launchPersistentContext(mkdtempSync(join(tmpdir(), "phishguard-e2e-")), {
      channel: browserChannel, // a full browser: the headless shell cannot load extensions
      args: [`--disable-extensions-except=${dist}`, `--load-extension=${dist}`],
    });
    await use(context);
    await context.close();
  },

  outsideRequests: async ({ context }, use) => {
    const outside: string[] = [];
    context.on("request", (request) => {
      const url = request.url();
      if (!/^(chrome-extension|chrome|edge|devtools|data|blob|about):/.test(url)) outside.push(url);
    });
    await use(outside);
  },

  // eslint-disable-next-line no-empty-pattern
  extensionId: async ({}, use) => {
    await use(unpackedExtensionId(dist));
  },

  popup: async ({ context, extensionId, outsideRequests }, use) => {
    void outsideRequests; // start recording requests before the popup opens
    const page = await context.newPage();
    await page.setViewportSize({ width: 420, height: 900 });
    try {
      await page.goto(`chrome-extension://${extensionId}/src/popup/index.html`);
    } catch {
      throw new Error("the extension did not load: run `pnpm build` first");
    }
    await use(page);
  },
});

export { expect } from "@playwright/test";

export async function finishOnboarding(popup: Page) {
  await popup.getByRole("button", { name: "Got it" }).click();
}

export async function analysePasted(popup: Page, text: string) {
  // Mark any result already on screen, so we wait for this analysis and not the last one.
  await popup.evaluate(() => document.querySelector(".result")?.setAttribute("data-stale", ""));
  await popup.getByLabel("Or paste a link or a message").fill(text);
  await popup.getByRole("button", { name: "Analyse pasted text" }).click();
  await popup.locator(".result:not([data-stale]) [data-testid=model-version]").waitFor();
}
