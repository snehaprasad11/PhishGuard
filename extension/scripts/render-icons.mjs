// Renders icons/icon.svg to the PNG sizes the manifest needs: `pnpm icons`.
// Uses the same browser as the end-to-end tests, so no image library is required.
/* global process, console */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { chromium } from "@playwright/test";

const icons = resolve(import.meta.dirname, "../icons");
const svg = readFileSync(resolve(icons, "icon.svg"), "utf-8");
const channel = process.env.PW_CHANNEL ?? (process.platform === "win32" ? "msedge" : "chromium");

const browser = await chromium.launch({ channel });
for (const size of [16, 48, 128]) {
  const page = await browser.newPage({ viewport: { width: size, height: size } });
  await page.setContent(
    `<style>html,body{margin:0;background:transparent}svg{display:block;width:${size}px;height:${size}px}</style>${svg}`,
  );
  await page.screenshot({ path: resolve(icons, `icon${size}.png`), omitBackground: true });
  await page.close();
}
await browser.close();
console.log("rendered icon16.png, icon48.png, icon128.png");
