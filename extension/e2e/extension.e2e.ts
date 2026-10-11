import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { analysePasted, expect, finishOnboarding, test } from "./fixtures";

const PHISHING_SMS =
  "Your parcel is on hold. Confirm delivery: http://secure-login.account-verify.dhl-update.top/confirm?id=829 today";
const readModel = (name: string) =>
  JSON.parse(readFileSync(resolve(import.meta.dirname, "../../models", name), "utf-8"));
const metadata = readModel("url-model.json");
const textMetadata = readModel("text-model.json");

test("first run shows the privacy promise once", async ({ popup }) => {
  await expect(popup.getByText("never scans anything until you click Analyse")).toBeVisible();
  await finishOnboarding(popup);
  await expect(popup.getByText("Nothing analysed yet")).toBeVisible();
  await popup.reload();
  await expect(popup.getByText("Nothing analysed yet")).toBeVisible();
});

test("a pasted phishing message gets a verdict, evidence and highlights", async ({ popup }, testInfo) => {
  await finishOnboarding(popup);
  await analysePasted(popup, PHISHING_SMS);

  await expect(popup.getByRole("heading", { name: "Likely phishing", level: 2 })).toBeVisible();
  const meter = popup.getByRole("meter");
  expect(Number(await meter.getAttribute("aria-valuenow"))).toBeGreaterThan(50);
  await expect(popup.locator("[aria-live=polite]")).toContainText("Likely phishing. Risk score");

  // The address is shown normalised, with the suspicious parts marked and explained.
  // The link: shown normalised, with the suspicious parts marked and explained.
  const link = popup.locator("[data-part=address]");
  await expect(link.locator(".address")).toHaveText("secure-login.account-verify.dhl-update.top/confirm?id=829");
  await expect(link.locator(".address mark")).toContainText(["secure", "login", "account", "verify"]);
  await expect(link.locator(".reasons")).toContainText("on a site that is not well known");
  await expect(link.locator(".evidence li").first()).toBeVisible();

  // The wording: the pasted message, judged separately by the text model.
  const wording = popup.locator("[data-part=message]");
  await expect(wording.locator(".address")).toContainText("Your parcel is on hold");
  await expect(wording.locator(".address mark").first()).toBeVisible();
  await expect(popup.getByTestId("model-version")).toContainText(textMetadata.model_version);
  await expect(popup.getByText("Do not enter a password")).toBeVisible();
  await expect(popup.getByTestId("model-version")).toContainText(metadata.model_version);

  await link.getByText("All factors").click();
  await expect(link.locator(".factors li")).toHaveCount(20);
  await popup.screenshot({ path: testInfo.outputPath("phishing.png"), fullPage: true });
});

test("a well-known sign-in page raises no warning, and is never called safe", async ({ popup }, testInfo) => {
  await finishOnboarding(popup);
  await analysePasted(popup, "https://accounts.google.com/signin");
  await expect(popup.getByRole("heading", { name: "No warning signs found", level: 2 })).toBeVisible();
  await expect(popup.locator("[data-part=address] .address mark")).toHaveCount(0);
  await expect(popup.locator("[data-part=message]")).toHaveCount(0); // a bare link has no wording
  await expect(popup.getByText("This is not a guarantee")).toBeVisible();
  await expect(popup.locator(".result")).not.toContainText(/\bis safe\b/i);
  await popup.screenshot({ path: testInfo.outputPath("clear.png"), fullPage: true });
});

test("a scam message is flagged by its wording even when its link is to a well-known site", async ({ popup }, testInfo) => {
  await finishOnboarding(popup);
  await analysePasted(
    popup,
    "Dear customer, your account will be suspended. Verify your password immediately by clicking here: https://accounts.google.com/signin",
  );
  await expect(popup.locator("[data-part=address] .part-verdict")).toContainText("no warning signs");
  await expect(popup.locator("[data-part=message] .part-verdict")).not.toContainText("no warning signs");
  await expect(popup.locator("[data-part=message] .address mark")).toContainText(["Dear customer", "suspended"]);
  await expect(popup.getByRole("heading", { name: "No warning signs found", level: 2 })).toHaveCount(0);
  await popup.screenshot({ path: testInfo.outputPath("message.png"), fullPage: true });
});

test("an ordinary message with no link is analysed and not flagged", async ({ popup }) => {
  await finishOnboarding(popup);
  await analysePasted(popup, "Hi, your parcel will be delivered today between 2pm and 4pm. Track it in the app.");
  await expect(popup.getByRole("heading", { name: "No warning signs found", level: 2 })).toBeVisible();
  await expect(popup.locator("[data-part=address]")).toHaveCount(0);
  await expect(popup.locator("[data-part=message]")).toBeVisible();
  // The wording of a message is never stored, even in history.
  expect(JSON.stringify(await popup.evaluate(() => chrome.storage.local.get(null)))).not.toContain("parcel");
});

test("nothing analysable gives a friendly error and a retry", async ({ popup }) => {
  await finishOnboarding(popup);
  await popup.getByLabel("Or paste a link or a message").fill("?? !!");
  await popup.getByRole("button", { name: "Analyse pasted text" }).click();
  await expect(popup.getByRole("alert")).toContainText("Nothing to analyse");
  await expect(popup.getByRole("button", { name: "Try again" })).toBeVisible();
});

test("history stores no address by default and can be cleared", async ({ popup }) => {
  await finishOnboarding(popup);
  await analysePasted(popup, PHISHING_SMS);
  await analysePasted(popup, "https://accounts.google.com/signin");

  await popup.getByText("History (2)").click();
  await expect(popup.locator(".history li")).toHaveCount(2);
  await expect(popup.locator(".history")).toContainText("address not stored");
  const stored = JSON.stringify(await popup.evaluate(() => chrome.storage.local.get(null)));
  expect(stored).not.toContain("dhl-update");
  expect(stored).not.toContain("google");

  await popup.getByRole("button", { name: "Clear history" }).click();
  await expect(popup.getByText("History (0)")).toBeVisible();
  await popup.reload();
  await expect(popup.getByText("History (0)")).toBeVisible();
});

test("options: addresses are kept only on request, without the query string", async ({ popup, context, extensionId }) => {
  await finishOnboarding(popup);
  const options = await context.newPage();
  await options.goto(`chrome-extension://${extensionId}/src/options/index.html`);
  await options.getByLabel("Show addresses in history").check();
  await expect(options.getByRole("status")).toHaveText("Saved.");

  await analysePasted(popup, PHISHING_SMS);
  await popup.getByText("History (1)").click();
  await expect(popup.locator(".history")).toContainText("dhl-update.top/confirm?…");
  expect(JSON.stringify(await popup.evaluate(() => chrome.storage.local.get(null)))).not.toContain("id=829");

  // Reporting a wrong result is stored locally and visible in options.
  await popup.getByRole("button", { name: "This result is wrong" }).click();
  await expect(popup.getByText("saved on this device only")).toBeVisible();
  await options.reload();
  await expect(options.getByText("1 result you reported as wrong")).toBeVisible();

  // Turning history off deletes it.
  await options.getByLabel("Keep history for").selectOption("0");
  await popup.reload();
  await expect(popup.getByText("History (0)")).toBeVisible();
});

test("pasted markup is never rendered as HTML", async ({ popup }) => {
  let dialogs = 0;
  popup.on("dialog", (dialog) => {
    dialogs++;
    void dialog.dismiss();
  });
  await finishOnboarding(popup);
  await analysePasted(popup, 'http://evil-login.top/x"><img src=x onerror=alert(1) id=injected><script>alert(2)</script>');
  await expect(popup.locator("#injected")).toHaveCount(0);
  await expect(popup.locator(".result img, .result script")).toHaveCount(0);
  // The markup is shown back as plain text in the message section.
  await expect(popup.locator("[data-part=message] .address")).toContainText("<script>alert(2)</script>");
  expect(dialogs).toBe(0);
});

test("the popup is usable with the keyboard alone", async ({ popup }) => {
  await expect(popup.getByRole("button", { name: "Got it" })).toBeFocused();
  await popup.keyboard.press("Enter");
  await expect(popup.getByText("Nothing analysed yet")).toBeVisible();
  await popup.getByLabel("Or paste a link or a message").focus();
  await popup.keyboard.type("http://secure-login.account-verify.top/");
  await popup.keyboard.press("Tab");
  await popup.keyboard.press("Enter");
  await expect(popup.getByRole("heading", { name: "Likely phishing", level: 2 })).toBeVisible();
});

test("privacy: analysing sends nothing off the device and touches no other tab", async ({ popup, context, outsideRequests }) => {
  // Another tab is open on an ordinary page while the popup is used.
  const other = await context.newPage();
  const session = await context.newCDPSession(other);
  const isolatedWorlds: string[] = [];
  session.on("Runtime.executionContextCreated", ({ context: created }) => {
    // An extension script injected into a page runs in an isolated world. The
    // test driver creates one of its own, which is not the extension's doing.
    if (created.auxData?.type === "isolated" && !created.name.startsWith("__playwright")) {
      isolatedWorlds.push(created.name);
    }
  });
  await session.send("Runtime.enable");
  await other.setContent("<title>Bank</title><h1>Sign in</h1><input type=password>");

  await finishOnboarding(popup);
  await analysePasted(popup, PHISHING_SMS);
  await analysePasted(popup, "https://accounts.google.com/signin");

  expect(outsideRequests).toEqual([]);
  // No extension script was injected into the other tab: nothing read it.
  expect(isolatedWorlds).toEqual([]);
});

test("the model card opens from inside the extension", async ({ context, extensionId }) => {
  const card = await context.newPage();
  await card.goto(`chrome-extension://${extensionId}/models/model_card.html`);
  await expect(card.getByRole("heading", { name: "PhishGuard URL model" })).toBeVisible();
  await expect(card.getByText(metadata.model_version).first()).toBeVisible();
});
