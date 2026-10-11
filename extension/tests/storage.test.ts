import { beforeEach, describe, expect, it } from "vitest";
import { type AnalyseResponse, type PartResult, firstUrlIn, validateRequest } from "../src/lib/messages";
import { LocalStore, memoryArea, redactQuery } from "../src/lib/storage";

const DAY = 24 * 60 * 60 * 1000;

function part(overrides: Partial<PartResult> = {}): PartResult {
  return {
    displayText: "evil-login.shop/reset?token=SECRET123",
    score: 91,
    probability: 0.91,
    verdict: "phishing",
    confidence: "high",
    modelVersion: "url-test",
    evidence: [],
    highlights: [],
    ...overrides,
  };
}

/** A result for an address; `score` and `verdict` overrides apply to the whole result. */
function result(overrides: Partial<Pick<AnalyseResponse, "score" | "verdict">> = {}): AnalyseResponse {
  const address = part(overrides);
  return {
    type: "ANALYSE_RESULT",
    verdict: address.verdict,
    score: address.score,
    address,
    timing: { loadMs: 0, analyseMs: 1 },
  };
}

const MESSAGE_TEXT = "Dear customer, verify your PIN 4821 now";

function messageOnly(): AnalyseResponse {
  const message = part({ displayText: MESSAGE_TEXT, modelVersion: "text-test" });
  return { type: "ANALYSE_RESULT", verdict: "phishing", score: 91, message, timing: { loadMs: 0, analyseMs: 1 } };
}

describe("LocalStore", () => {
  let now: number;
  let area: ReturnType<typeof memoryArea>;
  let store: LocalStore;
  const raw = async () => JSON.stringify(await area.get(["settings", "history", "feedback"]));

  beforeEach(() => {
    now = 1_800_000_000_000;
    area = memoryArea();
    store = new LocalStore(area, () => now);
  });

  it("stores a hash and no address by default", async () => {
    await store.addHistory(result(), "page");
    const [entry] = await store.listHistory();
    expect(entry!.urlHash).toMatch(/^[0-9a-f]{64}$/);
    expect(entry!.address).toBeUndefined();
    expect(await raw()).not.toContain("evil-login");
    expect(await raw()).not.toContain("SECRET123");
  });

  it("keeps the address without its query string only when the user opts in", async () => {
    await store.saveSettings({ keepAddresses: true });
    await store.addHistory(result(), "pasted");
    expect((await store.listHistory())[0]!.address).toBe("evil-login.shop/reset?…");
    expect(await raw()).not.toContain("SECRET123");

    await store.saveSettings({ keepAddresses: false });
    expect((await store.listHistory())[0]!.address).toBeUndefined();
    expect(await raw()).not.toContain("evil-login");
  });

  it("lists newest first and drops entries older than the retention period", async () => {
    await store.addHistory(result({ score: 1 }), "page");
    now += 5 * DAY;
    await store.addHistory(result({ score: 2 }), "page");
    expect((await store.listHistory()).map((e) => e.score)).toEqual([2, 1]);

    await store.saveSettings({ retentionDays: 7 });
    now += 3 * DAY; // the first entry is now 8 days old
    expect((await store.listHistory()).map((e) => e.score)).toEqual([2]);
    expect(await raw()).not.toContain('"score":1');
  });

  it("stores nothing, and deletes what exists, when history is turned off", async () => {
    await store.addHistory(result(), "page");
    await store.saveSettings({ retentionDays: 0 });
    expect(await store.listHistory()).toEqual([]);
    await store.addHistory(result(), "page");
    expect(await store.listHistory()).toEqual([]);
  });

  it("caps history at 200 entries", async () => {
    for (let i = 0; i < 205; i++) await store.addHistory(result({ score: i % 100 }), "page");
    expect(await store.listHistory()).toHaveLength(200);
  });

  it("clears history and feedback separately", async () => {
    await store.addHistory(result(), "page");
    await store.addFeedback(result());
    await store.clearHistory();
    expect(await store.stats()).toEqual({ analyses: 0, markedWrong: 1 });
    await store.clearFeedback();
    expect(await store.stats()).toEqual({ analyses: 0, markedWrong: 0 });
  });

  it("records feedback as the opposite of the verdict, without the query string", async () => {
    const flagged = await store.addFeedback(result({ verdict: "phishing" }));
    const missed = await store.addFeedback(result({ verdict: "clear" }));
    expect(flagged.userLabel).toBe("safe");
    expect(missed.userLabel).toBe("phishing");
    expect(flagged.address).toBe("evil-login.shop/reset?…");
    expect(await raw()).not.toContain("SECRET123");
  });

  it("never stores the wording of a pasted message", async () => {
    await store.saveSettings({ keepAddresses: true });
    await store.addHistory(messageOnly(), "pasted");
    const feedback = await store.addFeedback(messageOnly());
    const [entry] = await store.listHistory();
    expect(entry!.address).toBeUndefined();
    expect(entry!.urlHash).toMatch(/^[0-9a-f]{64}$/);
    expect(entry!.modelVersion).toBe("text-test");
    expect(entry!.subject).toBe("message");
    expect(feedback.address).toBeUndefined();
    expect(await raw()).not.toContain("customer");
    expect(await raw()).not.toContain("4821");
  });

  it("records both model versions when both models ran", async () => {
    const both = { ...result(), message: part({ displayText: MESSAGE_TEXT, modelVersion: "text-test" }) };
    await store.addHistory(both, "pasted");
    expect((await store.listHistory())[0]!.modelVersion).toBe("url-test + text-test");
    expect(await raw()).not.toContain("customer");
  });

  it("remembers that onboarding was completed", async () => {
    expect((await store.getSettings()).onboarded).toBe(false);
    await store.saveSettings({ onboarded: true });
    expect(await new LocalStore(area).getSettings()).toMatchObject({ onboarded: true, retentionDays: 30 });
  });
});

describe("redactQuery", () => {
  it.each([
    ["example.com/a?token=1", "example.com/a?…"],
    ["example.com/a", "example.com/a"],
    ["example.com/?a=1?b=2", "example.com/?…"],
  ])("%s -> %s", (input, expected) => expect(redactQuery(input)).toBe(expected));
});

describe("validateRequest", () => {
  it("accepts a well-formed request", () => {
    expect(validateRequest({ type: "ANALYSE", url: "https://example.com" })).toBeNull();
    expect(validateRequest({ type: "ANALYSE", text: "hello there friend" })).toBeNull();
    expect(validateRequest({ type: "ANALYSE", url: "example.com", text: "see example.com now" })).toBeNull();
  });

  it.each([
    [null, "object"],
    ["https://example.com", "object"],
    [{ type: "OTHER", url: "x" }, "unknown request type"],
    [{ type: "ANALYSE" }, "url or text"],
    [{ type: "ANALYSE", url: "   " }, "url must be a non-empty string"],
    [{ type: "ANALYSE", url: 42 }, "url must be a non-empty string"],
    [{ type: "ANALYSE", url: "a".repeat(8193) }, "url is longer than"],
    [{ type: "ANALYSE", text: "" }, "text must be a non-empty string"],
    [{ type: "ANALYSE", url: "example.com", text: "a".repeat(20001) }, "text is longer than"],
  ])("rejects %j", (message, reason) => {
    expect(validateRequest(message)).toContain(reason);
  });
});

describe("firstUrlIn", () => {
  it.each([
    ["Your parcel is held. Pay here: https://dhl-redelivery.top/pay?id=4. Thanks", "https://dhl-redelivery.top/pay?id=4"],
    ["visit www.example.com/login now", "www.example.com/login"],
    ["go to secure-bank.example.shop/verify!", "secure-bank.example.shop/verify"],
    ["bare domain evil-site.top please", "evil-site.top"],
    ["https://example.com", "https://example.com"],
    ["(see https://example.com/a).", "https://example.com/a"],
    ["Not today.Reply STOP to opt out. Visit evil-site.top now", "evil-site.top"],
  ])("%s", (text, expected) => expect(firstUrlIn(text)).toBe(expected));

  it("returns null when there is no address", () => {
    expect(firstUrlIn("call me at 5pm about the meeting")).toBeNull();
    // A missing space after a full stop is not a web address.
    expect(firstUrlIn("See you today.Reply when you can. Thanks.Bye")).toBeNull();
    expect(firstUrlIn("version 2.5 of the report.pdf is attached")).toBeNull();
  });
});
