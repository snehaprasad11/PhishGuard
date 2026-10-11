// Local-only persistence: history, feedback and settings live in
// chrome.storage.local on this device and are never synced or uploaded.
//
// Privacy defaults (spec §10, §17): a history entry stores a hash of the
// address, not the address. The address itself is kept only if the user turns
// that on, and even then without its query string, which is where tokens and
// personal data usually sit.

import { type AnalyseResponse, type Verdict, decidingPart } from "./messages";
import { sha256Hex } from "./model";

export interface StorageArea {
  get(keys: string[]): Promise<Record<string, unknown>>;
  set(items: Record<string, unknown>): Promise<void>;
  remove(keys: string[]): Promise<void>;
}

export interface Settings {
  /** 0 turns history off. */
  retentionDays: 0 | 7 | 30;
  keepAddresses: boolean;
  onboarded: boolean;
}

export const DEFAULT_SETTINGS: Settings = { retentionDays: 30, keepAddresses: false, onboarded: false };

export type InputType = "page" | "pasted";

export interface HistoryEntry {
  id: string;
  ts: number;
  inputType: InputType;
  /** What was analysed: an address (possibly with a message), or a message with no link. */
  subject: "address" | "message";
  urlHash: string;
  /** Present only when the user chose to keep addresses; never includes the query string. */
  address?: string;
  score: number;
  probability: number;
  verdict: Verdict;
  modelVersion: string;
}

export interface FeedbackEntry {
  id: string;
  ts: number;
  /**
   * Stored because the user explicitly reported this result; query string removed.
   * Absent when a message without a link was reported: message text is never stored.
   */
  address?: string;
  probability: number;
  verdict: Verdict;
  modelVersion: string;
  /** What the user says the page really is. */
  userLabel: "safe" | "phishing";
}

const MAX_HISTORY = 200;
const DAY_MS = 24 * 60 * 60 * 1000;
const KEYS = { settings: "settings", history: "history", feedback: "feedback" } as const;

/** `example.com/reset?token=abc` -> `example.com/reset?…` */
export function redactQuery(normalisedUrl: string): string {
  const q = normalisedUrl.indexOf("?");
  return q < 0 ? normalisedUrl : normalisedUrl.slice(0, q) + "?…";
}

/** What identifies an analysis: its address if it had one, otherwise the message wording. */
const subjectOf = (result: AnalyseResponse): string =>
  (result.address ?? result.message)!.displayText;

const modelVersions = (result: AnalyseResponse): string =>
  [result.address, result.message].flatMap((part) => (part ? [part.modelVersion] : [])).join(" + ");

export class LocalStore {
  constructor(
    private readonly area: StorageArea,
    private readonly now: () => number = Date.now,
  ) {}

  private async read<T>(key: string, fallback: T): Promise<T> {
    const value = (await this.area.get([key]))[key];
    return value === undefined ? fallback : (value as T);
  }

  async getSettings(): Promise<Settings> {
    return { ...DEFAULT_SETTINGS, ...(await this.read<Partial<Settings>>(KEYS.settings, {})) };
  }

  async saveSettings(changes: Partial<Settings>): Promise<Settings> {
    const settings = { ...(await this.getSettings()), ...changes };
    await this.area.set({ [KEYS.settings]: settings });
    if (settings.retentionDays === 0) await this.clearHistory();
    if (!settings.keepAddresses) await this.dropStoredAddresses();
    return settings;
  }

  /** History within the retention window, newest first. Expired entries are deleted. */
  async listHistory(): Promise<HistoryEntry[]> {
    const { retentionDays } = await this.getSettings();
    const all = await this.read<HistoryEntry[]>(KEYS.history, []);
    const cutoff = this.now() - retentionDays * DAY_MS;
    const kept = all.filter((entry) => entry.ts >= cutoff).slice(0, MAX_HISTORY);
    if (kept.length !== all.length) await this.area.set({ [KEYS.history]: kept });
    return kept;
  }

  async addHistory(result: AnalyseResponse, inputType: InputType): Promise<void> {
    const settings = await this.getSettings();
    if (settings.retentionDays === 0) return;
    const entry: HistoryEntry = {
      id: crypto.randomUUID(),
      ts: this.now(),
      inputType,
      subject: result.address ? "address" : "message",
      urlHash: await sha256Hex(subjectOf(result)),
      score: result.score,
      probability: decidingPart(result).probability,
      verdict: result.verdict,
      modelVersion: modelVersions(result),
    };
    // Only ever an address. The wording of a pasted message is never stored.
    if (settings.keepAddresses && result.address) {
      entry.address = redactQuery(result.address.displayText);
    }
    await this.area.set({ [KEYS.history]: [entry, ...(await this.listHistory())].slice(0, MAX_HISTORY) });
  }

  async clearHistory(): Promise<void> {
    await this.area.remove([KEYS.history]);
  }

  private async dropStoredAddresses(): Promise<void> {
    const history = await this.read<HistoryEntry[]>(KEYS.history, []);
    if (history.some((entry) => entry.address !== undefined)) {
      await this.area.set({
        [KEYS.history]: history.map((entry) => {
          const copy = { ...entry };
          delete copy.address;
          return copy;
        }),
      });
    }
  }

  async listFeedback(): Promise<FeedbackEntry[]> {
    return this.read<FeedbackEntry[]>(KEYS.feedback, []);
  }

  /** Record that the user disagrees with a result. The label is the opposite of the verdict. */
  async addFeedback(result: AnalyseResponse): Promise<FeedbackEntry> {
    const entry: FeedbackEntry = {
      id: crypto.randomUUID(),
      ts: this.now(),
      ...(result.address && { address: redactQuery(result.address.displayText) }),
      probability: decidingPart(result).probability,
      verdict: result.verdict,
      modelVersion: modelVersions(result),
      userLabel: result.verdict === "clear" ? "phishing" : "safe",
    };
    await this.area.set({ [KEYS.feedback]: [entry, ...(await this.listFeedback())] });
    return entry;
  }

  async clearFeedback(): Promise<void> {
    await this.area.remove([KEYS.feedback]);
  }

  /** Local-only product metrics: how often results are reported wrong (spec §19). */
  async stats(): Promise<{ analyses: number; markedWrong: number }> {
    return {
      analyses: (await this.listHistory()).length,
      markedWrong: (await this.listFeedback()).length,
    };
  }
}

/** In-memory stand-in for chrome.storage.local: used in tests and outside an extension. */
export function memoryArea(): StorageArea {
  const data = new Map<string, unknown>();
  return {
    async get(keys) {
      return Object.fromEntries(keys.filter((k) => data.has(k)).map((k) => [k, structuredClone(data.get(k))]));
    },
    async set(items) {
      for (const [key, value] of Object.entries(items)) data.set(key, structuredClone(value));
    },
    async remove(keys) {
      for (const key of keys) data.delete(key);
    },
  };
}

export function defaultStore(): LocalStore {
  const inExtension = typeof chrome !== "undefined" && chrome.storage?.local !== undefined;
  return new LocalStore(inExtension ? chrome.storage.local : memoryArea());
}
