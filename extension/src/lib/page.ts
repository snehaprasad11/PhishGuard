import { readPageAddress } from "../content/content-script";

export const inExtension = (): boolean =>
  typeof chrome !== "undefined" && chrome.scripting !== undefined && chrome.tabs !== undefined;

export class PageAccessError extends Error {}

/**
 * Address of the tab the user is looking at.
 *
 * Call this only from the Analyse click handler. Opening the popup grants
 * `activeTab` for the current tab, but nothing is read until this runs — that
 * is the "never scans until you click Analyse" promise.
 */
export async function readActivePageAddress(): Promise<string> {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab?.id === undefined) throw new PageAccessError("no active tab");
  try {
    const [injection] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: readPageAddress,
    });
    if (typeof injection?.result !== "string") throw new Error("no result");
    return injection.result;
  } catch {
    // Browser pages (chrome://, the extension store, a blank tab) refuse injection.
    throw new PageAccessError("this page cannot be analysed");
  }
}
