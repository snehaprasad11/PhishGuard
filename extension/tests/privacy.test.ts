// The privacy promise, checked mechanically: what the extension is permitted
// to do (manifest) and what its code actually does (source audit).
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative, resolve } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { readPageAddress } from "../src/content/content-script";
import { PageAccessError, readActivePageAddress } from "../src/lib/page";

const extensionRoot = resolve(import.meta.dirname, "..");
const manifest = JSON.parse(readFileSync(join(extensionRoot, "manifest.json"), "utf-8"));

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    return statSync(path).isDirectory() ? sourceFiles(path) : /\.(ts|tsx)$/.test(name) ? [path] : [];
  });
}
const sources = sourceFiles(join(extensionRoot, "src")).map((path) => ({
  file: relative(extensionRoot, path).replaceAll("\\", "/"),
  text: readFileSync(path, "utf-8"),
}));
const filesMatching = (pattern: RegExp) => sources.filter((s) => pattern.test(s.text)).map((s) => s.file);

describe("manifest", () => {
  it("asks for the minimum permissions and nothing else", () => {
    expect(manifest.manifest_version).toBe(3);
    expect([...manifest.permissions].sort()).toEqual(["activeTab", "scripting", "storage"]);
    for (const key of ["host_permissions", "optional_permissions", "optional_host_permissions"]) {
      expect(manifest[key], key).toBeUndefined();
    }
  });

  it("has nothing that can run without the user opening the popup", () => {
    for (const key of ["content_scripts", "background", "externally_connectable", "web_accessible_resources"]) {
      expect(manifest[key], key).toBeUndefined();
    }
  });

  it("locks scripts and connections to the extension itself", () => {
    const csp: string = manifest.content_security_policy.extension_pages;
    expect(csp).toContain("default-src 'self'");
    expect(csp).toContain("connect-src 'self'");
    expect(csp).toContain("object-src 'none'");
    expect(csp).not.toMatch(/https?:|\*|unsafe-eval'(?<!wasm-unsafe-eval')/);
    expect(csp.match(/script-src[^;]*/)![0]).toBe("script-src 'self' 'wasm-unsafe-eval'");
  });
});

describe("source audit", () => {
  it("reads the page in exactly one place", () => {
    expect(filesMatching(/chrome\.scripting|chrome\.tabs\.query/)).toEqual(["src/lib/page.ts"]);
    // Page content is never read: the text model only ever sees text the user pasted.
    expect(filesMatching(/location\.href|document\.body|innerText|textContent/)).toEqual([
      "src/content/content-script.ts",
    ]);
  });

  it("only asks for the page address from the Analyse button's click handler", () => {
    const app = sources.find((s) => s.file === "src/popup/App.tsx")!.text;
    // Imported once, called once (inside analysePage), and analysePage is only wired to the click.
    expect(app.match(/readActivePageAddress/g) ?? []).toHaveLength(2);
    expect(app).toContain("const analysePage = async () => ({ url: await readActivePageAddress() });");
    expect(app.match(/analysePage/g) ?? []).toHaveLength(2);
    expect(app).toContain('onClick={() => void run(analysePage, "page")}');
  });

  it("makes network requests only from the worker, and only for packaged files", () => {
    expect(filesMatching(/\bfetch\(/)).toEqual(["src/worker/inference-worker.ts"]);
    const worker = sources.find((s) => s.file === "src/worker/inference-worker.ts")!.text;
    expect(worker).toContain("fetch(asset(path))");
    expect(worker).toContain("new URL(path, self.location.origin)");
    expect(filesMatching(/XMLHttpRequest|WebSocket|sendBeacon|EventSource|importScripts/)).toEqual([]);
  });

  it("never syncs storage to the cloud", () => {
    expect(filesMatching(/chrome\.storage\.(sync|managed)/)).toEqual([]);
  });

  it("never writes raw HTML", () => {
    expect(filesMatching(/innerHTML|outerHTML|dangerouslySetInnerHTML|insertAdjacentHTML|document\.write/)).toEqual([]);
  });

  it("never logs", () => {
    expect(filesMatching(/console\.(log|info|debug|warn|error)/)).toEqual([]);
  });
});

describe("readActivePageAddress", () => {
  afterEach(() => vi.unstubAllGlobals());

  function stubChrome(executeScript: (...args: unknown[]) => Promise<unknown>) {
    const query = vi.fn(async () => [{ id: 7 }]);
    const execute = vi.fn(executeScript);
    vi.stubGlobal("chrome", { tabs: { query }, scripting: { executeScript: execute } });
    return { query, execute };
  }

  it("injects the one content function into the active tab and returns its address", async () => {
    const { query, execute } = stubChrome(async () => [{ result: "https://example.com/a" }]);
    expect(await readActivePageAddress()).toBe("https://example.com/a");
    expect(query).toHaveBeenCalledWith({ active: true, currentWindow: true });
    expect(execute).toHaveBeenCalledWith({ target: { tabId: 7 }, func: readPageAddress });
  });

  it("turns a refused injection into a page-access error", async () => {
    stubChrome(async () => {
      throw new Error("Cannot access a chrome:// URL");
    });
    await expect(readActivePageAddress()).rejects.toBeInstanceOf(PageAccessError);
  });

  it("the injected function is self-contained", () => {
    // It is serialised and run inside the page, so it may not reference anything outside itself.
    expect(readPageAddress.toString()).toMatch(/^function readPageAddress\(\)\s*\{\s*return location\.href;?\s*\}$/);
  });
});
