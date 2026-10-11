// The only code that ever runs inside a web page, and only when the user
// clicks Analyse: it is injected on demand by lib/page.ts and is not
// registered in the manifest, so it cannot run on its own.
//
// It must stay self-contained (no imports, no outer variables): the browser
// serialises this function and runs it in the page.
export function readPageAddress(): string {
  return location.href;
}
