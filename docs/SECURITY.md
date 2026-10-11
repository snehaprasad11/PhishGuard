# Security and privacy

PhishGuard's promise is: **nothing is read until you click Analyse, and nothing leaves your
device.** This page says how that is enforced and how it is checked.

## Permissions

The extension requests three permissions and no host permissions.

| Permission | Why it is needed | What it does not allow |
| --- | --- | --- |
| `activeTab` | Lets the extension act on the tab you are looking at, only after you open its popup | Access to any tab you have not invoked it on |
| `scripting` | Runs one function in that tab, on your click, to read its address | Running anything automatically; no content script is registered |
| `storage` | Keeps history, feedback and settings in `chrome.storage.local` | Syncing to an account; `storage.sync` is never used |

There is no background service worker, no registered content script, no
`web_accessible_resources` and no `externally_connectable`. Nothing can run unless you open
the popup.

## What happens when you click Analyse

1. The popup injects a one-line function into the active tab that returns `location.href`.
2. The address is sent to a Web Worker inside the extension.
3. The worker computes 33 numbers from the address, runs the ONNX model, and returns a
   verdict with its reasons.

Only the address is read. Page content, form fields and cookies are never touched.

When you paste text instead, the first web address in it goes to the URL model, and if the
text is more than a bare link its wording goes to the text model. Both run in the same
worker, on your device.

## No network

Both models (`url-model.onnx`, `text-model.onnx`), their metadata and the known-sites list are
packaged inside the extension. The worker's only `fetch` calls are for those packaged files. The content
security policy restricts scripts and connections to the extension itself:

```
default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline';
connect-src 'self'; object-src 'none'
```

`'wasm-unsafe-eval'` is required to run ONNX Runtime's WebAssembly. It permits compiling
WebAssembly, not evaluating JavaScript strings. There is no remote code.

## Stored data

| Data | Default | With "Show addresses in history" on |
| --- | --- | --- |
| History entry | time, verdict, score, model version, **SHA-256 of the address** | also the address **without its query string** |
| Feedback entry ("This result is wrong") | address without its query string, verdict, your label | same |
| A pasted message | **never stored**: only a hash, the verdict and the score | same |
| Settings | retention period, the option above, onboarding seen | same |

- Query strings are removed because that is where tokens and personal data usually sit.
- History is kept for 30 days by default; you can choose 7 days or none. Turning it off
  deletes what is stored.
- History and feedback can each be cleared with one click.
- The extension never logs: there are no `console` calls in its source.

A hash of an address is not anonymous against someone who can guess the address, so treat
the default as "not readable at a glance", not as encryption.

## Untrusted input

Addresses and pasted text are untrusted. They are rendered only as text through React; the
source contains no `innerHTML`, `dangerouslySetInnerHTML` or `document.write`. The model's
metadata is checked against the build before use, and the model file's SHA-256 is checked
against its metadata, so a mismatched or corrupted model is refused instead of run.

## How this is checked

| Claim | Check |
| --- | --- |
| Permissions are minimal; nothing runs on its own | `extension/tests/privacy.test.ts` (manifest audit) |
| The page is read in one place, only from the Analyse click handler | `privacy.test.ts` (source audit) |
| Requests go only to packaged files; no logging; no raw HTML | `privacy.test.ts` (source audit) |
| Analysing sends nothing off the device | `extension/e2e/extension.e2e.ts` records every request in a real browser |
| No other tab is touched | the same test watches another tab for injected scripts |
| The wording of a pasted message is never stored | `storage.test.ts` and an end-to-end test |
| No address in storage by default; no query string ever | `extension/tests/storage.test.ts` and the end-to-end tests |
| Pasted markup is never rendered | end-to-end test |

One path is not covered by an automated browser test: clicking **Analyse this page** from the
real toolbar popup, because test tools cannot open a toolbar popup with `activeTab` granted.
That path is covered by unit tests with a mocked browser API and should be checked by hand
before a release.

## Threats this does not solve

- **Evasion.** Anyone can read the feature list and craft an address that scores low.
  PhishGuard is an aid to judgement, not a guarantee, and the interface never says "safe".
- **Phishing on trusted platforms.** A fake login form hosted on a well-known service has an
  address that looks legitimate. See the model card for measured recall on those.
- **A compromised device.** Local storage is readable by anything that can read your browser
  profile.

## Reporting a vulnerability

Please open a private security advisory on the repository rather than a public issue.
