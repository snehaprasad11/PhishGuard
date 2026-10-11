# Store listing draft

Text for a Chrome Web Store or Edge Add-ons listing. Nothing has been submitted. Publishing
needs a developer account (Chrome charges a one-time fee) and a final review of this text.

## Name

PhishGuard

## Short description (132 characters maximum)

Checks links and messages for signs of phishing, on your device. Nothing is scanned until you
click Analyse. Nothing is uploaded.

## Detailed description

PhishGuard gives you a second opinion on a suspicious link or message, without sending it
anywhere.

Open a page and click Analyse, or paste a link, a text message or an email. PhishGuard shows:

- a risk score and how confident it is;
- the parts of the address or message that look suspicious, highlighted;
- the reasons behind the verdict, in plain language;
- what to do next.

How it is different:

- It runs on your device. The detection models are inside the extension. There is no server
  and no account.
- It does nothing until you ask. It reads a page's address only when you click Analyse, and
  never reads the page's content.
- It explains itself. The reasons shown are the model's actual calculation, not a summary.
- It is honest about its limits. It never says a page is "safe", and its measured accuracy,
  including where it is weak, is published in a model card you can open from the extension.

PhishGuard is an aid to your judgement, not a guarantee. It will miss some phishing and will
sometimes be wrong about a legitimate site. Always check before entering a password.

## Category

Productivity (Chrome) / Security (Edge)

## Permission justifications

| Permission | Justification |
| --- | --- |
| `activeTab` | To read the address of the tab you are viewing, only after you open the extension and click Analyse. |
| `scripting` | To run one function in that tab that returns its address. No script is injected automatically and page content is not read. |
| `storage` | To keep your history, your "this result is wrong" reports and your settings on your device. Nothing is synced. |

Host permissions: none. Remote code: none.

## Privacy disclosure

- **Data collected:** none. PhishGuard makes no network requests.
- **Data stored locally:** for each analysis, a hash of the address, the verdict and the time.
  The address itself is stored only if you turn that on, and never with its query string. The
  text of a pasted message is never stored.
- **Data sold or shared:** none.

## Screenshots

In `docs/screenshots/`, regenerated from the real interface with `pnpm screenshots`:

1. `phishing-message.png` — a pasted scam message, with the link and the wording explained.
2. `scam-wording-real-link.png` — a scam message whose link is genuine.
3. `no-warning-signs.png` — a well-known sign-in page.
4. `options.png` — history and privacy settings.
5. `phishing-message-dark.png` — dark theme.

## Before submitting

- [ ] Load the unpacked build and click "Analyse this page" on several real pages.
- [ ] Replace the generated icon with a designed one at 16, 48 and 128 pixels.
- [ ] Record the demo video (outline in `docs/PORTFOLIO.md`).
- [ ] Publish the model card somewhere public and link it from the listing.
- [ ] Re-read the permission justifications against the manifest.
