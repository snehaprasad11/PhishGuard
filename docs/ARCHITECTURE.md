# Architecture

PhishGuard has two halves that share nothing except a model file:

```
training/  (Python, offline)                     extension/  (TypeScript, in the browser)
  pinned data -> features -> train -> calibrate    popup -> content script -> Web Worker
              -> benchmark -> evaluate -> export        loads the exported model, scores a URL
                                   |                              ^
                                   +------- models/*.onnx --------+
```

The training half is described first, then the extension.

## Pipeline stages

| Stage | Module | Reads | Writes |
| --- | --- | --- | --- |
| Fetch | `training/data/fetch.py` | live feeds, or a published snapshot | `training/data/raw/*`, `manifest.json` |
| Load | `training/data/loaders.py` | raw files + manifest | DataFrames (verified against the manifest) |
| Split | `training/data/split.py` | labelled URLs | train / val / test rows |
| Features | `training/features/url_features.py` | a URL + the known-sites list | 33 numbers in a fixed order |
| Train | `training/train.py` | train + val | calibrated model, `dataset.parquet` |
| Benchmark | `training/benchmark.py` | train + val | `benchmark.json` |
| Export | `training/export_onnx.py` | the calibrated model | `models/url-model.onnx`, `url-model.json`, parity fixture |
| Evaluate | `training/evaluate.py` | val + test, external checks | `eval.json`, `models/model_card.html` |

`python -m training.pipeline` runs train, benchmark, export, the text-model pipeline and
evaluate in order. The export
is refused if ONNX Runtime and scikit-learn disagree by more than 1e-4 on any sampled URL. Fetching is separate
on purpose: the pipeline only reads what the manifest pins, so it cannot silently train on
different data. Two consecutive runs produce byte-identical outputs.

## Data

| Dataset | Role | Notes |
| --- | --- | --- |
| PhishTank verified feed | phishing, trained on | full URLs with a submission timestamp |
| Common Crawl `cluster.idx` | benign, trained on | a 1-in-3000 sample of each crawl's URL index |
| Tranco top sites | benign tiers, known-sites list | ranks each benign domain's popularity |
| OpenPhish feed | external check only | independent phishing source, never trained on |
| `curated_benign.txt` | regression check | well-known sign-in, bank and government pages; guided the design |
| `curated_benign_holdout.txt` | external check only | a second such list, never used for decisions |

Decisions worth knowing:

- **Benign URLs come from a crawl, not from a top-sites list.** A top-sites list is bare domains;
  paired with full phishing URLs the model would learn "has a path means phishing".
- **Benign URLs are sampled in three popularity tiers:** domains in the Tranco top 100,000
  (up to 5 URLs each), domains ranked 100,000 to 1,000,000, and domains Tranco does not list
  (one URL each from a random sample). Without the last two the model never sees a small
  legitimate site and its false-positive rate there cannot be measured. The unlisted tier has
  no reputation filter, so a small share of it may not be benign.
- **Benign URLs are stored normalised** (lowercase, no scheme, no leading `www.`) because that is
  the form Common Crawl's index uses. The featuriser applies the same normalisation to every URL
  so the two classes are comparable.
- **A homepage row is added for every benign site**, because crawled URLs are almost all deep
  pages while much phishing is a bare host.
- **URLhaus is not used**, although the original plan listed it: it tracks malware distribution
  URLs, which is a different problem from phishing and would blur the label.
- **Live feeds cannot reproduce a pinned hash** (PhishTank changes hourly). The exact raw files
  must be published somewhere and restored with `python -m training.data.fetch --from-snapshot`.

## Splits and leakage

- A *site* is the registered domain, counting shared-hosting suffixes as separate sites
  (`a.web.app` and `b.web.app` are different sites).
- Every URL of a site is in exactly one split, so the test split contains only unseen sites.
- Phishing is split by submission date. A split never contains phishing submitted after its own
  cut-off.
- Benign-only sites are assigned by a hash of the site name: the benign crawl window is ten weeks,
  too short to split by time.
- Each site contributes at most 10 URLs per class.
- `assert_no_leakage` runs inside every pipeline run and raises if either rule is broken.

## Features

All features are pure string functions of the normalised URL, with no Public Suffix List and no
locale-dependent calls, so they can be reimplemented exactly in TypeScript. The one piece of
outside knowledge is `models/known-sites.txt`, described below. Host and path are
measured separately so a long benign-looking path cannot offset a suspicious host. Lengths and
counts are stored as `log(1 + x)`.

Each host-side feature also has a `:bare_host` column that is non-zero only when the URL is a
bare host. A bare-host URL and a URL with a path behave like different populations, and this
lets the linear model weigh the host differently in each case. The model is still linear, so a
feature's contribution is exact: the sum of its two columns. Binning the numeric features was
tried instead and made the model worse.

Two deliberate differences from the original feature list: the number of host labels is used
instead of a subdomain count (which needs the Public Suffix List), and the TLD is bucketed by a
fixed rule (legacy / country-code / other) instead of a learned risk score.

## Well-known sites

A model that sees only URL text cannot tell `accounts.google.com/signin` from an imitation:
both contain "account" and "signin". Without help, it flagged 8 of 70 well-known sign-in and
banking pages.

`models/known-sites.txt` is the Tranco top 10,000 domains minus shared-hosting platforms
(recognised as public suffixes, such as `web.app` or `weebly.com`). Lines starting with `!`
are exact hosts that are *not* trusted despite their domain, such as `!sites.google.com` or
`!bit.ly`: any host on a known domain with five or more phishing reports in the training
period. They are found from training-period data only, so the list carries nothing from the
validation or test periods. The list is used two ways:

- `on_known_site` is a feature, so the model learns how much weight reputation deserves.
- Official-sounding keywords are not counted on a known site, where `accounts.` or `/signin`
  is simply what a real sign-in page looks like.

A hard allowlist was rejected: about a quarter of recent PhishTank URLs are hosted on these
well-known domains (form builders, link shorteners), so skipping them outright would miss all
of that. Removing the untrusted hosts brings that share down to under a tenth, and those
URLs are still scored by the model rather than skipped. The list is derived deterministically from the pinned Tranco snapshot and is shipped
with the model, because the browser must compute the identical feature.

The cost is honest to state: phishing on a still-trusted host is harder to catch.

## Model, calibration and threshold

- **Model:** standardised features into a logistic regression. A gradient-boosted model scores
  higher (see the model card) but the linear model's per-feature contributions are exact, which
  is what the extension shows the user as evidence.
- **Class weighting** was measured and did not help, so it is not used.
- **Calibration:** Platt (sigmoid) scaling fitted on the validation split and folded into the
  coefficients, so the calibrated model is still one linear model. Isotonic calibration was
  measured and is not meaningfully better.
- **Threshold:** chosen on the validation split for a 1% false-positive rate, as the 90th
  percentile over site-level bootstrap resamples. That leaves a margin instead of aiming exactly
  at the target.

## Experiment tracking

MLflow with a local SQLite store (`mlflow.db`). Each stage logs a run with its parameters,
metrics, the data manifest and the config. View with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Text model

`training/text/` holds a second, independent model for the wording of a pasted message.

| Stage | Module | Notes |
| --- | --- | --- |
| Data | `training/text/data.py` | SMS Spam Collection, Nazario phishing email, SpamAssassin legitimate email |
| Signals | `training/text/lexicon.py` | phrase lists for seven kinds of wording |
| Features | `training/text/features.py` | 13 numbers; also the TF-IDF vectoriser used only for comparison |
| Train, evaluate, export | `training/text/pipeline.py` | same calibration, threshold and parity discipline as the URL model |

- **Leakage.** Messages that differ only in digits, case or spacing are one template, and
  only one copy is kept, so a template cannot sit in two splits. Phishing email is split by
  the year of its mailbox, with the newest year as the test split; the other sources carry no
  usable dates and are split by hash.
- **Why engineered signals and not a bag of words.** The legitimate email available publicly
  is mailing-list traffic from the early 2000s. A bag-of-words model learns that modern
  wording is suspicious, and the test split cannot reveal this because it has the same bias.
  Two hand-written lists of ordinary modern messages do reveal it; the second was written
  after the design was frozen and is only ever reported. Both models' results on both lists
  are in the model card.
- **Links are not a text feature.** The legitimate training email is full of footer links, so
  a model would learn that a link is reassuring. Links are judged by the URL model.
- **Scope.** It is applied only to text the user pastes, never to page content: it was
  trained on messages, and a web page is a different kind of text.
- **The lists ship with the model.** `models/text-model.json` contains the phrase lists and
  feature order, and the browser reads them from there, so only the tokeniser and the
  arithmetic are mirrored in TypeScript.

## Release gate

`python -m training.evaluate` writes `models/metrics.json`, a small summary of the model that
would ship. `python -m training.gate` compares it with `models/baseline-metrics.json`, the last
model a person accepted, using the tolerances under `gates:` in `config.yaml`. CI fails if any
gate breaks. `python -m training.gate --accept` promotes the current metrics to the baseline;
it is a separate, deliberate command so the gate can never quietly lower its own bar.

`training/tests/test_published_model.py` checks that the committed files in `models/` agree
with each other and with the current code, so CI catches a stale export even without data.

## Extension

| Part | File | Role |
| --- | --- | --- |
| Popup | `extension/src/popup/` | The interface; owns UI state |
| Page access | `extension/src/lib/page.ts`, `src/content/content-script.ts` | Reads the active tab's address, only from the Analyse click |
| Worker | `extension/src/worker/inference-worker.ts` | Loads the packaged model and runs it off the UI thread |
| Featuriser | `extension/src/lib/featurise.ts` | The TypeScript mirror of `url_features.py` |
| Evidence | `extension/src/lib/evidence.ts` | Per-feature contributions and address highlights |
| Text features and evidence | `extension/src/lib/text-features.ts`, `text-evidence.ts` | The TypeScript mirror of the text featuriser; phrase highlights |
| Storage | `extension/src/lib/storage.ts` | History, feedback and settings in `chrome.storage.local` |
| Options | `extension/src/options/` | Retention, address storage, clearing data |

**Data flow.** Popup → (on click) address of the active tab or a pasted address → worker →
features → ONNX model → probability → verdict, confidence, evidence and highlights → popup →
history. No step leaves the device.

**Feature parity** is the main risk in a design like this: if the browser computes a feature
differently from Python, the model silently scores the wrong numbers. Three checks cover it:

1. The export writes a fixture of URLs with the features and probabilities Python computed.
2. `extension/tests/featurise.test.ts` requires the TypeScript featuriser to reproduce those
   features to 12 decimal places, and `model.test.ts` requires the ONNX model in ONNX Runtime
   Web to reproduce the probabilities.
3. At load time the worker refuses a model whose feature list, keyword list or file hash does
   not match the build.

**Model delivery.** The model is packaged in the extension instead of fetched from a CDN, so
there is no network request at all and nothing to cache. A model update ships as an extension
update.

**Two parts, one verdict.** A pasted message produces an *address* part (URL model) and a
*message* part (text model); a bare link or a page produces only the address part. The overall
verdict is the more serious of the two and the score is the higher, so a scam message that
links to a genuine site is still flagged, with the link shown as fine and the wording
explained.

**Verdicts.** Above the model's threshold is "Likely phishing"; above half of it is "Be
careful"; otherwise "No warning signs found". Confidence reflects distance from the threshold
in log-odds: near the threshold a small change in the address would flip the verdict.

Releasing, rollback, shadow comparison and drift monitoring are covered in
[MLOPS.md](MLOPS.md).
