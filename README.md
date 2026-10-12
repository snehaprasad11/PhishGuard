# PhishGuard

A browser extension that checks a link or a message for signs of phishing, using
machine-learning models that run entirely inside your browser.

**Nothing is scanned until you click Analyse, and nothing leaves your device.** There is no
server and no account. The models are packaged inside the extension.

| A scam message | A real sign-in page | Scam wording, genuine link |
| --- | --- | --- |
| ![A pasted scam message flagged as likely phishing, with the link and the wording each explained](docs/screenshots/phishing-message.png) | ![A well-known sign-in page with no warning signs found](docs/screenshots/no-warning-signs.png) | ![A message whose link is genuine but whose wording is flagged](docs/screenshots/scam-wording-real-link.png) |

## What it does

- **On a web page:** click **Analyse this page**. PhishGuard reads the page's address (not its
  content) and scores it.
- **For a message:** paste a suspicious link, SMS or email. Its link is scored by a URL model
  and its wording by a text model, and you see both.

Either way you get a risk score, how confident the model is, the suspicious parts highlighted,
the reasons in plain language, and what to do next. The reasons are the model's actual
calculation, not a summary of it. PhishGuard never calls anything "safe": a clean result means
no warning signs were found.

It keeps a local history (a hash of each address by default, never the text of a message) and
lets you mark a result as wrong. Both stay on your device and can be cleared at any time.

## How well it works

Every number is produced by the training pipeline or the benchmark, on data the models never
saw during training. The full results, including the weak spots, are in the
[live model card](https://snehaprasad11.github.io/PhishGuard/).

| | URL model | Text model |
| --- | --- | --- |
| What it is | logistic regression, 33 address features, 1.1 KB | logistic regression, 13 wording signals, 0.7 KB |
| Phishing caught, held-out test | 65.7% | 36.8% |
| Legitimate wrongly flagged, held-out test | 1.01% | 0.55% |
| Well-known legitimate examples wrongly flagged | 0 of 68 pages | 0 of 20 messages |

In the browser a verdict takes about 6 ms from click to screen ([details](docs/PERFORMANCE.md)).

Read these as what they are: the URL model misses about one phishing link in three, and the
text model is a second opinion on wording, not a filter. PhishGuard is an aid to judgement, not
a guarantee.

### Two decisions the numbers drove

- **A more accurate URL model was not shipped.** Gradient boosting catches more, but only a
  linear model can show the user exact reasons, so the linear model is used and the gap is
  reported in the model card.
- **A text model with 96% test recall was rejected.** It flagged 11 of 20 ordinary modern
  messages such as delivery updates and one-time codes. The public "legitimate" email it
  learned from is twenty years old, so it treated modern wording as suspicious, and the test
  data shared that bias. The shipped model flags none of the 20.

## How it works

```
training/  (Python, offline)                       extension/  (TypeScript, in your browser)
  pinned data -> features -> train -> calibrate      popup -> Web Worker (ONNX Runtime Web)
              -> export to ONNX -> evaluate                   reads only the packaged models
                              |                                         ^
                              +---------- models/*.onnx, *.json --------+
```

- **Training** builds splits with no leakage (no site or message template appears in two
  splits, and phishing is split by date), calibrates the probabilities, picks the threshold for
  a 1% false-positive rate, and exports to ONNX. The export is refused unless ONNX Runtime and
  scikit-learn agree.
- **The extension** recomputes the same features in TypeScript, checked against values exported
  from Python to 12 decimal places, and runs the models in a Web Worker.
- **Releases** pass a gate that rejects a model worse than the accepted one, are stored in a
  registry with one-command rollback, and are watched by a weekly drift check.
- **The privacy claim is tested:** a browser test records every network request during an
  analysis and fails if there is one.

## Install

Download `phishguard-extension-0.3.0.zip` from the
[latest release](https://github.com/snehaprasad11/PhishGuard/releases/latest) and unzip it. Open
`chrome://extensions` or `edge://extensions`, turn on **Developer mode**, click **Load unpacked**
and choose the unzipped folder. Step-by-step instructions are in [docs/BETA.md](docs/BETA.md).

## Build and test

**Extension** (Node 20+ and pnpm, in `extension/`):

```bash
pnpm install
pnpm build        # output in extension/dist
pnpm test         # unit tests, including parity with Python and the privacy audit
pnpm test:e2e     # the built extension in a real browser
```

**Training pipeline** (Python 3.12, from the repository root):

```bash
pip install -r training/requirements.lock
pytest
python -m training.gate   # checks the committed models against the accepted baseline
```

Retraining needs the raw data, which is not in this repository (see below). To fetch your own
copy and retrain:

```bash
python -m training.data.fetch phishtank openphish tranco commoncrawl sms_spam nazario spamassassin
python -m training.pipeline
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md): the pipeline, the features and why each design choice
  was made.
- [Security and privacy](docs/SECURITY.md): permissions, what is stored, and how each claim is
  tested.
- [Releases, rollback and drift](docs/MLOPS.md)
- [Changelog](CHANGELOG.md)

## Limitations

- The URL model sees only the address, so it cannot recognise a fake login form hosted on a
  legitimate platform.
- The text model knows only English phrases.
- Page content is never analysed: the text model sees only what you paste.
- Phishing changes constantly, so accuracy falls between retrains.

## Data

Trained on [PhishTank](https://phishtank.org/) phishing reports, benign addresses from
[Common Crawl](https://commoncrawl.org/) ranked with the [Tranco list](https://tranco-list.eu/),
Jose Nazario's phishing email corpus (CC-BY-4.0), the SpamAssassin public corpus and the UCI SMS
Spam Collection. [OpenPhish](https://openphish.com/) is used only as an independent check.

The raw data is not redistributed here, because several of these sources do not permit it.
Only hashes of the files used are committed, in `training/data/manifest.json`.

## License

MIT
