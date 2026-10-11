# Portfolio notes

Material for a resume, a LinkedIn post and interviews. Every number comes from a file in this
repository, named next to it. Re-check them after any retrain: the model card and
`models/metrics.json` are the source of truth.

## Resume bullets

- Built a privacy-preserving phishing-detection browser extension running two calibrated ML
  models **entirely in-browser** via ONNX Runtime Web, at about **6 ms** median click-to-verdict
  latency and under **2 KB** of model files. *(`docs/PERFORMANCE.md`, `models/metrics.json`)*
- Designed a leakage-safe, time-based training pipeline (scikit-learn → ONNX) with probability
  calibration and threshold selection at a **1% target false-positive rate**, reaching **65.5%
  recall** on held-out sites the model had never seen. *(`models/metrics.json`)*
- Shipped the MLOps around it: an MLflow registry with champion/challenger rollback, CI metric
  gates that block regressions, shadow comparison, and a scheduled drift report.
- Rejected a text model with **96% test recall** after showing it flagged **11 of 20** ordinary
  messages, tracing the cause to dataset bias, and shipped an interpretable alternative that
  flags **0 of 20**. *(model card)*

## Short description

PhishGuard is a Manifest V3 browser extension that flags phishing links and messages with
on-device ML. You click Analyse and get a calibrated risk score plus the exact factors behind
it, with nothing leaving your device. It is backed by a reproducible scikit-learn → ONNX
pipeline with calibration, CI metric gates and registry rollback.

## Repository description

On-device phishing detection browser extension (MV3 + ONNX Runtime Web) with a reproducible
scikit-learn → ONNX training pipeline, calibration, CI metric gates and model-registry rollback.

## Keywords

Machine Learning, ONNX, ONNX Runtime Web, scikit-learn, model calibration, PR-AUC,
false-positive rate, feature engineering, data leakage, time-based split, MLflow, model
registry, drift detection, MLOps, CI/CD, browser extension, Manifest V3, TypeScript,
privacy-preserving ML.

## Stories worth telling in an interview

**"Tell me about a time a metric misled you."** The bag-of-words text model scored 96% recall
at 1% false positives on its held-out split. A list of 20 ordinary messages I wrote by hand
(one-time codes, delivery updates) showed it flagged 11 of them. The legitimate email in the
public corpora is from early-2000s mailing lists, so "your account" or a price looked like a
scam, and the test split shared that bias. I replaced it with 13 wording signals, which score
lower on the test split and flag none of the 20.

**"How did you prevent leakage?"** Every URL of a site is in exactly one split, with shared
hosts like `a.web.app` counted as separate sites. Phishing is split by submission date, and a
split never contains phishing from after its cut-off. For messages, near-identical templates
are collapsed to one so a template cannot sit in two splits. A check runs inside every pipeline
run and fails it if either rule is broken.

**"Why logistic regression when gradient boosting scores higher?"** The evidence panel shows
the user why a URL was flagged. With a linear model each factor's contribution is exact and
they add up to the score. I measured gradient boosting, reported the gap in the model card, and
recovered part of it by giving host features a second weight on bare-host URLs, which keeps the
model linear.

**"How do you keep Python and the browser in agreement?"** The export writes a fixture of
inputs with the features and probabilities Python computed. The TypeScript featurisers must
reproduce the features to 12 decimal places and the ONNX models must reproduce the
probabilities, in unit tests. At load time the extension refuses a model whose feature list or
file hash does not match the build. That check caught a real bug: a re-export had dropped the
decision threshold from the metadata.

**"How do you know the privacy claim is true?"** A browser test records every network request
while the extension analyses, and another tab is watched for injected scripts; both must stay
empty. A source audit fails the build if anything but the worker calls `fetch`, or if the page
is read anywhere but the Analyse click handler.

**"How do you monitor a model with no server?"** From the outside. A weekly job compares the
newest public phishing reports with a reference saved at training time: PSI per feature, PSI of
the model's score, and the share the shipped model still catches. An alert opens an issue.

**"What would you do with more time?"** Find modern legitimate email to train the text model
on, since that is its real limit; collect opt-in feedback from beta users; and analyse page
content for phishing hosted on trusted platforms, which no URL model can see.

## What to be candid about

- The URL model misses about one phishing URL in three at its operating point.
- The text model is a weak second opinion, not a filter.
- The hand-written check lists are small, so their counts are indicative.
- Nothing here has been validated by real users yet.

## Demo outline (3 to 5 minutes)

1. **Problem (30 s).** Show a phishing SMS. Blocklists miss new campaigns and need your URL.
2. **Flow (60 s).** Paste it, click Analyse: verdict, score, highlighted link and wording.
3. **Evidence (45 s).** Open "All factors": the bars are the model's own terms and add up.
4. **Architecture (45 s).** The diagram in the README: no server, models packaged, CI offline.
5. **Hard problems (45 s).** Leakage-safe splits, the rejected text model, the gate failing a
   deliberately worse model.
6. **Results (30 s).** The model card: held-out numbers, and the limitations section.
7. **Next (15 s).** Better legitimate email data, opt-in feedback, page content.
