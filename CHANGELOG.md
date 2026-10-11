# Changelog

Changes to the product and to the models, newest first. Model metrics quoted here are from the
held-out test split at the time; the current numbers are always in the model card.

## Extension 0.3.0

- **Text model.** A pasted message is now judged twice: its link by the URL model and its
  wording by a new text model. The result shows both, each with its own highlights and reasons.
  A message with no link is analysed on its wording alone.
- The wording of a pasted message is never stored, in history or in feedback.
- Registry with promote and rollback, shadow comparison and a weekly drift report.

### Models

- `text-v0.1.0` — logistic regression over 13 wording signals. A bag-of-words alternative was
  measured and rejected: 96% recall on the test split, but 11 of 20 ordinary modern messages
  wrongly flagged, against 0 of 20 for the shipped model.
- Quantisation measured for both models: not applicable to single-layer models in this format.

## Extension 0.2.0

- First working extension: Analyse this page, paste a link, verdict with risk score,
  confidence, highlighted address, exact per-factor reasons and next steps.
- Local history (address hashed by default) and "This result is wrong" feedback.
- Options: history retention, whether addresses are kept, clearing stored data.
- Release gate in CI; latency benchmark.

### Models

- `url-v0.4.0` — adds less-popular and unlisted domains to the benign data, stops trusting
  hosts on well-known domains that were repeatedly used for phishing, and gives host features a
  second weight on bare-host URLs. False-positive rate measured separately for popular, less
  popular and unlisted sites.
- `url-v0.3.0` — adds a well-known-sites feature and stops counting official-sounding words on
  those sites. Well-known legitimate pages wrongly flagged fell from 8 of 70 to 1 of 70.
- `url-v0.2.0` — host and path measured separately, lengths log-scaled, a bare-host flag, and a
  threshold with a safety margin. Bare-host false positives fell from 2.5% to 1.2%.
- `url-v0.1.0` — baseline: logistic regression over 17 URL features.

## Lessons recorded along the way

- A hard allowlist of well-known sites was considered and rejected after measuring that about a
  quarter of recent phishing URLs sit on well-known domains.
- Binning numeric features to capture curves made the URL model worse; the gap to gradient
  boosting comes from feature interactions.
- Good test-split numbers did not mean a good text model. A small hand-written list of ordinary
  messages exposed what the test split could not.
