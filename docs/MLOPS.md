# Releasing, rolling back and monitoring models

How a model gets from a training run to the extension, how to undo that, and how to know when
it is time to retrain. Everything here runs on a laptop and on free CI; there is no server.

## The life of a release

```
python -m training.pipeline          train, export, evaluate  ->  models/ (candidate files)
python -m training.gate              is it at least as good as the accepted baseline?
python -m training.registry register record models/ as a version, alias "challenger"
python -m training.shadow            how would its verdicts differ from the champion's?
python -m training.registry promote  challenger -> champion (refused if the gate fails)
cd extension && pnpm build           the extension packages whatever is in models/
```

A **release** is the whole of what ships: both ONNX models, their metadata, the known-sites
list, the metrics and the model card. The registry (MLflow, in the local `mlflow.db`) stores
every release as a version of one registered model, `phishguard`.

| Alias | Meaning |
| --- | --- |
| `challenger` | the most recently registered release, awaiting a decision |
| `champion` | the release the extension should ship |
| `previous` | the champion before the last promotion: what a rollback restores |

## The gate

`models/metrics.json` describes the candidate; `models/baseline-metrics.json` describes the
champion. `python -m training.gate` fails if the candidate breaks any rule under `gates:` in
`config.yaml`: a drop in PR-AUC or recall beyond tolerance, a false-positive rate over the
limit, a model that is too large or too slow, or too many well-known legitimate pages or
ordinary messages wrongly flagged. CI runs it on every change.

Promotion is the only thing that moves the baseline, and it is a deliberate command. A gate
that updated its own baseline would approve any slow decline.

## Shadow comparison

Two models with the same recall can disagree about *which* URLs are phishing.
`python -m training.shadow` runs the champion's and the challenger's ONNX files, exactly as the
browser would, over the same held-out test split and reports:

- each model's PR-AUC, recall and false-positive rate at its own threshold;
- how many verdicts change, split into phishing newly caught, phishing newly missed, benign
  newly flagged and benign no longer flagged.

A challenger that gains recall by newly flagging a few hundred benign URLs is a different
proposition from one that simply catches more. If the two releases were built with different
feature lists the champion cannot be re-scored, and the tool says so instead of comparing
unlike things.

## Rollback

```
python -m training.registry rollback
cd extension && pnpm build
```

This points `champion` back at the previous release, copies that release's exact files back
into `models/`, and resets the baseline to its metrics. No retraining and no code change is
involved, so it takes seconds. The withdrawn release stays in the registry as `challenger` so it
can be investigated. `python -m training.registry status` shows where the aliases point.

## Drift monitoring

The extension sends nothing, so there is no stream of user traffic to watch. Drift is watched
from outside instead: `.github/workflows/drift.yml` runs `python -m training.drift --live` every
Monday, downloads the current PhishTank feed, keeps the phishing reported since the training
snapshot, and compares it with `models/drift-reference.json` (bin edges and shares saved at
training time; it contains no URLs).

It reports:

- **PSI per feature**: has the shape of phishing URLs changed?
- **PSI of the model's score**: is the model reacting differently?
- **Catch rate**: what share of the newest phishing does the shipped model flag? This is the
  closest thing to a live recall figure.

A PSI above the alert level in `config.yaml` (0.25) opens a GitHub issue with the report. PSI
under 0.1 is stable; 0.1 to 0.25 is worth watching.

PhishTank rate-limits repeated downloads. One download a week is well within that; if a run
is refused anyway, the job warns and skips instead of failing.

## When to retrain

- The drift job raises an alert, or the catch rate falls noticeably below the test-split recall
  in the model card.
- Users report legitimate sites being flagged (see "This result is wrong" in the extension).
- A new kind of campaign appears that the features cannot express, which needs new features,
  not just new data.

To retrain: fetch a fresh snapshot (`python -m training.data.fetch ...`, which re-pins the
manifest), publish the snapshot files, move the split dates in `config.yaml` forward, bump the
model version, and go through the release steps above.

## What is deliberately not here

- **Opt-in telemetry.** The specification lists it as a future option. Nothing is collected
  today, and the product works without it.
- **WebGPU.** Inference takes a fraction of a millisecond on WebAssembly for models of a few
  dozen weights. WebGPU would add a far larger runtime and a start-up cost to speed up something
  that is not slow, and the ONNX-ML operators these models use have no WebGPU implementation, so
  it would fall back to WebAssembly anyway.
- **Fetching models from a CDN and caching them.** The models are a few kilobytes and ship
  inside the extension, so there is no download to cache and no network request at all.
- **Quantisation.** Measured in the pipeline and reported in the model card; it does not apply
  to single-layer models in this format.
