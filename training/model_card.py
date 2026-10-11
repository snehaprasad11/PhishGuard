"""Render the model card as a self-contained HTML page from an evaluation report.

Every number on the card comes from the report dict produced by evaluate.py —
nothing is typed in by hand, so the card cannot drift from the measured model.
"""

from html import escape
from typing import Any

_STYLE = """
body{font-family:system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;
     line-height:1.5;color:#1a1a1a;background:#fff}
h1{margin-bottom:0}h2{margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:.2rem}
table{border-collapse:collapse;margin:.5rem 0}
th,td{border:1px solid #ccc;padding:.3rem .7rem;text-align:right}
th:first-child,td:first-child{text-align:left}
.note{color:#555;font-size:.92rem}code{background:#f2f2f2;padding:0 .25rem}
"""


def _pct(value: float | None, missing: str = "not measured yet") -> str:
    return missing if value is None else f"{100 * value:.1f}%"


def _num(value: float | None, digits: int = 3) -> str:
    return "not measured yet" if value is None else f"{value:.{digits}f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in header)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _reliability_svg(curve: list[dict[str, Any]]) -> str:
    size, pad = 320, 40
    scale = size - 2 * pad

    def point(x: float, y: float) -> str:
        return f"{pad + x * scale:.1f},{size - pad - y * scale:.1f}"

    line = " ".join(point(b["mean_predicted"], b["observed_rate"]) for b in curve)
    dots = "".join(
        f'<circle cx="{point(b["mean_predicted"], b["observed_rate"]).split(",")[0]}" '
        f'cy="{point(b["mean_predicted"], b["observed_rate"]).split(",")[1]}" r="3.5" '
        f'fill="#1f5fbf"/>' for b in curve
    )
    return (
        f'<svg width="{size}" height="{size}" role="img" '
        f'aria-label="Reliability curve: predicted probability against observed phishing rate">'
        f'<rect x="{pad}" y="{pad}" width="{scale}" height="{scale}" fill="none" stroke="#999"/>'
        f'<polyline points="{point(0, 0)} {point(1, 1)}" stroke="#999" stroke-dasharray="4" '
        f'fill="none"/>'
        f'<polyline points="{line}" stroke="#1f5fbf" stroke-width="2" fill="none"/>{dots}'
        f'<text x="{size / 2}" y="{size - 8}" text-anchor="middle" font-size="12">'
        f'predicted probability</text>'
        f'<text x="12" y="{size / 2}" text-anchor="middle" font-size="12" '
        f'transform="rotate(-90 12 {size / 2})">observed phishing rate</text></svg>'
    )


def _external_section(external: dict[str, Any]) -> str:
    openphish = external["openphish_unseen_sites"]

    def must_not_flag(key: str, title: str, note: str) -> str:
        if key not in external:
            return ""
        check = external[key]
        flagged = _table(
            ["Wrongly flagged page", "Probability"],
            [[f["url"], _num(f["probability"])] for f in check["flagged_urls"]],
        ) if check["flagged_urls"] else ""
        return (
            f"<p><strong>{title}</strong>: {check['flagged']} of {check['total']} wrongly "
            f"flagged. <span class=\"note\">{note}</span></p>{flagged}"
        )

    return (
        "<h2>Checks outside the training distribution</h2>"
        "<p>Well-known legitimate pages: banks, sign-in pages and government services.</p>"
        + must_not_flag(
            "curated_benign_holdout", "Held-out list",
            "Never used for any design decision.",
        )
        + must_not_flag(
            "curated_benign", "Development list",
            "This list guided the model's design, so it is a regression check, not a fair test.",
        )
        +
        f"<p><strong>Independent phishing feed</strong> (OpenPhish, only sites absent from "
        f"training and validation): {_pct(openphish['recall'])} of "
        f"{openphish['total']:,} URLs caught.</p>"
    )


def _benchmark_section(benchmark: dict[str, Any]) -> str:
    model_rows = [
        [name.replace("_", " ") + (" (shipped)" if name == benchmark["shipped"] else ""),
         _num(scores["val_pr_auc"]), _pct(scores["val_recall_at_target_fpr"])]
        for name, scores in benchmark["models"].items()
    ]
    brier_rows = [[k, _num(v, 4)] for k, v in benchmark["calibration_brier"].items()]
    return (
        "<h2>Alternatives considered</h2>"
        "<p class=\"note\">Measured on the validation split. The logistic regression is "
        "shipped despite lower accuracy because its per-feature contributions are exact, "
        "which is what the evidence panel shows the user.</p>"
        + _table(["Model", "PR-AUC", "Recall at target FPR"], model_rows)
        + "<p class=\"note\">Calibration method, cross-fitted Brier score (lower is better):</p>"
        + _table(["Calibration", "Brier"], brier_rows)
    )


def _text_section(text: dict[str, Any]) -> str:
    metrics, threshold, bow = text["metrics"], text["threshold"], text["bag_of_words"]

    def curated_row(name: str, title: str) -> list[str]:
        ours, theirs = text["curated"][name], bow["curated"][name]
        return [
            title,
            f"{len(ours['legitimate_flagged'])} of {ours['legitimate_total']}",
            f"{ours['scam_caught']} of {ours['scam_total']}",
            f"{theirs['legitimate_flagged']} of {theirs['legitimate_total']}",
            f"{theirs['scam_caught']} of {theirs['scam_total']}",
        ]

    flagged = [
        [f["text"], _num(f["probability"])]
        for check in text["curated"].values() for f in check["legitimate_flagged"]
    ]
    rows = text["rows"]
    return (
        f"<h1 style=\"margin-top:3rem\">PhishGuard text model</h1>"
        f"<p class=\"note\">Version <code>{escape(text['model_version'])}</code></p>"
        "<p>A logistic regression over engineered signals in the wording of a pasted message: "
        "pressure to act, threats, requests for secrets, prizes, generic greetings. It is "
        "applied to pasted messages only, never to page content.</p>"
        "<h2>Data</h2>"
        "<p class=\"note\">SMS: UCI SMS Spam Collection. Phishing email: Nazario phishing "
        "corpus (CC-BY-4.0), split by year. Legitimate email: SpamAssassin public corpus. "
        "Near-identical messages are reduced to one, so no template is in two splits.</p>"
        + _table(
            ["Split", "SMS legitimate", "SMS scam", "Email legitimate", "Email phishing"],
            [[split, f"{r['sms_legitimate']:,}", f"{r['sms_scam']:,}",
              f"{r['email_legitimate']:,}", f"{r['email_scam']:,}"] for split, r in rows.items()],
        )
        + "<h2>Held-out test results</h2>"
        + _table(["Metric", "Value"], [
            ["PR-AUC", _num(metrics["pr_auc"])],
            ["Recall at 1% false-positive rate", _pct(metrics["recall_at_fpr_1pct"])],
            ["Brier score (lower is better)", _num(metrics["brier"], 4)],
            ["Model size", f"{metrics['model_size_kb']:.2f} KB"],
            ["Chosen threshold", _num(threshold["chosen_threshold"])],
            ["False-positive rate on test", _pct(threshold["test_fpr"])],
            ["Recall on test", _pct(threshold["test_recall"])],
            ["Precision on test", _pct(threshold["test_precision"])],
        ])
        + _table(
            ["Test slice", "Legitimate", "Scam", "False-positive rate", "Recall"],
            [[name, f"{s['legitimate']:,}", f"{s['scam']:,}", _pct(s["fpr"], "n/a"),
              _pct(s["recall"], "n/a")] for name, s in text["slices"].items()],
        )
        + "<h2>Why not a bag-of-words model</h2>"
        "<p>A TF-IDF bag-of-words model was trained on the same data for comparison. On the "
        f"test split it is far stronger: PR-AUC {_num(bow['metrics']['pr_auc'])}, recall at 1% "
        f"false positives {_pct(bow['metrics']['recall_at_fpr_1pct'])}. But the test split comes "
        "from the same corpora as the training data, and the legitimate email in them is "
        "mailing-list traffic from the early 2000s. On ordinary modern messages written for "
        "this project (one-time codes, delivery updates, bank notices) it flags many "
        "legitimate ones:</p>"
        + _table(
            ["Hand-written messages", "Shipped: legitimate flagged", "Shipped: scams caught",
             "Bag of words: legitimate flagged", "Bag of words: scams caught"],
            [curated_row("holdout", "Held-out list (never used for design)"),
             curated_row("development", "Development list (guided the design)")],
        )
        + "<p class=\"note\">Words the bag-of-words model treats as most legitimate: "
        + escape(", ".join(bow["strongest_legitimate_terms"]))
        + ". These are quirks of the corpus, not signs of a safe message.</p>"
        + (_table(["Legitimate message the shipped model flags", "Probability"], flagged)
           if flagged else "")
        + "<h2>What drives the text score</h2>"
        + _table(["Signal", "Coefficient"], [
            [name.replace("_", " "), f"{value:+.2f}"]
            for name, value in sorted(text["coefficients"].items(), key=lambda kv: -abs(kv[1]))
        ])
        + "<h2>Known limitations of the text model</h2><ul>"
        "<li>It catches well under half of the scam messages in its own test split. It is a "
        "second opinion on wording, not a filter.</li>"
        "<li>It only knows English phrases, and only the ones in its lists.</li>"
        "<li>The hand-written lists are small, so their counts are indicative, not precise.</li>"
        "<li>Its legitimate training email is two decades old.</li></ul>"
    )


def _quantisation_note(report: dict[str, Any]) -> str:
    notes = []
    for label, q in (("URL model", report.get("quantisation")),
                     ("Text model", report.get("text_model", {}).get("quantisation"))):
        if q:
            notes.append(f"<li>{label}: shipped as {escape(q['shipped'])} "
                         f"({q['original_kb']:.2f} KB). {escape(q['reason']).capitalize()}.</li>")
    return f"<h2>Quantisation</h2><ul>{''.join(notes)}</ul>" if notes else ""


def render_model_card(report: dict[str, Any]) -> str:
    metrics, threshold, rows = report["metrics"], report["threshold"], report["rows"]
    size_kb, latency = metrics["model_size_kb"], metrics["latency_ms_p50"]

    data_rows = [
        [name, entry["snapshot_date"], entry["sha256"][:16] + "…"]
        for name, entry in report.get("trained_on", {}).items()
    ]
    split_rows = [[s, f"{rows[s]['benign']:,}", f"{rows[s]['phishing']:,}"] for s in rows]
    metric_rows = [
        ["PR-AUC", _num(metrics["pr_auc"])],
        ["Recall at 1% false-positive rate", _pct(metrics["recall_at_fpr_1pct"])],
        ["Brier score (lower is better)", _num(metrics["brier"], 4)],
        ["Model size", "not measured yet" if size_kb is None else f"{size_kb:.1f} KB"],
        ["Click to verdict in the browser, p50",
         "not measured yet" if latency is None else f"{latency:.1f} ms"],
    ]
    threshold_rows = [
        ["Target false-positive rate", _pct(threshold["target_fpr"])],
        ["Chosen threshold (flag if probability is above)", _num(threshold["chosen_threshold"])],
        ["Chosen on", threshold["chosen_on"]],
        ["False-positive rate on validation", _pct(threshold["val_fpr"])],
        ["False-positive rate on test", _pct(threshold["test_fpr"])],
        ["Recall on test", _pct(threshold["test_recall"])],
        ["Precision on test", _pct(threshold["test_precision"])],
    ]
    slice_rows = [
        [name.replace("_", " "), f"{s['benign']:,}", f"{s['phishing']:,}",
         _pct(s["fpr"], "n/a"), _pct(s["recall"], "n/a")]
        for name, s in report["slices"].items()
    ]
    reliability_rows = [
        [b["bin"], f"{b['count']:,}", _num(b["mean_predicted"]), _num(b["observed_rate"])]
        for b in report["reliability_curve"]
    ]
    external_html = _external_section(report["external"]) if "external" in report else ""
    benchmark_html = _benchmark_section(report["benchmark"]) if "benchmark" in report else ""
    text_html = _text_section(report["text_model"]) if "text_model" in report else ""
    quantisation_html = _quantisation_note(report)
    known = report.get("known_sites")
    known_html = (
        f"<li>The model trusts {known['count']:,} well-known domains (Tranco top "
        f"{known['tranco_top_n']:,}, minus shared-hosting platforms) more than other sites, "
        f"except {known['untrusted_hosts']:,} hosts on them that were repeatedly used for "
        f"phishing in the training period. "
        f"{_pct(known['raw_phishing_share'])} of recent PhishTank URLs are on the "
        f"still-trusted hosts and are harder to catch. "
        f"See the well-known-site rows above.</li>"
    ) if known else ""
    coefficient_rows = [
        [name, f"{value:+.2f}"]
        for name, value in sorted(report["coefficients"].items(), key=lambda kv: -abs(kv[1]))
    ]

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PhishGuard model card — {escape(report["model_version"])}</title>
<style>{_STYLE}</style></head><body>
<h1>PhishGuard URL model</h1>
<p class="note">Version <code>{escape(report["model_version"])}</code> ·
generated automatically from the evaluation run</p>

<h2>What this model is</h2>
<p>A logistic regression over {len(report["coefficients"])} hand-engineered URL features,
with Platt-scaled probabilities. It looks only at the text of a URL and whether its host
is a well-known site — never at page content — and is an aid to judgement, not a
guarantee.</p>

<h2>Data</h2>
{_table(["Dataset", "Snapshot date", "SHA-256"], data_rows)}
<p class="note">Split: {escape(report["split"])}. No site appears in more than one split.</p>
{_table(["Split", "Benign URLs", "Phishing URLs"], split_rows)}

<h2>Held-out test results</h2>
{_table(["Metric", "Value"], metric_rows)}

<h2>Operating point</h2>
{_table(["", ""], threshold_rows)}

<h2>Results by URL shape</h2>
{_table(["Test slice", "Benign", "Phishing", "False-positive rate", "Recall"], slice_rows)}

{external_html}
<h2>Calibration</h2>
<p>If the model is well calibrated, the points sit on the dashed diagonal.</p>
{_reliability_svg(report["reliability_curve"])}
{_table(["Predicted range", "URLs", "Mean predicted", "Observed phishing rate"], reliability_rows)}

<h2>What drives the score</h2>
<p class="note">Coefficients on standardised features; positive pushes towards phishing.</p>
{_table(["Feature", "Coefficient"], coefficient_rows)}

{benchmark_html}
<h2>Known limitations</h2>
<ul>
{known_html}
<li>Benign training URLs are mostly deep pages from a web crawl while phishing URLs are
mostly short landing pages, so results differ by URL shape. See the table above.</li>
<li>The benign sample cannot be filtered by page status, so it includes some error pages
and redirects on legitimate sites.</li>
<li>Phishing hosted on legitimate platforms (form builders, site builders, URL shorteners)
cannot be told apart from benign pages on the same platform by the URL alone.</li>
<li>Benign URLs from domains Tranco does not list are sampled from a web crawl without a
reputation filter, so a small share of them may not be benign.</li>
<li>Phishing changes continuously; performance degrades between retrains.</li>
</ul>
{quantisation_html}
{text_html}
</body></html>
"""
