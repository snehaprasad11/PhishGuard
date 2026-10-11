import type { Confidence, EvidenceItem, Highlight, PartResult, Verdict } from "../lib/messages";
import type { HistoryEntry } from "../lib/storage";
import { VERDICT_COPY } from "../lib/verdict";

// Risk is never conveyed by colour alone: every verdict has its own icon and wording.
const VERDICT_ICON: Record<Verdict, string> = { phishing: "⚠", suspicious: "?", clear: "✓" };

export function VerdictBadge({ verdict }: { verdict: Verdict }) {
  return (
    <div className={`verdict verdict-${verdict}`}>
      <span className="verdict-icon" aria-hidden="true">
        {VERDICT_ICON[verdict]}
      </span>
      <div>
        <h2>{VERDICT_COPY[verdict].title}</h2>
        <p>{VERDICT_COPY[verdict].summary}</p>
      </div>
    </div>
  );
}

export function RiskMeter({ score, verdict }: { score: number; verdict: Verdict }) {
  return (
    <div className="meter-row">
      <div
        className="meter"
        role="meter"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={score}
        aria-label={`Risk score ${score} out of 100`}
      >
        <div className={`meter-fill verdict-${verdict}`} style={{ width: `${Math.max(score, 2)}%` }} />
      </div>
      <span className="meter-value">
        <strong>{score}</strong> / 100
      </span>
    </div>
  );
}

/** A model estimate is never certain, so the extremes are shown as bounds, not 0% or 100%. */
export function formatProbability(probability: number): string {
  if (probability > 0.99) return "over 99%";
  if (probability < 0.01) return "under 1%";
  return `${Math.round(probability * 100)}%`;
}

const CONFIDENCE_COPY: Record<Confidence, string> = {
  high: "High confidence — the result is well clear of the decision line.",
  medium: "Medium confidence.",
  low: "Low confidence — this is close to the decision line, so treat the result with care.",
};

export function ConfidenceBand({ confidence, probability }: { confidence: Confidence; probability: number }) {
  return (
    <p className="confidence">
      <span className={`pill pill-${confidence}`}>{confidence} confidence</span>
      <span className="visually-hidden">{CONFIDENCE_COPY[confidence]}</span>
      <span className="muted"> Estimated phishing probability {formatProbability(probability)}</span>
    </p>
  );
}

/** The analysed text with its suspicious parts marked, and the reason for each below it. */
export function HighlightedInput({
  text,
  highlights,
  label,
  monospace,
}: {
  text: string;
  highlights: Highlight[];
  label: string;
  monospace: boolean;
}) {
  const parts: { text: string; marked: boolean }[] = [];
  let position = 0;
  for (const highlight of highlights) {
    if (highlight.start > position) parts.push({ text: text.slice(position, highlight.start), marked: false });
    parts.push({ text: text.slice(highlight.start, highlight.end), marked: true });
    position = highlight.end;
  }
  if (position < text.length) parts.push({ text: text.slice(position), marked: false });
  const reasons = [...new Set(highlights.map((highlight) => highlight.reason))];

  return (
    <div role="group" aria-label={label}>
      <p className={monospace ? "address" : "address prose"}>
        {parts.map((part, i) => (part.marked ? <mark key={i}>{part.text}</mark> : <span key={i}>{part.text}</span>))}
      </p>
      {reasons.length > 0 && (
        <ul className="reasons">
          {reasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function EvidenceList({ evidence, verdict }: { evidence: EvidenceItem[]; verdict: Verdict }) {
  // The reasons shown must support the verdict: what raised the risk for a
  // warning, what lowered it for a clear result. `evidence` is sorted from the
  // biggest raise to the biggest reduction.
  const top =
    verdict === "clear"
      ? evidence.filter((item) => item.contribution < -0.05).reverse().slice(0, 3)
      : evidence.filter((item) => item.contribution > 0.05).slice(0, 3);
  const largest = Math.max(...evidence.map((item) => Math.abs(item.contribution)), 1e-9);
  return (
    <div>
      <h4>{verdict === "clear" ? "What lowered the risk" : "What raised the risk"}</h4>
      {top.length === 0 ? (
        <p className="muted">No single factor stood out.</p>
      ) : (
        <ol className="evidence">
          {top.map((item) => (
            <li key={item.feature}>{item.explanation}</li>
          ))}
        </ol>
      )}
      <details>
        <summary>All factors</summary>
        <p className="muted small">
          Each bar is exactly how much that factor moved the score. The model is a simple weighted sum, so
          these add up to the result.
        </p>
        <ul className="factors">
          {evidence.map((item) => (
            <li key={item.feature}>
              <span>{item.explanation}</span>
              <span className={item.contribution > 0 ? "factor-up" : "factor-down"}>
                <span
                  className="factor-bar"
                  style={{ width: `${(Math.abs(item.contribution) / largest) * 100}%` }}
                  aria-hidden="true"
                />
                {item.contribution > 0 ? "raises risk" : "lowers risk"}
              </span>
            </li>
          ))}
        </ul>
      </details>
    </div>
  );
}

const PART_VERDICT: Record<Verdict, string> = {
  phishing: "likely phishing",
  suspicious: "some warning signs",
  clear: "no warning signs",
};

/** One model's judgement: of the link, or of the message's wording. */
export function PartSection({ kind, part }: { kind: "address" | "message"; part: PartResult }) {
  const title = kind === "address" ? "The link" : "The wording";
  return (
    <section className="part" data-part={kind} aria-label={title}>
      <h3>
        {title}
        <span className={`part-verdict verdict-${part.verdict}`}>
          <span aria-hidden="true">{VERDICT_ICON[part.verdict]} </span>
          {PART_VERDICT[part.verdict]} · {part.score}/100
        </span>
      </h3>
      <HighlightedInput
        text={part.displayText}
        highlights={part.highlights}
        label={kind === "address" ? "Analysed address" : "Analysed message"}
        monospace={kind === "address"}
      />
      <EvidenceList evidence={part.evidence} verdict={part.verdict} />
    </section>
  );
}

export function NextSteps({ verdict }: { verdict: Verdict }) {
  return (
    <section aria-label="What to do next">
      <h3>What to do</h3>
      <ul className="steps">
        {VERDICT_COPY[verdict].steps.map((step) => (
          <li key={step}>{step}</li>
        ))}
      </ul>
    </section>
  );
}

export function FeedbackButton({ sent, onSend }: { sent: boolean; onSend: () => void }) {
  return sent ? (
    <p className="muted small" role="status">
      Thanks. Your report is saved on this device only.
    </p>
  ) : (
    <button type="button" className="link" onClick={onSend}>
      This result is wrong
    </button>
  );
}

export function ModelVersionTag({
  versions,
  timing,
}: {
  versions: string[];
  timing: { loadMs: number; analyseMs: number };
}) {
  return (
    <p className="muted small" data-testid="model-version" data-analyse-ms={timing.analyseMs} data-load-ms={timing.loadMs}>
      {versions.length === 1 ? "Model" : "Models"} {versions.join(" + ")} · analysed in {timing.analyseMs.toFixed(1)}{" "}
      ms on this device
    </p>
  );
}

export function HistoryDrawer({ history, onClear }: { history: HistoryEntry[]; onClear: () => void }) {
  return (
    <details className="history">
      <summary>History ({history.length})</summary>
      {history.length === 0 ? (
        <p className="muted small">No analyses stored.</p>
      ) : (
        <>
          <ul>
            {history.slice(0, 20).map((entry) => (
              <li key={entry.id}>
                <span className={`dot verdict-${entry.verdict}`} aria-hidden="true" />
                <span>
                  {VERDICT_COPY[entry.verdict].title} · {entry.score}/100
                  <br />
                  <span className="muted small">
                    {new Date(entry.ts).toLocaleString()} ·{" "}
                    {entry.address ??
                      (entry.subject === "message" ? "pasted message, text not stored" : "address not stored")}
                  </span>
                </span>
              </li>
            ))}
          </ul>
          <button type="button" className="link" onClick={onClear}>
            Clear history
          </button>
        </>
      )}
    </details>
  );
}
