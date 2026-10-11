import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { type AnalyseResponse, type ErrorResponse, decidingPart, firstUrlIn } from "../lib/messages";
import { PageAccessError, inExtension, readActivePageAddress } from "../lib/page";
import { type HistoryEntry, type InputType, defaultStore } from "../lib/storage";
import { hasWording } from "../lib/text-features";
import { VERDICT_COPY } from "../lib/verdict";
import { InferenceClient } from "../lib/worker-client";
import {
  ConfidenceBand,
  FeedbackButton,
  HistoryDrawer,
  ModelVersionTag,
  NextSteps,
  PartSection,
  RiskMeter,
  VerdictBadge,
} from "./components";

type Status =
  | { kind: "idle" }
  | { kind: "working"; label: string }
  | { kind: "error"; message: string; retry: () => void }
  | { kind: "result"; result: AnalyseResponse; feedbackSent: boolean };

const ERROR_COPY: Record<ErrorResponse["code"], string> = {
  INVALID_INPUT: "That doesn't look like a web address or a message.",
  MODEL_LOAD_FAILED: "Couldn't load the detection model.",
  INFERENCE_FAILED: "Couldn't analyse this.",
};

function openPage(path: string) {
  const url = inExtension() ? chrome.runtime.getURL(path) : `/${path}`;
  if (inExtension()) void chrome.tabs.create({ url });
  else window.open(url, "_blank", "noopener");
}

function Onboarding({ onDone }: { onDone: () => void }) {
  // Focus the only action so the screen can be dismissed with Enter.
  const done = useRef<HTMLButtonElement>(null);
  useEffect(() => done.current?.focus(), []);
  return (
    <main className="onboarding">
      <h1>Welcome to PhishGuard</h1>
      <p className="lead">PhishGuard never scans anything until you click Analyse, and nothing leaves your device.</p>
      <ul>
        <li>It reads a page's address only when you ask it to.</li>
        <li>The detection model runs inside your browser. There is no server.</li>
        <li>It is an aid, not a guarantee: always check before entering a password.</li>
      </ul>
      <button type="button" className="primary" onClick={onDone} ref={done}>
        Got it
      </button>
    </main>
  );
}

export function App() {
  const store = useMemo(defaultStore, []);
  const client = useMemo(() => new InferenceClient(), []);
  const modelLoaded = useRef(false);
  const [onboarded, setOnboarded] = useState<boolean | null>(null);
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [pasted, setPasted] = useState("");
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  useEffect(() => {
    void store.getSettings().then((settings) => setOnboarded(settings.onboarded));
    void store.listHistory().then(setHistory);
  }, [store]);

  const run = useCallback(
    async (getInput: () => Promise<{ url?: string; text?: string }>, inputType: InputType) => {
      const retry = () => void run(getInput, inputType);
      setStatus({ kind: "working", label: modelLoaded.current ? "Analysing…" : "Loading model…" });
      let input: { url?: string; text?: string };
      try {
        input = await getInput();
      } catch (error) {
        const message =
          error instanceof PageAccessError
            ? "Couldn't analyse this page. Browser pages and new tabs can't be read."
            : "Nothing to analyse: paste a link or the text of a message.";
        return setStatus({ kind: "error", message, retry });
      }
      const response = await client.analyse(input);
      if (response.type === "ERROR") {
        return setStatus({ kind: "error", message: ERROR_COPY[response.code], retry });
      }
      modelLoaded.current = true;
      setStatus({ kind: "result", result: response, feedbackSent: false });
      await store.addHistory(response, inputType);
      setHistory(await store.listHistory());
    },
    [client, store],
  );

  // A pasted message is judged twice: its link by the URL model, its wording by
  // the text model. A bare link has no wording worth judging.
  const analysePasted = () =>
    run(async () => {
      const url = firstUrlIn(pasted) ?? undefined;
      const text = hasWording(pasted, 5000) ? pasted : undefined;
      if (!url && !text) throw new Error("nothing to analyse");
      return { ...(url && { url }), ...(text && { text }) };
    }, "pasted");

  // The page's address is requested here and nowhere else: only from the click handler.
  const analysePage = async () => ({ url: await readActivePageAddress() });

  if (onboarded === null) return null;
  if (!onboarded) {
    return (
      <Onboarding
        onDone={() => {
          setOnboarded(true);
          void store.saveSettings({ onboarded: true });
        }}
      />
    );
  }

  const busy = status.kind === "working";
  return (
    <main>
      <header>
        <h1>PhishGuard</h1>
        <span className="muted small">Runs on your device</span>
      </header>

      <section className="actions" aria-label="Analyse">
        {inExtension() && (
          <button
            type="button"
            className="primary"
            disabled={busy}
            onClick={() => void run(analysePage, "page")}
          >
            Analyse this page
          </button>
        )}
        <label htmlFor="pasted">Or paste a link or a message</label>
        <textarea
          id="pasted"
          rows={2}
          value={pasted}
          maxLength={8192}
          placeholder="https://… or the text of an SMS or email"
          onChange={(event) => setPasted(event.target.value)}
        />
        <button type="button" disabled={busy || pasted.trim() === ""} onClick={() => void analysePasted()}>
          Analyse pasted text
        </button>
      </section>

      {/* Announced to screen readers whenever the verdict changes. */}
      <div aria-live="polite" className="visually-hidden">
        {status.kind === "result" &&
          `${VERDICT_COPY[status.result.verdict].title}. Risk score ${status.result.score} out of 100.`}
        {status.kind === "error" && status.message}
        {status.kind === "working" && status.label}
      </div>

      {status.kind === "idle" && (
        <p className="empty">Nothing analysed yet — click Analyse on a page or paste a message.</p>
      )}
      {status.kind === "working" && <p className="empty">{status.label}</p>}
      {status.kind === "error" && (
        <div className="error" role="alert">
          <p>{status.message}</p>
          <button type="button" onClick={status.retry}>
            Try again
          </button>
        </div>
      )}
      {status.kind === "result" && (
        <article className="result">
          <VerdictBadge verdict={status.result.verdict} />
          <RiskMeter score={status.result.score} verdict={status.result.verdict} />
          <ConfidenceBand
            confidence={decidingPart(status.result).confidence}
            probability={decidingPart(status.result).probability}
          />
          {status.result.address && <PartSection kind="address" part={status.result.address} />}
          {status.result.message && <PartSection kind="message" part={status.result.message} />}
          <NextSteps verdict={status.result.verdict} />
          <FeedbackButton
            sent={status.feedbackSent}
            onSend={() => {
              void store.addFeedback(status.result);
              setStatus({ ...status, feedbackSent: true });
            }}
          />
          <ModelVersionTag
            versions={[status.result.address, status.result.message].flatMap((part) =>
              part ? [part.modelVersion] : [],
            )}
            timing={status.result.timing}
          />
        </article>
      )}

      <HistoryDrawer
        history={history}
        onClear={() => {
          void store.clearHistory().then(() => setHistory([]));
        }}
      />

      <footer>
        <button type="button" className="link" onClick={() => openPage("models/model_card.html")}>
          Model card
        </button>
        <button type="button" className="link" onClick={() => openPage("src/options/index.html")}>
          Options
        </button>
      </footer>
    </main>
  );
}
