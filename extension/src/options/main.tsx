import { StrictMode, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { type FeedbackEntry, type Settings, defaultStore } from "../lib/storage";
import "../popup/popup.css";

function Options() {
  const store = useMemo(defaultStore, []);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [feedback, setFeedback] = useState<FeedbackEntry[]>([]);
  const [analyses, setAnalyses] = useState(0);
  const [notice, setNotice] = useState("");

  const refresh = async () => {
    setSettings(await store.getSettings());
    setFeedback(await store.listFeedback());
    setAnalyses((await store.stats()).analyses);
  };
  useEffect(() => {
    void refresh();
  }, []);

  if (!settings) return null;
  const save = async (changes: Partial<Settings>) => {
    // Show the change at once; the controls would otherwise lag behind the click.
    setSettings({ ...settings, ...changes });
    await store.saveSettings(changes);
    await refresh();
    setNotice("Saved.");
  };

  return (
    <main>
      <h1>PhishGuard options</h1>
      <p className="muted">Everything here is stored on this device only. Nothing is uploaded or synced.</p>

      <div className="field">
        <label htmlFor="retention">Keep history for</label>
        <select
          id="retention"
          value={settings.retentionDays}
          onChange={(event) => void save({ retentionDays: Number(event.target.value) as Settings["retentionDays"] })}
        >
          <option value={0}>Don't keep history</option>
          <option value={7}>7 days</option>
          <option value={30}>30 days</option>
        </select>
        <span className="muted small">Turning history off deletes what is stored now.</span>
      </div>

      <div className="field">
        <label className="check">
          <input
            type="checkbox"
            checked={settings.keepAddresses}
            onChange={(event) => void save({ keepAddresses: event.target.checked })}
          />
          <span>
            Show addresses in history
            <br />
            <span className="muted small">
              Off by default: history then stores only a scrambled fingerprint of each address. When on, the
              address is stored without anything after a “?”. Turning this off removes stored addresses.
            </span>
          </span>
        </label>
      </div>

      <h3>Stored on this device</h3>
      <p>
        {analyses} {analyses === 1 ? "analysis" : "analyses"} in history · {feedback.length}{" "}
        {feedback.length === 1 ? "result" : "results"} you reported as wrong
      </p>
      <div className="row">
        <button
          type="button"
          onClick={() => void store.clearHistory().then(refresh).then(() => setNotice("History cleared."))}
        >
          Clear history
        </button>
        <button
          type="button"
          onClick={() => void store.clearFeedback().then(refresh).then(() => setNotice("Reports cleared."))}
        >
          Clear reports
        </button>
      </div>

      {feedback.length > 0 && (
        <details>
          <summary>Results you reported as wrong</summary>
          <ul>
            {feedback.map((entry) => (
              <li key={entry.id}>
                {entry.address ?? "A pasted message (text not stored)"} — PhishGuard said “{entry.verdict}”, you said “{entry.userLabel}”
                <span className="muted small"> · {new Date(entry.ts).toLocaleString()}</span>
              </li>
            ))}
          </ul>
        </details>
      )}

      <p className="muted small" role="status" aria-live="polite">
        {notice}
      </p>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Options />
  </StrictMode>,
);
