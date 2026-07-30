import type { Metadata } from "next";

import { AppShell } from "@/components/app-shell";
import { TechnicalDetails } from "@/components/technical-details";

export const metadata: Metadata = {
  title: "Methodology",
  description: "How Apex Pulse creates and evaluates Formula 1 qualifying predictions."
};

const SECTIONS = [
  {
    title: "What Apex Pulse predicts",
    body: "The core output is each driver’s predicted gap to pole and the resulting qualifying ranking. The public artifacts do not currently claim pole or Q3 probabilities."
  },
  {
    title: "Data sources",
    body: "FastF1 provides public historical practice laps, results, timing, weather and limited public telemetry. Local cached event schedules supply the displayed session times."
  },
  {
    title: "Practice checkpoints",
    body: "The dataset supports predictions after FP1, FP2 and FP3. Each checkpoint can use only the sessions that have already happened."
  },
  {
    title: "Feature engineering",
    body: "Interpretable practice pace, sector performance, tyre context, teammate-relative pace, session conditions and strictly time-aware historical form are the foundation."
  },
  {
    title: "Model policy",
    body: "Candidate tabular models are compared with strong practice and historical baselines using season-aware, walk-forward evaluation. The selected policy is exported by the backend."
  },
  {
    title: "Forecast and settlement",
    body: "The forecast is written before qualifying and preserved immutably. Settlement is a separate post-qualifying comparison; it never rewrites or fills missing predictions."
  },
  {
    title: "Evaluation metrics",
    body: "Gap MAE and RMSE measure pace error. Mean absolute position error and top-k agreement show how well the ranking matched the official order."
  },
  {
    title: "Public-data limitations",
    body: "The project has no private team data, paid live feed, fuel state, brake pressure, battery state or real-time telemetry. Public signals are partial and may be stale or incomplete."
  },
  {
    title: "Prospective evidence and backtests",
    body: "Predictions made before an observed qualifying session are reported separately from historical model-development backtests and legacy noncanonical records."
  }
];

export default function MethodologyPage() {
  return (
    <AppShell health={null}>
      <div className="space-y-12">
        <section className="rounded-[2rem] border border-white/10 bg-apex-ink px-6 py-10 text-apex-onStrong shadow-hero sm:px-10 sm:py-14">
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-apex-accent">
            Methodology and trust
          </p>
          <h1 className="mt-3 max-w-4xl text-4xl font-semibold tracking-tight sm:text-5xl">
            A pre-qualifying prediction with a clear evidence trail.
          </h1>
          <p className="mt-4 max-w-3xl text-sm leading-7 text-apex-onStrongMuted">
            Apex Pulse turns public practice-session evidence into a qualifying forecast, freezes
            it before qualifying, and evaluates it only after official results are available.
          </p>
        </section>
        <section className="grid gap-4 md:grid-cols-2">
          {SECTIONS.map((section, index) => (
            <article
              key={section.title}
              className="rounded-3xl border border-apex-border bg-apex-panel p-6 shadow-card"
            >
              <p className="text-xs font-semibold text-apex-accent">
                {String(index + 1).padStart(2, "0")}
              </p>
              <h2 className="mt-4 text-xl font-semibold text-apex-text">{section.title}</h2>
              <p className="mt-3 text-sm leading-7 text-apex-secondary">{section.body}</p>
            </article>
          ))}
        </section>
        <TechnicalDetails
          title="Technical lifecycle details"
          items={[
            { label: "Contract version", value: "Dashboard schema 1.0" },
            { label: "Access mode", value: "Read-only exported JSON" },
            { label: "Forecast preservation", value: "Immutable snapshots" },
            { label: "Schedule source", value: "Existing local FastF1 cache" },
            { label: "Operator actions", value: "Not exposed by the website" },
            { label: "Live telemetry", value: "Not provided" }
          ]}
        />
      </div>
    </AppShell>
  );
}
