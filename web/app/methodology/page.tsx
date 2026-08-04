import type { Metadata } from "next";

import { AppShell } from "@/components/app-shell";
import { TechnicalDetails } from "@/components/technical-details";

export const metadata: Metadata = {
  title: "Methodology",
  description: "How Apex Pulse creates and evaluates Formula 1 qualifying predictions."
};

const INTRODUCTION = [
  {
    number: "01",
    title: "What Apex Pulse predicts",
    body: "The core output is each driver’s predicted gap to pole and the resulting qualifying ranking. The public artifacts do not currently claim pole or Q3 probabilities."
  },
  {
    number: "02",
    title: "Data sources",
    body: "FastF1 provides public historical practice laps, results, timing, weather and limited public telemetry. Local cached event schedules supply the displayed session times."
  }
];

const PROCESS = [
  {
    number: "03",
    title: "Practice checkpoints",
    body: "The dataset supports predictions after FP1, FP2 and FP3. Each checkpoint can use only the sessions that have already happened."
  },
  {
    number: "04",
    title: "Feature engineering",
    body: "Interpretable practice pace, sector performance, tyre context, teammate-relative pace, session conditions and strictly time-aware historical form are the foundation."
  },
  {
    number: "05",
    title: "Model policy",
    body: "Candidate tabular models are compared with strong practice and historical baselines using season-aware, walk-forward evaluation. The selected policy is exported by the backend."
  },
  {
    number: "06",
    title: "Forecast and settlement",
    body: "The forecast is written before qualifying and preserved immutably. Settlement is a separate post-qualifying comparison; it never rewrites or fills missing predictions."
  }
];

const TRANSPARENCY = [
  {
    number: "07",
    title: "Evaluation metrics",
    body: "Gap MAE and RMSE measure pace error. Mean absolute position error and top-k agreement show how well the ranking matched the official order."
  },
  {
    number: "08",
    title: "Public-data limitations",
    body: "The project has no private team data, paid live feed, fuel state, brake pressure, battery state or real-time telemetry. Public signals are partial and may be stale or incomplete."
  },
  {
    number: "09",
    title: "Prospective evidence and backtests",
    body: "Predictions made before an observed qualifying session are reported separately from historical model-development backtests and legacy noncanonical records."
  }
];

export default function MethodologyPage() {
  return (
    <AppShell health={null}>
      <div className="space-y-16">
        <section className="relative overflow-hidden rounded-[2rem] border border-white/10 bg-apex-ink px-6 py-10 text-apex-onStrong shadow-hero sm:px-10 sm:py-14">
          <div
            className="absolute inset-y-0 left-0 w-1.5 bg-apex-accent"
            aria-hidden="true"
          />
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

        <section aria-labelledby="methodology-foundation-title">
          <SectionHeading
            eyebrow="Prediction foundation"
            title="A focused output, built from public evidence."
            id="methodology-foundation-title"
          />
          <div className="mt-7 grid gap-5 md:grid-cols-2">
            {INTRODUCTION.map((section) => (
              <article
                key={section.number}
                className="group rounded-3xl border border-apex-border bg-apex-elevated p-6 shadow-card transition-colors hover:border-apex-accent/35 sm:p-8"
              >
                <div className="flex items-start justify-between gap-5">
                  <p className="text-xs font-semibold uppercase tracking-[0.16em] text-apex-accent">
                    Foundation
                  </p>
                  <span className="text-3xl font-semibold text-apex-border transition-colors group-hover:text-apex-accent/50">
                    {section.number}
                  </span>
                </div>
                <h2 className="mt-8 text-2xl font-semibold tracking-tight text-apex-text">
                  {section.title}
                </h2>
                <p className="mt-3 max-w-xl text-sm leading-7 text-apex-secondary">
                  {section.body}
                </p>
              </article>
            ))}
          </div>
        </section>

        <section
          aria-labelledby="methodology-process-title"
          className="overflow-hidden rounded-[2rem] border border-apex-border bg-apex-panel p-6 shadow-card sm:p-9"
        >
          <SectionHeading
            eyebrow="How Apex Pulse works"
            title="One controlled lifecycle from practice to evaluation."
            id="methodology-process-title"
          />
          <div className="relative mt-10">
            <div
              className="pointer-events-none absolute inset-x-0 top-5 hidden grid-cols-4 gap-x-7 lg:grid"
              aria-hidden="true"
            >
              <span className="col-start-1 col-end-4 -mr-12 ml-5 h-px bg-apex-border" />
            </div>
            <ol className="grid gap-x-7 gap-y-9 md:grid-cols-2 lg:grid-cols-4">
              {PROCESS.map((section) => (
                <li key={section.number} className="relative border-l border-apex-border pl-5 md:border-l-0 md:pl-0">
                  <span className="relative z-10 inline-flex h-10 min-w-10 items-center justify-center rounded-full border border-apex-accent/35 bg-apex-panel px-2 text-xs font-bold text-apex-accent shadow-card">
                    {section.number}
                  </span>
                  <h3 className="mt-5 text-lg font-semibold text-apex-text">{section.title}</h3>
                  <p className="mt-3 text-sm leading-6 text-apex-secondary">{section.body}</p>
                </li>
              ))}
            </ol>
          </div>
        </section>

        <section aria-labelledby="methodology-transparency-title">
          <SectionHeading
            eyebrow="Evaluation and transparency"
            title="Performance claims stay attached to their evidence."
            id="methodology-transparency-title"
          />
          <div className="mt-7 grid gap-5 lg:grid-cols-3">
            {TRANSPARENCY.map((section, index) => (
              <article
                key={section.number}
                className={`relative overflow-hidden rounded-3xl border p-6 shadow-card sm:p-7 ${
                  index === 1
                    ? "border-apex-accent/25 bg-apex-accentSoft"
                    : "border-apex-border bg-apex-panel"
                }`}
              >
                <div
                  className={`absolute inset-x-0 top-0 h-1 ${
                    index === 1 ? "bg-apex-accent" : "bg-apex-border"
                  }`}
                  aria-hidden="true"
                />
                <p className="text-xs font-semibold text-apex-accent">{section.number}</p>
                <h2 className="mt-7 text-xl font-semibold text-apex-text">{section.title}</h2>
                <p className="mt-3 text-sm leading-7 text-apex-secondary">{section.body}</p>
              </article>
            ))}
          </div>
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

function SectionHeading({ eyebrow, title, id }: { eyebrow: string; title: string; id: string }) {
  return (
    <div>
      <p className="text-xs font-semibold uppercase tracking-[0.16em] text-apex-accent">
        {eyebrow}
      </p>
      <h2 id={id} className="mt-2 max-w-3xl text-3xl font-semibold tracking-tight text-apex-text">
        {title}
      </h2>
    </div>
  );
}
