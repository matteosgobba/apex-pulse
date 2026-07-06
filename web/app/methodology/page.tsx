import { AppShell } from "@/components/app-shell";

const SECTIONS = [
  {
    title: "Prediction Target",
    body: "Apex Pulse predicts Formula 1 qualifying performance, centered on ranking and gap to pole. The public dashboard surfaces exported forecast and monitoring artifacts rather than raw model internals."
  },
  {
    title: "Weekend Lifecycle",
    body: "The monitored workflow ingests available FP1, FP2, and FP3 evidence, prepares event features, registers lineage, runs prospective preflight, exports forecasts only when allowed, then settles predictions after qualifying targets are ingested."
  },
  {
    title: "Preflight Safety",
    body: "Preflight gates prevent forecast publication when required evidence, chronology, registry lineage, or target-isolation checks are not satisfied. The website displays those states but never overrides them."
  },
  {
    title: "Forecast And Settlement",
    body: "Forecast artifacts use pre-qualifying information. Settlement artifacts are separate post-qualifying comparisons against actual qualifying classification and gap-to-pole outcomes."
  },
  {
    title: "Public Data Limits",
    body: "The system relies on public or freely accessible data. It does not claim private team telemetry, paid live feeds, or lap-by-lap real-time telemetry in this dashboard."
  },
  {
    title: "Evidence Classes",
    body: "Valid prospective monitoring evidence is kept separate from historical backtests and legacy descriptive records. Legacy Australia and Great Britain records remain labeled as noncanonical and excluded from valid prospective aggregates."
  }
];

export default function MethodologyPage() {
  return (
    <AppShell health={null} generatedAt={null}>
      <div className="space-y-6">
        <section className="rounded-lg border border-apex-border bg-apex-panel p-6 shadow-panel">
          <p className="text-sm font-semibold uppercase tracking-[0.18em] text-apex-muted">
            Methodology And Trust
          </p>
          <h1 className="mt-3 max-w-4xl text-3xl font-semibold text-apex-text md:text-5xl">
            Read-only qualifying intelligence from monitored artifacts.
          </h1>
          <p className="mt-4 max-w-3xl text-sm leading-6 text-slate-300">
            This page summarizes what the dashboard can safely say today. It is a technical
            portfolio interface over exported artifacts, not a news site, operator console, or live
            telemetry client.
          </p>
        </section>
        <section className="grid gap-4 md:grid-cols-2">
          {SECTIONS.map((section) => (
            <article
              key={section.title}
              className="rounded-lg border border-apex-border bg-apex-panel/85 p-5"
            >
              <h2 className="text-lg font-semibold text-apex-text">{section.title}</h2>
              <p className="mt-3 text-sm leading-6 text-slate-300">{section.body}</p>
            </article>
          ))}
        </section>
      </div>
    </AppShell>
  );
}
