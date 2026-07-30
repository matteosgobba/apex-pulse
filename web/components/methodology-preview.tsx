import Link from "next/link";

const STEPS = [
  {
    number: "01",
    title: "Observe practice",
    body: "Public FP1, FP2 and FP3 timing, weather and session context form the pre-qualifying evidence."
  },
  {
    number: "02",
    title: "Predict qualifying pace",
    body: "A selected tabular model estimates each driver’s gap to pole, then produces the predicted ranking."
  },
  {
    number: "03",
    title: "Preserve and evaluate",
    body: "The forecast is frozen before qualifying and compared with the official result afterwards."
  }
];

export function MethodologyPreview() {
  return (
    <section id="methodology" aria-labelledby="methodology-preview-title">
      <div className="rounded-[2rem] border border-apex-border bg-apex-elevated p-6 shadow-card sm:p-10">
        <div className="flex flex-wrap items-end justify-between gap-5">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-apex-accent">
              How it works
            </p>
            <h2
              id="methodology-preview-title"
              className="mt-2 max-w-2xl text-3xl font-semibold tracking-tight text-apex-text"
            >
              Practice data in. Preserved qualifying prediction out.
            </h2>
          </div>
          <Link
            href="/methodology"
            className="rounded-full bg-apex-text px-5 py-2.5 text-sm font-semibold text-apex-bg transition-colors hover:bg-apex-secondary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-apex-accent"
          >
            Explore methodology
          </Link>
        </div>
        <div className="mt-8 grid gap-4 md:grid-cols-3">
          {STEPS.map((step) => (
            <article
              key={step.number}
              className="rounded-2xl border border-apex-border bg-apex-panel p-5"
            >
              <p className="text-sm font-semibold text-apex-accent">{step.number}</p>
              <h3 className="mt-4 text-lg font-semibold text-apex-text">{step.title}</h3>
              <p className="mt-2 text-sm leading-6 text-apex-secondary">{step.body}</p>
            </article>
          ))}
        </div>
        <p className="mt-6 max-w-3xl text-xs leading-5 text-apex-muted">
          Apex Pulse uses public data only. It has no private team telemetry, paid live feed, or
          control over backend forecasting operations.
        </p>
      </div>
    </section>
  );
}
