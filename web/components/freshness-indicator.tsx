import { evaluateFreshness, type FreshnessState } from "@/lib/freshness";

export function FreshnessIndicator({
  generatedAt,
  label = "Generated"
}: {
  generatedAt: string | null | undefined;
  label?: string;
}) {
  const freshness = evaluateFreshness(generatedAt);
  return (
    <div
      className="rounded-lg border border-apex-border bg-apex-panelSoft px-3 py-2"
      title={`${freshness.exactUtcLabel}; ${freshness.exactLocalLabel}`}
    >
      <p className="text-[0.7rem] font-semibold uppercase tracking-[0.16em] text-apex-muted">
        {label}
      </p>
      <div className="mt-1 flex flex-wrap items-center gap-2">
        <p className="font-mono text-sm text-slate-100">{freshness.relativeLabel}</p>
        <FreshnessPill state={freshness.state} />
      </div>
      <p className="sr-only">
        {freshness.exactUtcLabel}. {freshness.exactLocalLabel}. Dashboard refresh depends on the
        operator artifact-export workflow.
      </p>
    </div>
  );
}

function FreshnessPill({ state }: { state: FreshnessState }) {
  const labels: Record<FreshnessState, string> = {
    fresh: "Fresh",
    aging: "Aging",
    stale: "Stale",
    unknown: "Unknown"
  };
  const className: Record<FreshnessState, string> = {
    fresh: "border-emerald-300/40 bg-emerald-300/10 text-emerald-100",
    aging: "border-amber-300/40 bg-amber-300/10 text-amber-100",
    stale: "border-rose-300/40 bg-rose-300/10 text-rose-100",
    unknown: "border-slate-500/40 bg-slate-500/10 text-slate-200"
  };

  return (
    <span className={`rounded-full border px-2 py-0.5 text-[0.65rem] font-semibold ${className[state]}`}>
      {labels[state]}
    </span>
  );
}
