export function KpiCard({
  label,
  value,
  hint
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  const unavailable = value === "Not available";
  return (
    <section className="rounded-lg border border-apex-border bg-apex-panel/85 p-4">
      <p className="text-xs font-semibold uppercase tracking-[0.14em] text-apex-muted">{label}</p>
      <p
        className={`mt-3 break-words font-mono text-2xl tabular-nums ${
          unavailable ? "text-slate-500" : "text-apex-text"
        }`}
      >
        {value}
      </p>
      {hint ? <p className="mt-2 text-sm text-apex-muted">{hint}</p> : null}
    </section>
  );
}
