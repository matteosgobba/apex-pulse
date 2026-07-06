import { formatDateTime } from "@/lib/formatters";

export function FreshnessIndicator({
  generatedAt,
  label = "Generated"
}: {
  generatedAt: string | null | undefined;
  label?: string;
}) {
  return (
    <div className="rounded-lg border border-apex-border bg-apex-panelSoft px-3 py-2">
      <p className="text-[0.7rem] font-semibold uppercase tracking-[0.16em] text-apex-muted">
        {label}
      </p>
      <p className="mt-1 font-mono text-sm text-slate-100">{formatDateTime(generatedAt)}</p>
    </div>
  );
}
