import type { FreshnessStatus } from "@/lib/freshness";

export function DataFreshnessNotice({ freshness }: { freshness: FreshnessStatus }) {
  const stale = freshness.state === "stale";
  return (
    <div
      role={stale ? "status" : undefined}
      className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium ${
        stale ? "bg-amber-50 text-amber-800" : "bg-slate-100 text-slate-600"
      }`}
      title={`${freshness.exactUtcLabel} · ${freshness.exactLocalLabel}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${stale ? "bg-amber-500" : "bg-slate-400"}`} />
      {stale ? `Data may be stale · ${freshness.relativeLabel}` : freshness.relativeLabel}
    </div>
  );
}
