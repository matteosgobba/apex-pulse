import type { LifecycleState } from "@/lib/dashboard-types";
import { humanizeToken } from "@/lib/formatters";

const STATE_STYLES: Record<LifecycleState, string> = {
  no_event_available: "border-slate-500/60 bg-slate-500/10 text-slate-200",
  practice_in_progress: "border-cyan-400/60 bg-cyan-400/10 text-cyan-100",
  ready_to_forecast: "border-emerald-300/60 bg-emerald-300/10 text-emerald-100",
  forecast_available: "border-sky-300/60 bg-sky-300/10 text-sky-100",
  awaiting_qualifying_targets: "border-amber-300/60 bg-amber-300/10 text-amber-100",
  settled: "border-violet-300/60 bg-violet-300/10 text-violet-100",
  blocked: "border-rose-300/60 bg-rose-300/10 text-rose-100",
  legacy_descriptive_only: "border-zinc-300/60 bg-zinc-300/10 text-zinc-100"
};

export function LifecycleBadge({
  state,
  label
}: {
  state: LifecycleState | null | undefined;
  label?: string | null;
}) {
  const resolved = state ?? "no_event_available";
  return (
    <span
      className={`inline-flex max-w-full items-center gap-2 rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.12em] ${STATE_STYLES[resolved]}`}
    >
      <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-current" aria-hidden="true" />
      <span className="truncate">{label || humanizeToken(resolved)}</span>
    </span>
  );
}
