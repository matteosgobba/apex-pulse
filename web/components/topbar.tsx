import type { HealthResponse } from "@/lib/dashboard-types";
import { humanizeToken } from "@/lib/formatters";
import { FreshnessIndicator } from "@/components/freshness-indicator";

export function Topbar({
  health,
  generatedAt
}: {
  health: HealthResponse | null;
  generatedAt: string | null | undefined;
}) {
  const status = health?.dashboard_artifact_status ?? "unavailable";
  return (
    <header className="flex flex-col gap-4 border-b border-apex-border bg-apex-bg/88 px-4 py-4 backdrop-blur md:flex-row md:items-center md:justify-between lg:px-8">
      <div>
        <p className="text-sm font-semibold uppercase tracking-[0.18em] text-apex-muted">
          Artifact-Driven Monitoring
        </p>
        <h1 className="mt-1 text-2xl font-semibold text-apex-text">Current Event Dashboard</h1>
        <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-300">
          Data may refresh after the local operator workflow exports new dashboard artifacts. This
          page does not claim live telemetry.
        </p>
      </div>
      <div className="grid gap-3 sm:grid-cols-2 md:min-w-[25rem]">
        <div className="rounded-lg border border-apex-border bg-apex-panelSoft px-3 py-2">
          <p className="text-[0.7rem] font-semibold uppercase tracking-[0.16em] text-apex-muted">
            API / Data
          </p>
          <p className="mt-1 text-sm font-semibold text-apex-text">{humanizeToken(status)}</p>
        </div>
        <FreshnessIndicator generatedAt={generatedAt} />
      </div>
    </header>
  );
}
