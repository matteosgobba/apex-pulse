import { ErrorState } from "@/components/error-state";
import { ForecastLeaderboard } from "@/components/forecast-leaderboard";
import { ForecastSummaryPanel } from "@/components/forecast-summary";
import { LegacyWarning } from "@/components/legacy-warning";
import { LifecycleBadge } from "@/components/lifecycle-badge";
import { TableEmptyState } from "@/components/table-empty-state";
import { dashboardRows } from "@/lib/dashboard-collections";
import type { ForecastLeaderboardRow, ForecastPageData } from "@/lib/dashboard-types";
import { formatDateTime, formatText, humanizeToken } from "@/lib/formatters";

export function ForecastPageView({ data }: { data: ForecastPageData }) {
  if (data.error) {
    return <ErrorState error={data.error} />;
  }

  const forecast = data.forecast;
  const current = data.currentEvent;
  const forecastData = forecast?.data;
  const currentData = current?.data;
  const identity = forecastData?.event_identity ?? currentData?.event_identity;
  const lifecycleState = forecastData?.lifecycle_state ?? currentData?.lifecycle?.state;
  const lifecycleLabel = currentData?.lifecycle?.display_label ?? humanizeToken(lifecycleState);
  const rows = dashboardRows(
    forecastData?.qualifying_eligible_forecast_rows ?? forecastData?.leaderboard
  );
  const forecastOnlyRows = dashboardRows(forecastData?.forecast_only_rows);
  const metadata = forecastData?.forecast_metadata;
  const legacy =
    lifecycleState === "legacy_descriptive_only" || currentData?.legacy_status?.legacy_noncanonical;

  if (!forecast || forecast.status === "empty" || rows.length === 0) {
    return (
      <TableEmptyState
        title="Forecast leaderboard unavailable."
        message="The dashboard API is reachable, but the current forecast artifact does not include ranked forecast rows."
      />
    );
  }

  return (
    <div className="space-y-6">
      {legacy ? <LegacyWarning /> : null}
      <section className="rounded-lg border border-apex-border bg-apex-panel p-6 shadow-panel">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="min-w-0">
            <p className="text-sm font-semibold uppercase tracking-[0.18em] text-apex-muted">
              {formatText(identity?.season)}
            </p>
            <h1 className="mt-2 break-words text-3xl font-semibold text-apex-text md:text-5xl">
              {formatText(identity?.event)} Forecast
            </h1>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-300">
              Forecast values are pre-qualifying estimates exported from the monitored workflow.
              They are not live telemetry and should not be read as certainty or probability.
            </p>
          </div>
          <div className="flex shrink-0 flex-col gap-3">
            <LifecycleBadge state={lifecycleState} label={lifecycleLabel} />
            <div className="rounded-md border border-apex-border bg-apex-bg/45 p-3 text-sm">
              <p className="text-apex-muted">Forecast timestamp</p>
              <p className="mt-1 font-mono text-sm text-apex-text">
                {formatDateTime(metadata?.forecast_timestamp)}
              </p>
            </div>
          </div>
        </div>
      </section>
      <ForecastSummaryPanel forecast={forecastData} />
      <ForecastLeaderboard rows={rows} />
      {forecastOnlyRows.length > 0 ? <ForecastOnlyAuditRows rows={forecastOnlyRows} /> : null}
    </div>
  );
}

function ForecastOnlyAuditRows({
  rows
}: {
  rows: ForecastLeaderboardRow[];
}) {
  return (
    <section className="rounded-lg border border-amber-300/35 bg-amber-300/10 p-5">
      <h2 className="text-lg font-semibold text-apex-text">Forecast-Only Audit Rows</h2>
      <p className="mt-2 text-sm leading-6 text-amber-50">
        These FP participants are preserved for auditability but excluded from the public qualifying
        leaderboard because they are not settlement-evaluable qualifying drivers.
      </p>
      <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {rows.map((row) => (
          <div
            key={`${row.driver_code ?? row.driver}-${row.forecast_only_reason}`}
            className="rounded-md border border-amber-200/25 bg-apex-bg/35 p-3"
          >
            <p className="font-semibold text-apex-text">{formatText(row.driver_code ?? row.driver)}</p>
            <p className="text-sm text-slate-300">{formatText(row.team)}</p>
            <p className="mt-1 text-xs text-amber-100">
              {humanizeToken(row.forecast_only_reason)}
            </p>
          </div>
        ))}
      </div>
    </section>
  );
}
