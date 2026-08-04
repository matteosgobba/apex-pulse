import { ErrorState } from "@/components/error-state";
import { LegacyWarning } from "@/components/legacy-warning";
import { LifecycleBadge } from "@/components/lifecycle-badge";
import { SettlementComparisonTable } from "@/components/settlement-comparison-table";
import { SettlementSummary } from "@/components/settlement-summary";
import { TableEmptyState } from "@/components/table-empty-state";
import { dashboardRows } from "@/lib/dashboard-collections";
import type { SettlementPageData } from "@/lib/dashboard-types";
import { formatEventNameWithFlag } from "@/lib/event-display";
import { formatDateTime, formatInteger, formatText, humanizeToken } from "@/lib/formatters";

export function SettlementPageView({ data }: { data: SettlementPageData }) {
  if (data.error) {
    return <ErrorState error={data.error} />;
  }

  const current = data.currentEvent?.data;
  const settlement = data.settlement;
  const settlementData = settlement?.data;
  const rows = dashboardRows(
    settlementData?.settlement_evaluable_rows ?? settlementData?.driver_comparison
  );
  const forecastRows = dashboardRows(
    data.forecast?.data.settlement_evaluable_rows
      ?? data.forecast?.data.qualifying_eligible_forecast_rows
      ?? data.forecast?.data.leaderboard
  );
  const forecastOnlyRows = dashboardRows(settlementData?.forecast_only_rows);
  const identity = settlementData?.event_identity ?? current?.event_identity;
  const lifecycleState = settlementData?.lifecycle_state ?? current?.lifecycle?.state;
  const legacy =
    lifecycleState === "legacy_descriptive_only" || current?.legacy_status?.legacy_noncanonical;

  if (!settlement || settlement.status === "empty" || rows.length === 0) {
    return (
      <TableEmptyState
        title="Settlement comparison unavailable."
        message="Settlement becomes available only after qualifying targets are separately ingested and the exported forecast is settled."
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
              {formatEventNameWithFlag(identity?.event) || formatText(null)} Settlement
            </h1>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-300">
              This comparison evaluates a pre-qualifying forecast against final qualifying outcomes.
              Prediction fields and observed qualifying results are shown separately.
            </p>
          </div>
          <div className="flex shrink-0 flex-col gap-3">
            <LifecycleBadge state={lifecycleState} label={humanizeToken(lifecycleState)} />
            <div className="rounded-md border border-apex-border bg-apex-bg/45 p-3 text-sm">
              <p className="text-apex-muted">Settled at</p>
              <p className="mt-1 font-mono text-sm text-apex-text">
                {formatDateTime(settlementData?.settlement_metadata?.settled_at_utc)}
              </p>
            </div>
          </div>
        </div>
      </section>
      {forecastOnlyRows.length > 0 ? (
        <section className="rounded-lg border border-amber-300/35 bg-amber-300/10 p-5">
          <h2 className="text-lg font-semibold text-apex-text">Partial Settlement Coverage</h2>
          <p className="mt-2 text-sm leading-6 text-amber-50">
            {formatInteger(rows.length)} settlement-evaluable drivers are shown in the main
            comparison. {formatInteger(forecastOnlyRows.length)} forecast-only rows are retained as
            audit-only records because they do not have evaluable qualifying targets.
          </p>
        </section>
      ) : null}
      <SettlementSummary settlement={settlementData} />
      <SettlementComparisonTable rows={rows} forecastRows={forecastRows} />
      <SettlementInterpretation legacy={Boolean(legacy)} />
    </div>
  );
}

function SettlementInterpretation({ legacy }: { legacy: boolean }) {
  return (
    <section className="rounded-lg border border-apex-border bg-apex-panel/85 p-5">
      <h2 className="text-lg font-semibold text-apex-text">Interpretation</h2>
      <div className="mt-3 space-y-2 text-sm leading-6 text-slate-300">
        <p>Forecasts are generated before qualifying from practice-derived evidence.</p>
        <p>
          Errors reflect uncertainty in public timing-derived pace estimates. Public data does not
          reveal hidden variables such as fuel loads, engine modes, setup, or tyre temperatures.
        </p>
        {legacy ? (
          <p>
            This settled record is legacy/noncanonical, so it must not be treated as valid
            prospective monitoring evidence.
          </p>
        ) : null}
      </div>
    </section>
  );
}
