import Link from "next/link";

import { EmptyState } from "@/components/empty-state";
import { ErrorState } from "@/components/error-state";
import { ForecastLeaderboard } from "@/components/forecast-leaderboard";
import { KpiCard } from "@/components/kpi-card";
import { LegacyWarning } from "@/components/legacy-warning";
import { LifecycleBadge } from "@/components/lifecycle-badge";
import { SessionStatusRow } from "@/components/session-status-row";
import { StatusCard } from "@/components/status-card";
import type { CurrentEventPageData, LifecycleState } from "@/lib/dashboard-types";
import {
  formatDateTime,
  formatInteger,
  formatPercent,
  formatSeconds,
  formatText,
  humanizeToken
} from "@/lib/formatters";

export function CurrentEventPageView({ data }: { data: CurrentEventPageData }) {
  if (data.error) {
    return <ErrorState error={data.error} />;
  }

  const current = data.currentEvent;
  const currentData = current?.data;
  const lifecycle = currentData?.lifecycle;
  const state = lifecycle?.state ?? "no_event_available";
  const identity = currentData?.event_identity;
  const kpis = currentData?.summary_kpis;
  const preflight = currentData?.preflight;
  const forecast = currentData?.forecast_status;
  const settlement = currentData?.settlement_status;
  const protocol = currentData?.monitoring_protocol;
  const lineage = currentData?.registry_lineage;
  const legacy = currentData?.legacy_status;
  const sessions = data.practiceStatus?.data.sessions ?? [];
  const forecastRows = data.forecast?.data.leaderboard ?? [];
  const isLegacy = state === "legacy_descriptive_only" || Boolean(legacy?.legacy_noncanonical);
  const settlementMetrics = data.settlement?.data.summary_metrics;
  const showSettlementPreview =
    (data.settlement?.data.driver_comparison?.length ?? 0) > 0 && !isLegacy;
  const validHistory = data.historicalMonitoring?.data.valid_prospective_monitoring;
  const showHistoryPreview = (validHistory?.event_count ?? 0) > 0;

  if (!current || current.status === "empty" || state === "no_event_available") {
    return (
      <EmptyState
        title="No monitored event is currently available."
        message="The dashboard API is reachable, but the exported dashboard artifacts do not identify an active, forecasted, or settled monitored Grand Prix yet."
      />
    );
  }

  return (
    <div className="space-y-6">
      {isLegacy ? <LegacyWarning /> : null}
      <section className="overflow-hidden rounded-lg border border-apex-border bg-apex-panel shadow-panel">
        <div className="border-b border-apex-border bg-apex-panelSoft/65 p-5 md:p-6">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
            <div className="min-w-0">
              <p className="text-sm font-semibold uppercase tracking-[0.18em] text-apex-muted">
                {formatText(identity?.season)}
              </p>
              <h2 className="mt-2 break-words text-3xl font-semibold text-apex-text md:text-5xl">
                {formatText(identity?.event)}
              </h2>
              <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-300">
                {lifecycle?.reason
                  ? humanizeToken(lifecycle.reason)
                  : lifecycleExplanation(state)}
              </p>
            </div>
            <div className="flex shrink-0 flex-col gap-3">
              <LifecycleBadge state={state} label={lifecycle?.display_label} />
              <div className="rounded-md border border-apex-border bg-apex-bg/45 p-3 text-sm">
                <p className="text-apex-muted">Event order</p>
                <p className="mt-1 font-mono text-lg text-apex-text">
                  {formatInteger(identity?.event_order)}
                </p>
              </div>
            </div>
          </div>
        </div>
        <div className="grid gap-4 p-5 md:grid-cols-3 md:p-6">
          <KpiCard
            label="Protocol"
            value={formatText(protocol?.protocol_name)}
            hint={formatText(protocol?.protocol_fingerprint)}
          />
          <KpiCard
            label="Forecast checkpoint"
            value={formatText(kpis?.forecast_checkpoint)}
            hint="Latest exported checkpoint"
          />
          <KpiCard
            label="Artifact generated"
            value={formatDateTime(current.generated_at_utc)}
            hint="Frontend preserves export timestamp"
          />
        </div>
      </section>

      {state === "blocked" ? (
        <section className="rounded-lg border border-rose-300/40 bg-rose-300/10 p-5">
          <h2 className="text-lg font-semibold text-apex-text">Forecast generation is blocked</h2>
          <p className="mt-2 text-sm leading-6 text-rose-50">
            {preflight?.status
              ? `Preflight status: ${humanizeToken(preflight.status)}.`
              : "Preflight did not approve this event for forecasting."}{" "}
            {preflight?.next_required_command
              ? `Next operator action: ${preflight.next_required_command}.`
              : "Review the exported monitoring artifacts before continuing."}
          </p>
        </section>
      ) : null}

      {state === "ready_to_forecast" ? (
        <section className="rounded-lg border border-emerald-300/40 bg-emerald-300/10 p-5">
          <h2 className="text-lg font-semibold text-apex-text">Ready in the operator workflow</h2>
          <p className="mt-2 text-sm leading-6 text-emerald-50">
            Preflight has marked this event ready to forecast. The website remains read-only and
            does not execute the forecast command.
          </p>
        </section>
      ) : null}

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
        <StatusCard
          title="Monitoring Protocol"
          status={protocol?.protocol_name}
          items={[
            { label: "Version", value: protocol?.protocol_version },
            { label: "Checkpoint", value: protocol?.checkpoint },
            { label: "Policy", value: protocol?.policy_recommendation }
          ]}
        />
        <StatusCard
          title="Registry Lineage"
          status={lineage?.event_order_lineage_status}
          tone={lineage?.eligible_for_valid_prospective_evidence ? "good" : "warn"}
          items={[
            {
              label: "Prospective evidence",
              value: lineage?.eligible_for_valid_prospective_evidence
            },
            { label: "Legacy", value: lineage?.legacy_noncanonical },
            { label: "Action", value: lineage?.reconciliation_action }
          ]}
        />
        <StatusCard
          title="Preflight"
          status={preflight?.status}
          tone={preflight?.forecast_allowed ? "good" : "danger"}
          detail={preflight?.next_required_command}
          items={[
            { label: "Allowed", value: preflight?.forecast_allowed },
            { label: "Blocking", value: preflight?.blocking_check_count },
            { label: "Warnings", value: preflight?.warning_check_count }
          ]}
        />
        <StatusCard
          title="Forecast"
          status={forecast?.available ? "Available" : "Not available"}
          tone={forecast?.available ? "good" : "neutral"}
          items={[
            { label: "Drivers", value: forecast?.forecasted_driver_count },
            { label: "Checkpoint", value: forecast?.checkpoint },
            { label: "Generated", value: formatDateTime(forecast?.forecast_created_at_utc) }
          ]}
        />
        <StatusCard
          title="Settlement"
          status={settlement?.available ? "Available" : "Not available"}
          tone={settlement?.settlement_valid ? "good" : "neutral"}
          items={[
            { label: "Valid", value: settlement?.settlement_valid },
            { label: "Scored", value: settlement?.scored_driver_count },
            { label: "Settled", value: formatDateTime(settlement?.settled_at_utc) }
          ]}
        />
      </section>

      <SessionStatusRow sessions={sessions} />

      <section className="grid gap-4 lg:grid-cols-[1fr_auto] lg:items-start">
        <div>
          {forecastRows.length > 0 ? (
            <ForecastLeaderboard rows={forecastRows} compact />
          ) : (
            <section className="rounded-lg border border-apex-border bg-apex-panel/85 p-5">
              <h2 className="text-lg font-semibold text-apex-text">Forecast Preview</h2>
              <p className="mt-2 text-sm leading-6 text-slate-300">
                No forecast leaderboard rows are available in the current dashboard artifact.
              </p>
            </section>
          )}
        </div>
        <div className="grid gap-3 lg:w-56">
          <Link
            href="/forecast"
            className="inline-flex justify-center rounded-md border border-apex-accent/50 bg-apex-accent/10 px-4 py-2 text-sm font-semibold text-apex-text transition hover:bg-apex-accent/15"
          >
            Open forecast
          </Link>
          <Link
            href="/practice"
            className="inline-flex justify-center rounded-md border border-apex-border bg-apex-panelSoft px-4 py-2 text-sm font-semibold text-apex-text transition hover:border-apex-accent/50"
          >
            Open practice status
          </Link>
        </div>
      </section>

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-6">
        <KpiCard label="Forecasted drivers" value={formatInteger(kpis?.forecasted_driver_count)} />
        <KpiCard label="Predicted pole" value={formatText(kpis?.predicted_pole_driver)} />
        <KpiCard label="Checkpoint" value={formatText(kpis?.forecast_checkpoint)} />
        <KpiCard
          label="Intervals available"
          value={formatPercent(kpis?.interval_availability_rate)}
        />
        <KpiCard label="Actual pole" value={formatText(kpis?.actual_pole_driver)} />
        <KpiCard label="Settlement MAE" value={formatSeconds(kpis?.settlement_mae_gap_sec)} />
      </section>

      {showSettlementPreview || showHistoryPreview ? (
        <section className="grid gap-4 lg:grid-cols-2">
          {showSettlementPreview ? (
            <section className="rounded-lg border border-apex-border bg-apex-panel/85 p-5">
              <h2 className="text-lg font-semibold text-apex-text">Settlement Preview</h2>
              <p className="mt-2 text-sm leading-6 text-slate-300">
                Latest settled MAE: {formatSeconds(settlementMetrics?.mae_gap_sec)}. Scored drivers:{" "}
                {formatInteger(settlementMetrics?.scored_driver_count)}.
              </p>
              <Link
                href="/settlement"
                className="mt-4 inline-flex rounded-md border border-apex-accent/50 bg-apex-accent/10 px-4 py-2 text-sm font-semibold text-apex-text transition hover:bg-apex-accent/15"
              >
                Open settlement
              </Link>
            </section>
          ) : null}
          {showHistoryPreview ? (
            <section className="rounded-lg border border-apex-border bg-apex-panel/85 p-5">
              <h2 className="text-lg font-semibold text-apex-text">Monitoring History Preview</h2>
              <p className="mt-2 text-sm leading-6 text-slate-300">
                Valid prospective events: {formatInteger(validHistory?.event_count)}. Settled
                events: {formatInteger(validHistory?.settled_event_count)}.
              </p>
              <Link
                href="/monitoring-history"
                className="mt-4 inline-flex rounded-md border border-apex-accent/50 bg-apex-accent/10 px-4 py-2 text-sm font-semibold text-apex-text transition hover:bg-apex-accent/15"
              >
                Open monitoring history
              </Link>
            </section>
          ) : null}
        </section>
      ) : null}

      <section className="rounded-lg border border-apex-border bg-apex-panel/85 p-5">
        <div className="grid gap-4 md:grid-cols-[1fr_auto] md:items-center">
          <div>
            <h2 className="text-lg font-semibold text-apex-text">Data Trust</h2>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-300">
              Forecasts are generated from pre-qualifying evidence and exported as immutable
              dashboard artifacts. This interface is a read-only consumer; availability depends on
              the monitored-event workflow, and uncertainty fields may remain unavailable until the
              artifact contract marks them present.
            </p>
          </div>
          <Link
            href="/methodology"
            className="inline-flex justify-center rounded-md border border-apex-accent/50 bg-apex-accent/10 px-4 py-2 text-sm font-semibold text-apex-text transition hover:bg-apex-accent/15"
          >
            Methodology
          </Link>
        </div>
      </section>
    </div>
  );
}

function lifecycleExplanation(state: LifecycleState): string {
  const descriptions: Record<LifecycleState, string> = {
    no_event_available: "No monitored event is available in the exported dashboard artifacts.",
    practice_in_progress: "Practice artifacts exist, but the event is not yet ready to forecast.",
    ready_to_forecast: "Preflight approved this event for forecast generation.",
    forecast_available: "A qualifying forecast artifact is available.",
    awaiting_qualifying_targets: "A forecast exists and qualifying target ingestion is pending.",
    settled: "Forecast and qualifying targets have been settled.",
    blocked: "Preflight checks blocked forecast generation.",
    legacy_descriptive_only: "This event is descriptive legacy context only."
  };
  return descriptions[state];
}
