import { TeamMark } from "@/components/team-mark";
import type {
  LegacyDescriptiveRecord,
  MonitoringHistoryEvent,
  MonitoringHistoryPageData
} from "@/lib/dashboard-types";
import { formatSignedGap } from "@/lib/formatters";
import {
  positionDeltaLabel,
  predictionCheckpointLabel,
  publicLifecycleLabel
} from "@/lib/public-view-model";
import { getTeamIdentity } from "@/lib/team-identity";

export function MonitoringHistoryPageView({ data }: { data: MonitoringHistoryPageData }) {
  if (data.error || !data.historicalMonitoring) {
    return (
      <HistoryEmpty
        title="Prediction history unavailable"
        detail="The exported monitoring history could not be loaded."
      />
    );
  }
  const history = data.historicalMonitoring.data;
  const prospective = history.valid_prospective_monitoring;
  const events = prospective?.events ?? [];
  const legacy = history.legacy_descriptive_records ?? [];
  const backtest = history.backtest_context;
  const aggregate = prospective?.aggregate_metrics;
  const aggregateMae =
    aggregate && "mae_gap_sec" in aggregate
      ? aggregate.mae_gap_sec
      : aggregate?.available && "value" in aggregate && aggregate.value
        ? aggregate.value.mae_gap_sec
        : null;

  return (
    <div className="space-y-14">
      <section className="rounded-[2rem] border border-white/10 bg-apex-ink px-6 py-10 text-apex-onStrong shadow-hero sm:px-10 sm:py-14">
        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-apex-accent">
          Prediction history
        </p>
        <h1 className="mt-3 max-w-3xl text-4xl font-semibold tracking-tight sm:text-5xl">
          Preserved forecasts across race weekends.
        </h1>
        <p className="mt-4 max-w-2xl text-sm leading-7 text-apex-onStrongMuted">
          Prospective predictions are kept separate from legacy descriptive records and historical
          backtests, so public performance claims use the right evidence.
        </p>
        <div className="mt-8 flex flex-wrap gap-3 text-sm">
          <HistoryStat label="Prospective events" value={prospective?.event_count} />
          <HistoryStat label="Settled" value={prospective?.settled_event_count} />
          <HistoryStat
            label="Aggregate gap MAE"
            value={typeof aggregateMae === "number" ? `${aggregateMae.toFixed(3)}s` : null}
          />
        </div>
      </section>

      <section aria-labelledby="past-predictions-title">
        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-apex-accent">
          Valid prospective evidence
        </p>
        <h2 id="past-predictions-title" className="mt-2 text-3xl font-semibold text-apex-text">
          Past predictions
        </h2>
        <p className="mt-2 max-w-2xl text-sm leading-6 text-apex-secondary">
          Open an event to inspect its preserved forecast and, where available, official comparison.
        </p>
        {events.length > 0 ? (
          <div className="mt-6 grid gap-4">
            {events.map((event) => (
              <PredictionHistoryEvent
                key={`${event.event_identity?.season}-${event.event_identity?.event_slug}`}
                event={event}
              />
            ))}
          </div>
        ) : (
          <HistoryEmpty
            title="No prospective predictions yet"
            detail="No eligible event records are present in this export."
          />
        )}
      </section>

      {legacy.length > 0 ? <TechnicalArchive records={legacy} /> : null}

      <section className="rounded-3xl border border-apex-border bg-apex-panel p-6 shadow-card">
        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-apex-muted">
          Historical backtest context
        </p>
        <h2 className="mt-2 text-2xl font-semibold text-apex-text">Model development evidence</h2>
        <p className="mt-3 max-w-3xl text-sm leading-6 text-apex-secondary">
          Backtests simulate earlier race weekends for model development. They are useful, but are
          not counted as prospective public predictions.
        </p>
        <div className="mt-5 flex flex-wrap gap-3 text-sm text-apex-secondary">
          <span className="rounded-full bg-apex-surface px-3 py-1.5">
            {backtest?.n_events ?? "—"} historical events
          </span>
          <span className="rounded-full bg-apex-surface px-3 py-1.5">
            {backtest?.n_folds_successful ?? "—"} successful folds
          </span>
          <span className="rounded-full bg-apex-surface px-3 py-1.5">
            Walk-forward evaluation
          </span>
        </div>
      </section>
    </div>
  );
}

function PredictionHistoryEvent({ event }: { event: MonitoringHistoryEvent }) {
  const lifecycle = event.lifecycle_state ?? "no_event_available";
  const forecastRows = event.forecast_rows ?? [];
  const comparisonRows = event.comparison_rows ?? [];
  const metrics = event.metrics;
  return (
    <details className="group overflow-hidden rounded-3xl border border-apex-border bg-apex-panel shadow-card">
      <summary className="cursor-pointer list-none p-5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-apex-accent sm:p-6">
        <div className="flex flex-wrap items-center justify-between gap-5">
          <div>
            <p className="text-sm text-apex-muted">{event.event_identity?.season ?? "—"}</p>
            <h3 className="mt-1 text-2xl font-semibold text-apex-text">
              {event.event_identity?.event ?? "Unknown event"}
            </h3>
            <div className="mt-3 flex flex-wrap gap-2 text-xs text-apex-secondary">
              <span className="rounded-full bg-apex-surface px-3 py-1.5">
                {publicLifecycleLabel(lifecycle)}
              </span>
              <span className="rounded-full bg-apex-surface px-3 py-1.5">
                {event.forecast_checkpoint
                  ? predictionCheckpointLabel(event.forecast_checkpoint)
                  : "Checkpoint unavailable"}
              </span>
              <span className="rounded-full bg-apex-surface px-3 py-1.5">
                Coverage {event.forecast_coverage ?? "unavailable"}
              </span>
            </div>
          </div>
          <div className="text-right">
            <p className="text-xs text-apex-muted">Gap MAE</p>
            <p className="mt-1 text-2xl font-semibold text-apex-text">
              {typeof metrics?.mae_gap_sec === "number"
                ? `${metrics.mae_gap_sec.toFixed(3)}s`
                : "Pending"}
            </p>
            <p className="mt-2 text-xs font-semibold text-apex-accent group-open:hidden">
              Open event +
            </p>
          </div>
        </div>
      </summary>
      <div className="border-t border-apex-border bg-apex-bg p-5 sm:p-6">
        {forecastRows.length > 0 ? (
          <div>
            <h4 className="text-lg font-semibold text-apex-text">Preserved prediction</h4>
            <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {forecastRows.map((row) => {
                const team = getTeamIdentity(row.team_key, row.team);
                const actual = comparisonRows.find(
                  (comparison) =>
                    (comparison.driver_code ?? comparison.driver) ===
                    (row.driver_code ?? row.driver)
                );
                const delta = positionDeltaLabel(row.predicted_position, actual?.actual_position);
                return (
                  <div
                    key={`${row.driver_code ?? row.driver}-${row.predicted_position}`}
                    className="flex items-center gap-3 rounded-2xl border border-apex-border bg-apex-panel p-3"
                  >
                    <span className="w-7 text-lg font-semibold text-apex-text">
                      {row.predicted_position ?? "—"}
                    </span>
                    <TeamMark team={team} size="sm" />
                    <div className="min-w-0 flex-1">
                      <p className="font-semibold text-apex-text">
                        {row.driver_code ?? row.driver ?? "—"}
                      </p>
                      <p className="truncate text-xs text-apex-muted">{team.displayName}</p>
                    </div>
                    <div className="text-right text-xs">
                      <p className="font-semibold text-apex-text">
                        {formatSignedGap(row.predicted_gap_to_pole_sec)}
                      </p>
                      {actual ? <p className="mt-1 text-apex-muted">{delta.shortLabel}</p> : null}
                    </div>
                  </div>
                );
              })}
            </div>
            {!event.settled ? (
              <p className="mt-4 text-sm text-apex-secondary">
                Official qualifying comparison is not available for this event.
              </p>
            ) : null}
          </div>
        ) : (
          <p className="text-sm text-apex-secondary">Detailed forecast rows are unavailable.</p>
        )}
      </div>
    </details>
  );
}

function TechnicalArchive({ records }: { records: LegacyDescriptiveRecord[] }) {
  return (
    <details className="rounded-3xl border border-apex-border bg-apex-panel">
      <summary className="cursor-pointer list-none p-6 font-semibold text-apex-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-apex-accent">
        Technical archive · {records.length} legacy descriptive records
      </summary>
      <div className="grid gap-3 border-t border-apex-border p-6 sm:grid-cols-2">
        {records.map((record) => (
          <article
            key={`${record.event_identity?.season}-${record.event_identity?.event_slug}`}
            className="rounded-2xl bg-apex-surface p-4"
          >
            <h3 className="font-semibold text-apex-text">
              {record.event_identity?.season} {record.event_identity?.event}
            </h3>
            <p className="mt-2 text-xs leading-5 text-apex-secondary">
              Descriptive only · excluded from prospective performance claims.
            </p>
          </article>
        ))}
      </div>
    </details>
  );
}

function HistoryStat({ label, value }: { label: string; value: string | number | null | undefined }) {
  return (
    <span className="rounded-full bg-white/10 px-4 py-2 text-apex-onStrongMuted">
      {label}: <strong className="text-apex-onStrong">{value ?? "—"}</strong>
    </span>
  );
}

function HistoryEmpty({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="mt-6 rounded-3xl border border-dashed border-apex-border bg-apex-panel p-8 text-center">
      <h2 className="text-xl font-semibold text-apex-text">{title}</h2>
      <p className="mt-2 text-sm text-apex-secondary">{detail}</p>
    </div>
  );
}
