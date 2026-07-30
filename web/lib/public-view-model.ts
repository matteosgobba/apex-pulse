import { dashboardRows } from "@/lib/dashboard-collections";
import type {
  CurrentEventPageData,
  EventSchedule,
  ForecastLeaderboardRow,
  LifecycleState,
  SessionStatus,
  SettlementDriverComparisonRow,
  UnforecastedActualEntrant
} from "@/lib/dashboard-types";
import { evaluateFreshness, type FreshnessStatus } from "@/lib/freshness";
import { getTeamIdentity, type TeamIdentity } from "@/lib/team-identity";

export interface PublicRankingRow {
  driverCode: string;
  driverName: string | null;
  team: TeamIdentity;
  predictedPosition: number | null;
  predictedGapSec: number | null;
  intervalLowerSec: number | null;
  intervalUpperSec: number | null;
}

export interface PublicComparisonRow extends PublicRankingRow {
  actualPosition: number | null;
  actualGapSec: number | null;
  absoluteGapErrorSec: number | null;
}

export interface PublicEventViewModel {
  available: boolean;
  season: number | null;
  eventName: string | null;
  eventOrder: number | null;
  location: string | null;
  circuit: string | null;
  lifecycle: LifecycleState;
  lifecycleLabel: string;
  lifecycleDetail: string;
  checkpoint: string | null;
  checkpointLabel: string;
  generatedAtUtc: string | null;
  freshness: FreshnessStatus;
  schedule: EventSchedule | null;
  sessions: SessionStatus[];
  ranking: PublicRankingRow[];
  comparison: PublicComparisonRow[];
  hasSettlement: boolean;
  metrics: {
    maeGapSec: number | null;
    rmseGapSec: number | null;
    meanAbsolutePositionError: number | null;
    top3Agreement: number | null;
    top5Agreement: number | null;
    top10Agreement: number | null;
    coverage: string | null;
    coveragePercentage: number | null;
  };
  unforecastedEntrants: UnforecastedActualEntrant[];
  technical: Array<{ label: string; value: string | number | boolean | null | undefined }>;
}

export function adaptCurrentEvent(
  page: CurrentEventPageData,
  options: { now?: Date } = {}
): PublicEventViewModel {
  const current = page.currentEvent;
  const currentData = current?.data;
  const identity = currentData?.event_identity;
  const lifecycle = currentData?.lifecycle?.state ?? "no_event_available";
  const schedule = currentData?.event_schedule ?? page.practiceStatus?.data.event_schedule ?? null;
  const forecastRows = dashboardRows(
    page.forecast?.data.qualifying_eligible_forecast_rows ?? page.forecast?.data.leaderboard
  );
  const comparisonRows = dashboardRows(
    page.settlement?.data.settlement_evaluable_rows ?? page.settlement?.data.driver_comparison
  ).filter((row) => row.included_in_metrics !== false);
  const summary = page.settlement?.data.summary_metrics;
  const checkpoint =
    page.forecast?.data.forecast_metadata?.checkpoint ??
    currentData?.forecast_status?.checkpoint ??
    null;
  const ranking = forecastRows.map(adaptRankingRow).sort(byPredictedPosition);
  const forecastsByDriver = new Map(
    forecastRows.map((row) => [row.driver_code ?? row.driver, row])
  );
  const comparison = comparisonRows
    .map((row) =>
      adaptComparisonRow(row, forecastsByDriver.get(row.driver_code ?? row.driver))
    )
    .sort(byPredictedPosition);
  const unforecastedEntrants =
    page.settlement?.data.unforecasted_actual_entrants ??
    summary?.unforecasted_actual_entrants ??
    currentData?.settlement_status?.unforecasted_actual_entrants ??
    [];

  return {
    available: Boolean(
      current &&
        current.status !== "empty" &&
        lifecycle !== "no_event_available" &&
        identity?.event
    ),
    season: identity?.season ?? null,
    eventName: identity?.event ?? null,
    eventOrder: identity?.event_order ?? null,
    location: schedule?.location ?? null,
    circuit: schedule?.circuit ?? null,
    lifecycle,
    lifecycleLabel: publicLifecycleLabel(lifecycle),
    lifecycleDetail: publicLifecycleDetail(
      lifecycle,
      summary?.evaluable_driver_count ?? summary?.scored_driver_count ?? comparison.length,
      summary?.actual_qualifying_driver_count ??
        currentData?.settlement_status?.actual_qualifying_driver_count ??
        null,
      unforecastedEntrants
    ),
    checkpoint,
    checkpointLabel: predictionCheckpointLabel(checkpoint),
    generatedAtUtc: current?.generated_at_utc ?? null,
    freshness: evaluateFreshness(current?.generated_at_utc, { now: options.now }),
    schedule,
    sessions: page.practiceStatus?.data.sessions ?? [],
    ranking,
    comparison,
    hasSettlement: comparison.length > 0,
    metrics: {
      maeGapSec: summary?.mae_gap_sec ?? null,
      rmseGapSec: summary?.rmse_gap_sec ?? null,
      meanAbsolutePositionError: summary?.mean_absolute_position_error ?? null,
      top3Agreement: summary?.top_3_agreement ?? null,
      top5Agreement: summary?.top_5_agreement ?? null,
      top10Agreement: summary?.top_10_agreement ?? null,
      coverage:
        summary?.forecast_coverage ?? currentData?.settlement_status?.forecast_coverage ?? null,
      coveragePercentage:
        summary?.forecast_coverage_percentage ??
        currentData?.settlement_status?.forecast_coverage_percentage ??
        null
    },
    unforecastedEntrants,
    technical: [
      { label: "Artifact lifecycle", value: lifecycle },
      { label: "Export timestamp", value: current?.generated_at_utc },
      { label: "Protocol version", value: currentData?.monitoring_protocol?.protocol_version },
      { label: "Checkpoint identifier", value: currentData?.monitoring_protocol?.checkpoint },
      {
        label: "Registry lineage",
        value: currentData?.registry_lineage?.event_order_lineage_status
      },
      { label: "Preflight status", value: currentData?.preflight?.status },
      {
        label: "Protocol fingerprint",
        value: page.forecast?.data.forecast_metadata?.protocol_fingerprint
      },
      {
        label: "Settlement valid",
        value: page.settlement?.data.settlement_metadata?.settlement_valid
      }
    ]
  };
}

export function positionDeltaLabel(
  predicted: number | null | undefined,
  actual: number | null | undefined
): { label: string; shortLabel: string; direction: "over" | "under" | "exact" | "unavailable" } {
  if (typeof predicted !== "number" || typeof actual !== "number") {
    return { label: "Not comparable", shortLabel: "—", direction: "unavailable" };
  }
  const delta = actual - predicted;
  if (delta === 0) {
    return { label: "Exact position", shortLabel: "Exact", direction: "exact" };
  }
  if (delta > 0) {
    return {
      label: `Model overpredicted by ${delta} ${delta === 1 ? "position" : "positions"}`,
      shortLabel: `↓ ${delta} over`,
      direction: "over"
    };
  }
  const magnitude = Math.abs(delta);
  return {
    label: `Model underpredicted by ${magnitude} ${magnitude === 1 ? "position" : "positions"}`,
    shortLabel: `↑ ${magnitude} under`,
    direction: "under"
  };
}

export function publicLifecycleLabel(lifecycle: LifecycleState): string {
  const labels: Record<LifecycleState, string> = {
    no_event_available: "Event unavailable",
    practice_in_progress: "Practice underway",
    ready_to_forecast: "Practice data ready",
    forecast_available: "Qualifying prediction available",
    awaiting_qualifying_targets: "Awaiting official qualifying result",
    settled: "Qualifying completed",
    settled_partial_coverage: "Qualifying completed",
    blocked: "Prediction unavailable",
    legacy_descriptive_only: "Legacy record"
  };
  return labels[lifecycle];
}

export function predictionCheckpointLabel(checkpoint: string | null | undefined): string {
  if (!checkpoint) {
    return "Prediction";
  }
  const session = checkpoint.replace(/^after_/i, "").toUpperCase();
  return `Prediction after ${session}`;
}

function publicLifecycleDetail(
  lifecycle: LifecycleState,
  evaluated: number | null,
  actual: number | null,
  missing: UnforecastedActualEntrant[]
): string {
  if (lifecycle === "settled_partial_coverage") {
    const denominator =
      typeof evaluated === "number" && typeof actual === "number"
        ? `${evaluated} of ${actual} entrants`
        : "available entrants";
    const count = missing.length || 1;
    return `Prediction evaluated on ${denominator}. ${count === 1 ? "One" : count} official ${
      count === 1 ? "entrant was" : "entrants were"
    } not included in the original pre-qualifying forecast.`;
  }
  if (lifecycle === "settled") {
    return "The preserved pre-qualifying prediction has been evaluated against the official result.";
  }
  if (lifecycle === "awaiting_qualifying_targets") {
    return "The pre-qualifying prediction is preserved while the official result is pending.";
  }
  if (lifecycle === "forecast_available") {
    return "The prediction was generated before qualifying from the available practice sessions.";
  }
  if (lifecycle === "practice_in_progress" || lifecycle === "ready_to_forecast") {
    return "Practice data is building toward the next pre-qualifying forecast.";
  }
  if (lifecycle === "blocked") {
    return "A safe public prediction is not available for this event.";
  }
  return "No current prospective prediction is available.";
}

function adaptRankingRow(row: ForecastLeaderboardRow): PublicRankingRow {
  const driverCode = row.driver_code ?? row.driver ?? "—";
  return {
    driverCode,
    driverName: row.driver && row.driver !== driverCode ? row.driver : null,
    team: getTeamIdentity(row.team_key, row.team),
    predictedPosition: row.predicted_position ?? null,
    predictedGapSec: row.predicted_gap_to_pole_sec ?? null,
    intervalLowerSec: row.interval_available ? (row.interval_lower_sec ?? null) : null,
    intervalUpperSec: row.interval_available ? (row.interval_upper_sec ?? null) : null
  };
}

function adaptComparisonRow(
  row: SettlementDriverComparisonRow,
  forecast?: ForecastLeaderboardRow
): PublicComparisonRow {
  const driverCode = row.driver_code ?? row.driver ?? "—";
  return {
    driverCode,
    driverName: row.driver && row.driver !== driverCode ? row.driver : null,
    team: getTeamIdentity(row.team_key ?? forecast?.team_key, row.team ?? forecast?.team),
    predictedPosition: row.predicted_position ?? null,
    predictedGapSec: row.predicted_gap_to_pole_sec ?? null,
    intervalLowerSec: null,
    intervalUpperSec: null,
    actualPosition: row.actual_position ?? null,
    actualGapSec: row.actual_gap_to_pole_sec ?? null,
    absoluteGapErrorSec: row.absolute_gap_error_sec ?? null
  };
}

function byPredictedPosition(
  left: { predictedPosition: number | null },
  right: { predictedPosition: number | null }
): number {
  return (
    (left.predictedPosition ?? Number.MAX_SAFE_INTEGER) -
    (right.predictedPosition ?? Number.MAX_SAFE_INTEGER)
  );
}
