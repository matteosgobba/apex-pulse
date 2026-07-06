import type { ForecastLeaderboardRow, SettlementDriverComparisonRow } from "@/lib/dashboard-types";
import { formatSignedGap } from "@/lib/formatters";

export function IntervalOutcome({
  forecast,
  settlement
}: {
  forecast?: ForecastLeaderboardRow | null;
  settlement: SettlementDriverComparisonRow;
}) {
  const lower = forecast?.interval_lower_sec;
  const upper = forecast?.interval_upper_sec;
  const actual = settlement.actual_gap_to_pole_sec;
  if (
    !forecast?.interval_available ||
    typeof lower !== "number" ||
    typeof upper !== "number" ||
    typeof actual !== "number"
  ) {
    return <span className="text-slate-400">Not available</span>;
  }
  const covered = actual >= lower && actual <= upper;
  return (
    <span className={covered ? "text-emerald-100" : "text-amber-100"}>
      {covered ? "Covered" : "Outside"} ({formatSignedGap(lower)} to {formatSignedGap(upper)})
    </span>
  );
}
