import { KpiCard } from "@/components/kpi-card";
import type { ForecastData } from "@/lib/dashboard-types";
import { formatDateTime, formatInteger, formatPercent, formatText, humanizeToken } from "@/lib/formatters";

export function ForecastSummaryPanel({ forecast }: { forecast: ForecastData | null | undefined }) {
  const metadata = forecast?.forecast_metadata;
  const summary = forecast?.summary;
  const method = metadata?.candidate_or_policy_identity;
  const methodText = [method?.model_name, method?.feature_group, method?.temporal_weighting_policy]
    .filter(Boolean)
    .map((value) => humanizeToken(value))
    .join(" / ");

  return (
    <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
      <KpiCard label="Checkpoint" value={formatText(metadata?.checkpoint ?? summary?.checkpoint)} />
      <KpiCard
        label="Forecast timestamp"
        value={formatDateTime(metadata?.forecast_timestamp)}
        hint="Exported snapshot time"
      />
      <KpiCard
        label="Qualifying-eligible drivers"
        value={formatInteger(summary?.forecasted_driver_count)}
      />
      <KpiCard
        label="Forecast-only audit rows"
        value={formatInteger(summary?.forecast_only_driver_count)}
      />
      <KpiCard
        label="Interval coverage"
        value={formatPercent(summary?.interval_availability_rate)}
        hint="Availability, not confidence"
      />
      <KpiCard label="Protocol" value={formatText(metadata?.protocol_name)} />
      <KpiCard label="Preflight" value={humanizeToken(metadata?.preflight_status)} />
      <KpiCard label="Method" value={methodText || "Not available"} />
      <KpiCard label="Prediction target" value={formatText(metadata?.prediction_target)} />
    </section>
  );
}
