import { KpiCard } from "@/components/kpi-card";
import type { SettlementData } from "@/lib/dashboard-types";
import { formatDecimal, formatInteger, formatPercent, formatSeconds } from "@/lib/formatters";

export function SettlementSummary({ settlement }: { settlement: SettlementData | null | undefined }) {
  const metrics = settlement?.summary_metrics;
  return (
    <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <KpiCard label="Forecasted drivers" value={formatInteger(metrics?.driver_count)} />
      <KpiCard label="Scored drivers" value={formatInteger(metrics?.scored_driver_count)} />
      <KpiCard label="MAE gap to pole" value={formatSeconds(metrics?.mae_gap_sec)} />
      <KpiCard label="RMSE gap to pole" value={formatSeconds(metrics?.rmse_gap_sec)} />
      <KpiCard
        label="Median abs gap error"
        value={formatSeconds(metrics?.median_absolute_gap_error_sec)}
      />
      <KpiCard
        label="Mean abs position error"
        value={formatDecimal(metrics?.mean_absolute_position_error)}
      />
      <KpiCard label="Top-3 agreement" value={formatPercent(metrics?.top_3_agreement)} />
      <KpiCard label="Top-5 agreement" value={formatPercent(metrics?.top_5_agreement)} />
      <KpiCard label="Top-10 agreement" value={formatPercent(metrics?.top_10_agreement)} />
      <KpiCard
        label="Interval coverage"
        value={formatPercent(intervalCoverage(settlement?.interval_diagnostics?.value))}
      />
      <KpiCard label="Mean interval width" value="Not available" />
      <KpiCard label="Actual pole" value={metrics?.actual_pole_driver ?? "Not available"} />
    </section>
  );
}

function intervalCoverage(value: Record<string, unknown> | null | undefined): number | null {
  if (!value) {
    return null;
  }
  const candidate = value.coverage_rate ?? value.interval_coverage ?? value.coverage;
  return typeof candidate === "number" ? candidate : null;
}
