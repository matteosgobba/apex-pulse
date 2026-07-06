import { KpiCard } from "@/components/kpi-card";
import type { ValidProspectiveMonitoring } from "@/lib/dashboard-types";
import { formatInteger, formatSeconds } from "@/lib/formatters";

export function MonitoringSummaryCards({
  monitoring
}: {
  monitoring: ValidProspectiveMonitoring | null | undefined;
}) {
  const aggregate = monitoring?.aggregate_metrics;
  const aggregateValue = aggregate?.available ? aggregate.value : null;
  const mae =
    aggregateValue && typeof aggregateValue.mae_gap_sec === "number"
      ? aggregateValue.mae_gap_sec
      : null;
  return (
    <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <KpiCard label="Valid events" value={formatInteger(monitoring?.event_count)} />
      <KpiCard label="Forecasted events" value={formatInteger(monitoring?.forecasted_event_count)} />
      <KpiCard label="Settled events" value={formatInteger(monitoring?.settled_event_count)} />
      <KpiCard label="Aggregate MAE" value={formatSeconds(mae)} />
    </section>
  );
}
