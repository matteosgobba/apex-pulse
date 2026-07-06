import { BacktestContextPanel } from "@/components/backtest-context-panel";
import { ErrorState } from "@/components/error-state";
import { LegacyRecordsPanel } from "@/components/legacy-records-panel";
import { MonitoringEventTable } from "@/components/monitoring-event-table";
import { MonitoringSummaryCards } from "@/components/monitoring-summary-cards";
import type { MonitoringHistoryPageData } from "@/lib/dashboard-types";

export function MonitoringHistoryPageView({ data }: { data: MonitoringHistoryPageData }) {
  if (data.error) {
    return <ErrorState error={data.error} />;
  }

  const history = data.historicalMonitoring?.data;
  const prospective = history?.valid_prospective_monitoring;
  const legacyRecords = history?.legacy_descriptive_records ?? [];

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-apex-border bg-apex-panel p-6 shadow-panel">
        <p className="text-sm font-semibold uppercase tracking-[0.18em] text-apex-muted">
          Monitoring History
        </p>
        <h1 className="mt-2 max-w-4xl text-3xl font-semibold text-apex-text md:text-5xl">
          Prospective evidence, legacy records, and backtests stay separate.
        </h1>
        <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-300">
          This page intentionally separates valid prospective monitoring evidence from
          legacy/noncanonical descriptive artifacts and historical backtest context.
        </p>
      </section>

      <section className="space-y-4">
        <div>
          <p className="text-sm font-semibold uppercase tracking-[0.16em] text-apex-muted">
            Prospective Monitoring
          </p>
          <h2 className="mt-1 text-2xl font-semibold text-apex-text">
            Valid prospective evidence only
          </h2>
        </div>
        <MonitoringSummaryCards monitoring={prospective} />
        <MonitoringEventTable events={prospective?.events ?? []} />
      </section>

      <LegacyRecordsPanel records={legacyRecords} />

      <BacktestContextPanel
        context={history?.backtest_context}
        modelSummary={data.modelSummary?.data}
      />
    </div>
  );
}
