import { AppShell } from "@/components/app-shell";
import { MonitoringHistoryPageView } from "@/components/monitoring-history-page";
import { loadMonitoringHistoryPageData } from "@/lib/api";

export default async function HistoryPage() {
  const data = await loadMonitoringHistoryPageData();
  return (
    <AppShell health={data.health} autoRefresh>
      <MonitoringHistoryPageView data={data} />
    </AppShell>
  );
}
