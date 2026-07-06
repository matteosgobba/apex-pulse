import { AppShell } from "@/components/app-shell";
import { SettlementPageView } from "@/components/settlement-page";
import { loadSettlementPageData } from "@/lib/api";

export default async function SettlementPage() {
  const data = await loadSettlementPageData();
  return (
    <AppShell health={data.health} generatedAt={data.settlement?.generated_at_utc ?? null}>
      <SettlementPageView data={data} />
    </AppShell>
  );
}
