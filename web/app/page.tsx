import { AppShell } from "@/components/app-shell";
import { CurrentEventPageView } from "@/components/current-event-page";
import { loadCurrentEventPageData } from "@/lib/api";

export default async function CurrentEventPage() {
  const data = await loadCurrentEventPageData();
  return (
    <AppShell
      health={data.health}
      generatedAt={data.currentEvent?.generated_at_utc ?? null}
      autoRefresh
    >
      <CurrentEventPageView data={data} />
    </AppShell>
  );
}
