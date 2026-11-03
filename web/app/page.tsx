import { RacingHero } from "@/components/racing-hero";
import { AppShell } from "@/components/app-shell";
import { CurrentEventPageView } from "@/components/current-event-page";
import { loadCurrentEventPageData } from "@/lib/api";

export default async function CurrentEventPage() {
  const data = await loadCurrentEventPageData();
  return (
    <AppShell
      health={data.health}
      generatedAt={data.currentEvent?.generated_at_utc ?? null}
      landing
      autoRefresh
    >
      <RacingHero />
      <div className="section-space">
        <CurrentEventPageView data={data} primaryHeading={false} />
      </div>
    </AppShell>
  );
}
