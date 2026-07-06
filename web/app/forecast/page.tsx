import { AppShell } from "@/components/app-shell";
import { ForecastPageView } from "@/components/forecast-page";
import { loadForecastPageData } from "@/lib/api";

export default async function ForecastPage() {
  const data = await loadForecastPageData();
  return (
    <AppShell health={data.health} generatedAt={data.forecast?.generated_at_utc ?? null}>
      <ForecastPageView data={data} />
    </AppShell>
  );
}
