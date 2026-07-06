import { AppShell } from "@/components/app-shell";
import { PracticePageView } from "@/components/practice-page";
import { loadPracticePageData } from "@/lib/api";

export default async function PracticePage() {
  const data = await loadPracticePageData();
  return (
    <AppShell health={data.health} generatedAt={data.practiceStatus?.generated_at_utc ?? null}>
      <PracticePageView data={data} />
    </AppShell>
  );
}
