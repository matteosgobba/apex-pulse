import { evaluateFreshness } from "@/lib/freshness";

export function StaleDataBanner({
  generatedAt
}: {
  generatedAt: string | null | undefined;
}) {
  const freshness = evaluateFreshness(generatedAt);
  if (freshness.state !== "stale") {
    return null;
  }

  return (
    <section className="border-b border-amber-300/35 bg-amber-300/10 px-4 py-3 lg:px-8">
      <div className="mx-auto max-w-7xl">
        <p className="text-sm font-semibold text-amber-100">Dashboard data may be stale.</p>
        <p className="mt-1 text-sm leading-6 text-amber-50/90">
          This interface refreshes after validated artifacts are exported by the operator workflow.
          It is not live telemetry. Latest artifact: {freshness.relativeLabel}.
        </p>
      </div>
    </section>
  );
}
