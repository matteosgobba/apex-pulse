import type { UnforecastedActualEntrant } from "@/lib/dashboard-types";

export function CoverageNotice({
  coverage,
  entrants
}: {
  coverage: string | null;
  entrants: UnforecastedActualEntrant[];
}) {
  if (!coverage && entrants.length === 0) {
    return null;
  }
  const names = entrants
    .map((entrant) => entrant.driver_code ?? entrant.driver)
    .filter(Boolean)
    .join(", ");
  return (
    <aside
      aria-label="Forecast coverage"
      className="status-warning rounded-2xl border p-4 text-sm"
    >
      <p className="font-semibold">
        Forecast coverage: {coverage ?? "partial"}
      </p>
      <p className="mt-1 leading-6">
        {names
          ? `${names} appeared in the official result but was not included in the original pre-qualifying forecast. No retrospective prediction has been added.`
          : "Only entrants included in the original forecast are evaluated."}
      </p>
    </aside>
  );
}
