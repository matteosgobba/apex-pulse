import { formatInteger } from "@/lib/formatters";

export function PositionDelta({
  predicted,
  actual
}: {
  predicted?: number | null;
  actual?: number | null;
}) {
  if (typeof predicted !== "number" || typeof actual !== "number") {
    return <span className="text-slate-400">Not available</span>;
  }
  const delta = actual - predicted;
  if (delta === 0) {
    return <span className="text-slate-100">Matched prediction</span>;
  }
  const magnitude = Math.abs(delta);
  const direction = delta > 0 ? "worse" : "better";
  const tone = delta > 0 ? "text-amber-100" : "text-emerald-100";
  return (
    <span className={tone}>
      {formatInteger(magnitude)} {direction}
    </span>
  );
}
