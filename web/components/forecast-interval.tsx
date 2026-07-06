import { formatSignedGap } from "@/lib/formatters";

export function ForecastInterval({
  available,
  lower,
  upper
}: {
  available?: boolean | null;
  lower?: number | null;
  upper?: number | null;
}) {
  if (!available || typeof lower !== "number" || typeof upper !== "number") {
    return (
      <span className="inline-flex rounded-full border border-slate-500/40 bg-slate-400/10 px-2 py-1 text-xs font-medium text-slate-300">
        Interval not available
      </span>
    );
  }

  return (
    <span className="font-mono text-sm text-slate-100">
      {formatSignedGap(lower)} to {formatSignedGap(upper)}
    </span>
  );
}
