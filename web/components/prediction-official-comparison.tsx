import { CoverageNotice } from "@/components/coverage-notice";
import { TeamMark } from "@/components/team-mark";
import { formatSignedGap } from "@/lib/formatters";
import {
  positionDeltaLabel,
  type PublicComparisonRow
} from "@/lib/public-view-model";
import type { UnforecastedActualEntrant } from "@/lib/dashboard-types";

export function PredictionOfficialComparison({
  rows,
  coverage,
  unforecastedEntrants
}: {
  rows: PublicComparisonRow[];
  coverage: string | null;
  unforecastedEntrants: UnforecastedActualEntrant[];
}) {
  if (rows.length === 0) {
    return null;
  }
  return (
    <section id="comparison" aria-labelledby="comparison-title">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-apex-accent">
            Prediction versus result
          </p>
          <h2 id="comparison-title" className="mt-2 text-3xl font-semibold tracking-tight text-apex-text">
            How the forecast compared
          </h2>
        </div>
        <p className="max-w-md text-xs leading-5 text-slate-500">
          “Over” means the model predicted a better finishing position than achieved. “Under”
          means the driver finished higher than predicted.
        </p>
      </div>
      <div className="mt-6">
        <CoverageNotice coverage={coverage} entrants={unforecastedEntrants} />
      </div>
      <div className="mt-4 overflow-hidden rounded-3xl border border-apex-border bg-white shadow-card">
        <div className="hidden grid-cols-[minmax(180px,1fr)_90px_90px_130px_110px_110px_100px] gap-3 border-b border-apex-border bg-apex-surface px-5 py-3 text-xs font-semibold uppercase tracking-wide text-slate-500 lg:grid">
          <span>Driver</span>
          <span>Predicted</span>
          <span>Official</span>
          <span>Position read</span>
          <span>Pred. gap</span>
          <span>Official gap</span>
          <span>Gap error</span>
        </div>
        <ol aria-label="Prediction and official result comparison">
          {rows.map((row) => {
            const delta = positionDeltaLabel(row.predictedPosition, row.actualPosition);
            return (
              <li
                key={`${row.driverCode}-${row.predictedPosition}`}
                className="border-b border-apex-border/70 p-4 last:border-b-0 lg:grid lg:grid-cols-[minmax(180px,1fr)_90px_90px_130px_110px_110px_100px] lg:items-center lg:gap-3 lg:px-5"
              >
                <div className="flex min-w-0 items-center gap-3">
                  <TeamMark team={row.team} size="sm" />
                  <div className="min-w-0">
                    <p className="font-semibold text-apex-text">{row.driverCode}</p>
                    <p className="truncate text-xs text-slate-500">{row.team.displayName}</p>
                  </div>
                </div>
                <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:contents">
                  <Value label="Predicted" value={row.predictedPosition} />
                  <Value label="Official" value={row.actualPosition} />
                  <div>
                    <span className="text-[10px] font-medium uppercase tracking-wide text-slate-500 lg:hidden">
                      Position read
                    </span>
                    <p
                      aria-label={delta.label}
                      className={`mt-1 text-sm font-semibold lg:mt-0 ${
                        delta.direction === "exact"
                          ? "text-emerald-700"
                          : delta.direction === "unavailable"
                            ? "text-slate-400"
                            : "text-slate-700"
                      }`}
                    >
                      {delta.shortLabel}
                    </p>
                  </div>
                  <Value label="Pred. gap" value={formatSignedGap(row.predictedGapSec)} />
                  <Value label="Official gap" value={formatSignedGap(row.actualGapSec)} />
                  <Value
                    label="Gap error"
                    value={
                      row.absoluteGapErrorSec === null
                        ? "—"
                        : `${row.absoluteGapErrorSec.toFixed(3)}s`
                    }
                  />
                </div>
              </li>
            );
          })}
        </ol>
      </div>
    </section>
  );
}

function Value({ label, value }: { label: string; value: string | number | null }) {
  return (
    <div>
      <span className="text-[10px] font-medium uppercase tracking-wide text-slate-500 lg:hidden">
        {label}
      </span>
      <p className="mt-1 text-sm font-semibold text-apex-text lg:mt-0">{value ?? "—"}</p>
    </div>
  );
}
