import { TeamMark } from "@/components/team-mark";
import { formatSignedGap } from "@/lib/formatters";
import type { PublicRankingRow } from "@/lib/public-view-model";

export function ForecastRanking({ rows }: { rows: PublicRankingRow[] }) {
  return (
    <section id="forecast" aria-labelledby="forecast-title">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-apex-accent">
            Qualifying prediction
          </p>
          <h2 id="forecast-title" className="mt-2 text-3xl font-semibold tracking-tight text-apex-text">
            Predicted starting order
          </h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600">
            Ranked from the pre-qualifying forecast. Gaps are relative to the predicted pole lap.
          </p>
        </div>
        <span className="rounded-full bg-slate-100 px-3 py-1.5 text-xs font-medium text-slate-600">
          {rows.length} forecast drivers
        </span>
      </div>
      {rows.length === 0 ? (
        <div className="mt-6 rounded-3xl border border-dashed border-slate-300 bg-white p-8 text-center">
          <h3 className="text-lg font-semibold text-apex-text">Prediction unavailable</h3>
          <p className="mt-2 text-sm text-slate-600">
            No forecast rows are present in the current dashboard export.
          </p>
        </div>
      ) : (
        <div className="mt-6 overflow-hidden rounded-3xl border border-apex-border bg-white shadow-card">
          <div className="hidden grid-cols-[64px_minmax(220px,1fr)_160px_130px] gap-4 border-b border-apex-border bg-apex-surface px-6 py-3 text-xs font-semibold uppercase tracking-wide text-slate-500 md:grid">
            <span>Rank</span>
            <span>Driver</span>
            <span>Team</span>
            <span className="text-right">Predicted gap</span>
          </div>
          <ol aria-label="Predicted qualifying ranking">
            {rows.map((row) => (
              <li
                key={`${row.driverCode}-${row.predictedPosition}`}
                className="relative grid grid-cols-[48px_1fr_auto] items-center gap-3 border-b border-apex-border/70 px-4 py-4 last:border-b-0 md:grid-cols-[64px_minmax(220px,1fr)_160px_130px] md:gap-4 md:px-6"
              >
                <span
                  className="absolute inset-y-0 left-0 w-1"
                  style={{ backgroundColor: row.team.primary }}
                  aria-hidden="true"
                />
                <span className="text-2xl font-semibold text-apex-text">
                  {row.predictedPosition ?? "—"}
                </span>
                <div className="flex min-w-0 items-center gap-3">
                  <TeamMark team={row.team} />
                  <div className="min-w-0">
                    <p className="text-lg font-semibold text-apex-text">{row.driverCode}</p>
                    <p className="truncate text-xs text-slate-500 md:hidden">
                      {row.team.displayName}
                    </p>
                  </div>
                </div>
                <p className="hidden truncate text-sm text-slate-600 md:block">
                  {row.team.displayName}
                </p>
                <div className="text-right">
                  <p className="font-semibold text-apex-text">{formatSignedGap(row.predictedGapSec)}</p>
                  {row.intervalLowerSec !== null && row.intervalUpperSec !== null ? (
                    <p className="mt-1 text-xs text-slate-500">
                      {formatSignedGap(row.intervalLowerSec)}–{formatSignedGap(row.intervalUpperSec)}
                    </p>
                  ) : null}
                </div>
              </li>
            ))}
          </ol>
        </div>
      )}
    </section>
  );
}
