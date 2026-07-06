import { ForecastInterval } from "@/components/forecast-interval";
import { IntervalOutcome } from "@/components/interval-outcome";
import { PositionDelta } from "@/components/position-delta";
import { TableEmptyState } from "@/components/table-empty-state";
import type { ForecastLeaderboardRow, SettlementDriverComparisonRow } from "@/lib/dashboard-types";
import { formatInteger, formatSignedGap, formatText, humanizeToken } from "@/lib/formatters";
import { teamAccent } from "@/lib/team-accent";

export function SettlementComparisonTable({
  rows,
  forecastRows
}: {
  rows: SettlementDriverComparisonRow[];
  forecastRows: ForecastLeaderboardRow[];
}) {
  if (rows.length === 0) {
    return (
      <TableEmptyState
        title="No settlement rows are available."
        message="The settlement artifact exists, but it does not include per-driver comparison rows."
      />
    );
  }
  const forecastByDriver = new Map(
    forecastRows.map((row) => [row.driver_code ?? row.driver ?? "", row])
  );

  return (
    <section className="rounded-lg border border-apex-border bg-apex-panel/85 shadow-panel">
      <div className="border-b border-apex-border px-4 py-3">
        <h2 className="text-base font-semibold text-apex-text">Driver Comparison</h2>
        <p className="mt-1 text-sm text-apex-muted">
          Settlement rows preserve the exported artifact ordering.
        </p>
      </div>
      <div className="hidden overflow-x-auto md:block">
        <table className="min-w-full border-collapse text-sm">
          <thead className="bg-apex-panelSoft text-xs uppercase tracking-[0.12em] text-apex-muted">
            <tr>
              <Th align="right">Predicted position</Th>
              <Th align="right">Actual position</Th>
              <Th>Position delta</Th>
              <Th>Driver</Th>
              <Th>Team</Th>
              <Th align="right">Predicted gap to pole</Th>
              <Th align="right">Actual gap to pole</Th>
              <Th align="right">Absolute gap error</Th>
              <Th>Prediction interval</Th>
              <Th>Interval outcome</Th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const forecast = forecastByDriver.get(row.driver_code ?? row.driver ?? "");
              return (
                <tr
                  key={`${row.predicted_position}-${row.driver_code ?? row.driver}`}
                  className="border-t border-apex-border/80"
                >
                  <Td numeric>{formatInteger(row.predicted_position)}</Td>
                  <Td numeric>{formatInteger(row.actual_position)}</Td>
                  <Td>
                    <PositionDelta predicted={row.predicted_position} actual={row.actual_position} />
                  </Td>
                  <Td>
                    <div className="font-semibold text-apex-text">
                      {formatText(row.driver_code ?? row.driver)}
                    </div>
                    <div className="text-xs text-apex-muted">
                      {row.settlement_evaluable
                        ? "Included in metrics"
                        : humanizeToken(row.settlement_exclusion_reason)}
                    </div>
                  </Td>
                  <Td>
                    <TeamLabel forecast={forecast} />
                  </Td>
                  <Td numeric>{formatSignedGap(row.predicted_gap_to_pole_sec)}</Td>
                  <Td numeric>{formatSignedGap(row.actual_gap_to_pole_sec)}</Td>
                  <Td numeric>{formatSignedGap(row.absolute_gap_error_sec)}</Td>
                  <Td>
                    <ForecastInterval
                      available={forecast?.interval_available}
                      lower={forecast?.interval_lower_sec}
                      upper={forecast?.interval_upper_sec}
                    />
                  </Td>
                  <Td>
                    <IntervalOutcome forecast={forecast} settlement={row} />
                  </Td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="grid gap-3 p-3 md:hidden">
        {rows.map((row) => {
          const forecast = forecastByDriver.get(row.driver_code ?? row.driver ?? "");
          return (
            <article
              key={`${row.predicted_position}-${row.driver_code ?? row.driver}-mobile`}
              className="rounded-lg border border-apex-border bg-apex-bg/45 p-4"
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.12em] text-apex-muted">
                    Driver
                  </p>
                  <h3 className="mt-1 text-xl font-semibold text-apex-text">
                    {formatText(row.driver_code ?? row.driver)}
                  </h3>
                  <div className="mt-1 text-sm text-slate-300">
                    <TeamLabel forecast={forecast} />
                  </div>
                </div>
                <div className="text-right text-sm">
                  <p className="text-apex-muted">Position</p>
                  <p className="font-mono text-apex-text">
                    P{formatInteger(row.predicted_position)} to P
                    {formatInteger(row.actual_position)}
                  </p>
                </div>
              </div>
              <dl className="mt-4 grid gap-3 text-sm">
                <Detail label="Position delta">
                  <PositionDelta predicted={row.predicted_position} actual={row.actual_position} />
                </Detail>
                <Detail label="Predicted gap">{formatSignedGap(row.predicted_gap_to_pole_sec)}</Detail>
                <Detail label="Actual gap">{formatSignedGap(row.actual_gap_to_pole_sec)}</Detail>
                <Detail label="Absolute error">{formatSignedGap(row.absolute_gap_error_sec)}</Detail>
                <Detail label="Interval">
                  <ForecastInterval
                    available={forecast?.interval_available}
                    lower={forecast?.interval_lower_sec}
                    upper={forecast?.interval_upper_sec}
                  />
                </Detail>
                <Detail label="Interval outcome">
                  <IntervalOutcome forecast={forecast} settlement={row} />
                </Detail>
              </dl>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function TeamLabel({ forecast }: { forecast?: ForecastLeaderboardRow | null }) {
  return (
    <span className="inline-flex items-center gap-2">
      <span
        className="h-2.5 w-2.5 rounded-full"
        style={{ backgroundColor: teamAccent(forecast?.team_key) }}
        aria-hidden="true"
      />
      <span>{formatText(forecast?.team)}</span>
    </span>
  );
}

function Th({ children, align = "left" }: { children: React.ReactNode; align?: "left" | "right" }) {
  return (
    <th className={`px-4 py-3 font-semibold ${align === "right" ? "text-right" : "text-left"}`}>
      {children}
    </th>
  );
}

function Td({ children, numeric = false }: { children: React.ReactNode; numeric?: boolean }) {
  return (
    <td
      className={`px-4 py-3 align-top ${
        numeric ? "text-right font-mono tabular-nums text-slate-100" : "text-slate-200"
      }`}
    >
      {children}
    </td>
  );
}

function Detail({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4">
      <dt className="text-apex-muted">{label}</dt>
      <dd className="max-w-[60%] text-right text-slate-100">{children}</dd>
    </div>
  );
}
