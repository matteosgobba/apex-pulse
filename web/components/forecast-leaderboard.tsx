import { ForecastInterval } from "@/components/forecast-interval";
import { TableEmptyState } from "@/components/table-empty-state";
import { TeamMark } from "@/components/team-mark";
import type { ForecastLeaderboardRow } from "@/lib/dashboard-types";
import { formatInteger, formatSignedGap, formatText, humanizeToken } from "@/lib/formatters";
import { getTeamIdentity } from "@/lib/team-identity";
import type { ReactNode } from "react";

export function ForecastLeaderboard({
  rows,
  compact = false
}: {
  rows: ForecastLeaderboardRow[];
  compact?: boolean;
}) {
  if (rows.length === 0) {
    return (
      <TableEmptyState
        title="No forecast rows are available."
        message="The forecast artifact exists, but it does not include ranked driver rows for the current event."
      />
    );
  }

  const displayRows = compact ? rows.slice(0, 5) : rows;
  const hasSettlement = displayRows.some(
    (row) =>
      row.actual_position !== null ||
      row.actual_gap_to_pole_sec !== null ||
      row.absolute_gap_error_sec !== null
  );

  return (
    <section className="rounded-lg border border-apex-border bg-apex-panel/85 shadow-panel">
      <div className="border-b border-apex-border px-4 py-3">
        <h2 className="text-base font-semibold text-apex-text">
          {compact ? "Forecast Preview" : "Qualifying Forecast Leaderboard"}
        </h2>
        <p className="mt-1 text-sm text-apex-muted">
          Exported leaderboard order is preserved. Values are pre-qualifying estimates.
        </p>
      </div>
      <div className="hidden overflow-x-auto md:block">
        <table className="min-w-full border-collapse text-sm">
          <thead className="bg-apex-panelSoft text-xs uppercase tracking-[0.12em] text-apex-muted">
            <tr>
              <Th>Predicted position</Th>
              <Th>Driver</Th>
              <Th>Team</Th>
              <Th align="right">Predicted gap to pole</Th>
              <Th>Prediction interval</Th>
              <Th>Method / provenance</Th>
              {hasSettlement ? (
                <>
                  <Th align="right">Actual position</Th>
                  <Th align="right">Actual gap</Th>
                  <Th align="right">Absolute gap error</Th>
                </>
              ) : null}
            </tr>
          </thead>
          <tbody>
            {displayRows.map((row) => (
              <tr
                key={`${row.predicted_position}-${row.driver_code ?? row.driver}`}
                className="border-t border-apex-border/80"
              >
                <Td numeric>{formatInteger(row.predicted_position)}</Td>
                <Td>
                  <div className="font-semibold text-apex-text">
                    {formatText(row.driver_code ?? row.driver)}
                  </div>
                  <div className="text-xs text-apex-muted">{formatText(row.driver)}</div>
                </Td>
                <Td>
                  <span className="inline-flex items-center gap-2">
                    <TeamMark team={getTeamIdentity(row.team_key, row.team)} size="sm" />
                    <span>{formatText(row.team)}</span>
                  </span>
                </Td>
                <Td numeric>{formatSignedGap(row.predicted_gap_to_pole_sec)}</Td>
                <Td>
                  <ForecastInterval
                    available={row.interval_available}
                    lower={row.interval_lower_sec}
                    upper={row.interval_upper_sec}
                  />
                </Td>
                <Td>
                  <MethodProvenance row={row} />
                </Td>
                {hasSettlement ? (
                  <>
                    <Td numeric>{formatInteger(row.actual_position)}</Td>
                    <Td numeric>{formatSignedGap(row.actual_gap_to_pole_sec)}</Td>
                    <Td numeric>{formatSignedGap(row.absolute_gap_error_sec)}</Td>
                  </>
                ) : null}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="grid gap-3 p-3 md:hidden">
        {displayRows.map((row) => (
          <article
            key={`${row.predicted_position}-${row.driver_code ?? row.driver}-mobile`}
            className="rounded-lg border border-apex-border bg-apex-bg/45 p-4"
          >
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="text-xs font-semibold uppercase tracking-[0.12em] text-apex-muted">
                  Predicted P{formatInteger(row.predicted_position)}
                </p>
                <h3 className="mt-1 text-xl font-semibold text-apex-text">
                  {formatText(row.driver_code ?? row.driver)}
                </h3>
                <p className="mt-1 flex items-center gap-2 text-sm text-slate-300">
                  <TeamMark team={getTeamIdentity(row.team_key, row.team)} size="sm" />
                  {formatText(row.team)}
                </p>
              </div>
              <div className="text-right">
                <p className="text-xs text-apex-muted">Gap</p>
                <p className="font-mono text-lg text-apex-text">
                  {formatSignedGap(row.predicted_gap_to_pole_sec)}
                </p>
              </div>
            </div>
            <dl className="mt-4 grid gap-3 text-sm">
              <Detail label="Prediction interval">
                <ForecastInterval
                  available={row.interval_available}
                  lower={row.interval_lower_sec}
                  upper={row.interval_upper_sec}
                />
              </Detail>
              <Detail label="Method">
                <MethodProvenance row={row} />
              </Detail>
              {hasSettlement ? (
                <>
                  <Detail label="Actual position">{formatInteger(row.actual_position)}</Detail>
                  <Detail label="Actual gap">{formatSignedGap(row.actual_gap_to_pole_sec)}</Detail>
                  <Detail label="Absolute gap error">
                    {formatSignedGap(row.absolute_gap_error_sec)}
                  </Detail>
                </>
              ) : null}
            </dl>
          </article>
        ))}
      </div>
    </section>
  );
}

function MethodProvenance({ row }: { row: ForecastLeaderboardRow }) {
  const method = row.selected_method;
  const provenance = row.provenance;
  const methodText = [method?.model_name, method?.feature_group]
    .filter(Boolean)
    .map((value) => humanizeToken(value))
    .join(" / ");
  const integrity = provenance?.forecast_integrity_status
    ? humanizeToken(provenance.forecast_integrity_status)
    : null;
  return (
    <div className="space-y-1">
      <p className="text-slate-100">{methodText || "Not available"}</p>
      <p className="text-xs text-apex-muted">
        {integrity ? `Integrity: ${integrity}` : "Provenance not available"}
      </p>
    </div>
  );
}

function Th({
  children,
  align = "left"
}: {
  children: ReactNode;
  align?: "left" | "right";
}) {
  return (
    <th className={`px-4 py-3 font-semibold ${align === "right" ? "text-right" : "text-left"}`}>
      {children}
    </th>
  );
}

function Td({
  children,
  numeric = false
}: {
  children: ReactNode;
  numeric?: boolean;
}) {
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

function Detail({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4">
      <dt className="text-apex-muted">{label}</dt>
      <dd className="max-w-[60%] text-right text-slate-100">{children}</dd>
    </div>
  );
}
