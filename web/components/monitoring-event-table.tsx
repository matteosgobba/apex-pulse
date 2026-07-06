import { LifecycleBadge } from "@/components/lifecycle-badge";
import { TableEmptyState } from "@/components/table-empty-state";
import type { MonitoringHistoryEvent } from "@/lib/dashboard-types";
import { formatInteger, formatPercent, formatSeconds, formatText } from "@/lib/formatters";

export function MonitoringEventTable({ events }: { events: MonitoringHistoryEvent[] }) {
  if (events.length === 0) {
    return (
      <TableEmptyState
        title="No valid prospective events yet."
        message="Valid prospective monitoring evidence will appear here only after eligible monitored events are forecasted and settled under canonical lineage."
      />
    );
  }
  return (
    <section className="rounded-lg border border-apex-border bg-apex-panel/85 shadow-panel">
      <div className="border-b border-apex-border px-4 py-3">
        <h2 className="text-base font-semibold text-apex-text">Eligible Event History</h2>
      </div>
      <div className="hidden overflow-x-auto md:block">
        <table className="min-w-full border-collapse text-sm">
          <thead className="bg-apex-panelSoft text-xs uppercase tracking-[0.12em] text-apex-muted">
            <tr>
              <Th>Season</Th>
              <Th>Event</Th>
              <Th>Lifecycle</Th>
              <Th>Forecast</Th>
              <Th>Settlement</Th>
              <Th>Checkpoint</Th>
              <Th align="right">MAE gap</Th>
              <Th align="right">Intervals</Th>
              <Th>Evidence</Th>
            </tr>
          </thead>
          <tbody>
            {events.map((event) => (
              <tr
                key={`${event.event_identity?.season}-${event.event_identity?.event_slug}`}
                className="border-t border-apex-border/80"
              >
                <Td>{formatInteger(event.event_identity?.season)}</Td>
                <Td>{formatText(event.event_identity?.event)}</Td>
                <Td>
                  <LifecycleBadge state={event.lifecycle_state} />
                </Td>
                <Td>{event.forecast_available ? "Available" : "Not available"}</Td>
                <Td>{event.settlement_available ? "Available" : "Not available"}</Td>
                <Td>{formatText(event.forecast_checkpoint)}</Td>
                <Td numeric>{formatSeconds(event.mae_gap_sec)}</Td>
                <Td numeric>{formatPercent(event.interval_availability_rate)}</Td>
                <Td>
                  {event.eligible_for_valid_prospective_evidence ? "Valid prospective" : "Excluded"}
                </Td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="grid gap-3 p-3 md:hidden">
        {events.map((event) => (
          <article
            key={`${event.event_identity?.season}-${event.event_identity?.event_slug}-mobile`}
            className="rounded-lg border border-apex-border bg-apex-bg/45 p-4"
          >
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.12em] text-apex-muted">
                  {formatInteger(event.event_identity?.season)}
                </p>
                <h3 className="mt-1 text-xl font-semibold text-apex-text">
                  {formatText(event.event_identity?.event)}
                </h3>
              </div>
              <LifecycleBadge state={event.lifecycle_state} />
            </div>
            <dl className="mt-4 grid gap-3 text-sm">
              <Detail label="Forecast">
                {event.forecast_available ? "Available" : "Not available"}
              </Detail>
              <Detail label="Settlement">
                {event.settlement_available ? "Available" : "Not available"}
              </Detail>
              <Detail label="Checkpoint">{formatText(event.forecast_checkpoint)}</Detail>
              <Detail label="MAE gap">{formatSeconds(event.mae_gap_sec)}</Detail>
            </dl>
          </article>
        ))}
      </div>
    </section>
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
