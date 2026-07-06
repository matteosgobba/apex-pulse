import type { SessionStatus } from "@/lib/dashboard-types";
import { formatDateTime, humanizeToken } from "@/lib/formatters";

const ORDER = ["FP1", "FP2", "FP3", "Q"];

export function SessionStatusRow({ sessions = [] }: { sessions?: SessionStatus[] }) {
  const bySession = new Map(sessions.map((session) => [session.session, session]));
  const normalized = ORDER.map(
    (session) =>
      bySession.get(session) ?? {
        session,
        available: false,
        status: "unavailable",
        artifact_available: false,
        last_known_timestamp: null,
        reason: "session_status_not_exported"
      }
  );

  return (
    <section className="rounded-lg border border-apex-border bg-apex-panel/85 p-4">
      <div className="flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 className="text-base font-semibold text-apex-text">Practice Session Availability</h2>
          <p className="mt-1 text-sm text-apex-muted">
            Artifact readiness only. This view does not represent lap-by-lap live timing.
          </p>
        </div>
      </div>
      <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {normalized.map((session) => (
          <article
            key={session.session}
            className="rounded-md border border-apex-border bg-apex-bg/45 p-3"
          >
            <div className="flex items-center justify-between gap-3">
              <h3 className="text-sm font-semibold text-apex-text">{session.session}</h3>
              <span
                className={`rounded-full px-2 py-1 text-xs font-semibold ${
                  session.available
                    ? "bg-emerald-300/10 text-emerald-100"
                    : "bg-slate-400/10 text-slate-300"
                }`}
              >
                {session.available ? "Available" : "Unavailable"}
              </span>
            </div>
            <dl className="mt-3 space-y-2 text-sm">
              <div className="flex justify-between gap-3">
                <dt className="text-apex-muted">Status</dt>
                <dd className="text-right text-slate-100">{humanizeToken(session.status)}</dd>
              </div>
              <div className="flex justify-between gap-3">
                <dt className="text-apex-muted">Timestamp</dt>
                <dd className="text-right font-mono text-slate-100">
                  {formatDateTime(session.last_known_timestamp)}
                </dd>
              </div>
              <div className="flex justify-between gap-3">
                <dt className="text-apex-muted">Reason</dt>
                <dd className="max-w-[58%] break-words text-right text-slate-100">
                  {humanizeToken(session.reason)}
                </dd>
              </div>
            </dl>
          </article>
        ))}
      </div>
    </section>
  );
}
