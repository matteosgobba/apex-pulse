import { PracticeSessionCard } from "@/components/practice-session-card";
import type { SessionStatus } from "@/lib/dashboard-types";

const SESSION_ORDER = ["FP1", "FP2", "FP3", "Q"];

export function orderedSessions(sessions: SessionStatus[] = []): SessionStatus[] {
  const bySession = new Map(sessions.map((session) => [session.session, session]));
  return SESSION_ORDER.map(
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
}

export function PracticeTimeline({ sessions }: { sessions: SessionStatus[] }) {
  const normalized = orderedSessions(sessions);
  return (
    <section className="rounded-lg border border-apex-border bg-apex-panel/85 p-5 shadow-panel">
      <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
        <div>
          <h2 className="text-lg font-semibold text-apex-text">FP1 to Qualifying Progression</h2>
          <p className="mt-1 text-sm leading-6 text-apex-muted">
            Session availability is artifact-based readiness, not lap-by-lap live timing.
          </p>
        </div>
      </div>
      <div className="mt-5 grid gap-4 lg:grid-cols-4">
        {normalized.map((session, index) => (
          <div key={session.session} className="relative">
            {index < normalized.length - 1 ? (
              <div
                className="absolute left-[calc(100%+0.25rem)] top-1/2 hidden h-px w-3 bg-apex-border lg:block"
                aria-hidden="true"
              />
            ) : null}
            <PracticeSessionCard session={session} index={index} />
          </div>
        ))}
      </div>
    </section>
  );
}
