import type {
  EventSchedule,
  LifecycleState,
  SessionStatus
} from "@/lib/dashboard-types";
import { selectNextSession, sessionDisplayName } from "@/lib/session-schedule";

const SESSION_CODES = ["FP1", "FP2", "FP3", "Q"];

export function WeekendTimeline({
  schedule,
  sessions,
  lifecycle,
  now = new Date()
}: {
  schedule: EventSchedule | null;
  sessions: SessionStatus[];
  lifecycle: LifecycleState;
  now?: Date;
}) {
  const next = selectNextSession(schedule, lifecycle, now);
  return (
    <section
      id="weekend"
      aria-labelledby="weekend-title"
      className="rounded-3xl border border-apex-border bg-white p-6 shadow-card"
    >
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
          Weekend progression
        </p>
        <h2 id="weekend-title" className="mt-2 text-2xl font-semibold text-apex-text">
          Practice to qualifying
        </h2>
      </div>
      <ol className="mt-7 grid gap-3 sm:grid-cols-4">
        {SESSION_CODES.map((code, index) => {
          const practice = sessions.find((session) => session.session === code);
          const scheduled = schedule?.sessions?.find((session) => session.session === code);
          const isNext = next.state === "upcoming" && next.session?.session === code;
          const completed =
            lifecycle === "settled" ||
            lifecycle === "settled_partial_coverage" ||
            Boolean(practice?.available && !isNext);
          const status = completed ? "Completed" : isNext ? "Next" : "Unavailable";
          return (
            <li key={code} className="relative">
              <div
                className={`h-full rounded-2xl border p-4 ${
                  isNext
                    ? "border-apex-accent bg-red-50"
                    : completed
                      ? "border-emerald-200 bg-emerald-50/60"
                      : "border-apex-border bg-slate-50"
                }`}
              >
                <div className="flex items-center gap-2">
                  <span
                    aria-hidden="true"
                    className={`flex h-6 w-6 items-center justify-center rounded-full text-xs font-bold ${
                      completed
                        ? "bg-emerald-600 text-white"
                        : isNext
                          ? "bg-apex-accent text-white"
                          : "bg-slate-200 text-slate-500"
                    }`}
                  >
                    {completed ? "✓" : index + 1}
                  </span>
                  <span className="font-semibold text-apex-text">{sessionDisplayName(code)}</span>
                </div>
                <p className="mt-3 text-xs font-medium text-slate-500">{status}</p>
                <p className="mt-1 text-xs text-slate-500">
                  {formatScheduleTime(scheduled?.scheduled_start_utc)}
                </p>
              </div>
            </li>
          );
        })}
      </ol>
      <p className="mt-4 text-xs leading-5 text-slate-500">
        Status reflects exported artifact availability, not lap-by-lap live coverage.
      </p>
    </section>
  );
}

function formatScheduleTime(value: string | null | undefined): string {
  if (!value) {
    return "Time unavailable";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return "Time unavailable";
  }
  return new Intl.DateTimeFormat("en", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  }).format(parsed);
}
