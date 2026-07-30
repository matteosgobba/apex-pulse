import type {
  EventSchedule,
  EventScheduleSession,
  LifecycleState
} from "@/lib/dashboard-types";

export type CountdownState = "upcoming" | "complete" | "unavailable";

export interface SessionCountdownSelection {
  state: CountdownState;
  session: EventScheduleSession | null;
  targetTimeMs: number | null;
  reason: string;
}

export interface CountdownParts {
  days: number;
  hours: number;
  minutes: number;
  seconds: number;
  complete: boolean;
}

const SESSION_ORDER = ["FP1", "FP2", "FP3", "Q"];

export function selectNextSession(
  schedule: EventSchedule | null | undefined,
  lifecycle: LifecycleState | null | undefined,
  now: Date
): SessionCountdownSelection {
  if (lifecycle === "settled" || lifecycle === "settled_partial_coverage") {
    return {
      state: "complete",
      session: schedule?.sessions?.find((session) => session.session === "Q") ?? null,
      targetTimeMs: null,
      reason: "Qualifying has completed."
    };
  }
  if (!schedule?.available || !Array.isArray(schedule.sessions)) {
    return unavailableSelection(schedule?.reason);
  }

  const sessions = schedule.sessions
    .map((session) => ({ session, time: parseTime(session.scheduled_start_utc) }))
    .filter(
      (entry): entry is { session: EventScheduleSession; time: number } => entry.time !== null
    )
    .sort(
      (left, right) =>
        left.time - right.time ||
        SESSION_ORDER.indexOf(left.session.session) - SESSION_ORDER.indexOf(right.session.session)
    );
  const next = sessions.find((entry) => entry.time > now.getTime());
  if (next) {
    return {
      state: "upcoming",
      session: next.session,
      targetTimeMs: next.time,
      reason: `Next session: ${next.session.display_name ?? sessionDisplayName(next.session.session)}.`
    };
  }

  const qualifying = sessions.find((entry) => entry.session.session === "Q");
  if (qualifying) {
    return {
      state: "complete",
      session: qualifying.session,
      targetTimeMs: null,
      reason: "Qualifying has completed."
    };
  }
  return unavailableSelection("No future session timestamp is available.");
}

export function countdownParts(targetTimeMs: number, nowTimeMs: number): CountdownParts {
  const remainingSeconds = Math.max(0, Math.floor((targetTimeMs - nowTimeMs) / 1000));
  return {
    days: Math.floor(remainingSeconds / 86_400),
    hours: Math.floor((remainingSeconds % 86_400) / 3_600),
    minutes: Math.floor((remainingSeconds % 3_600) / 60),
    seconds: remainingSeconds % 60,
    complete: remainingSeconds === 0
  };
}

export function sessionDisplayName(session: string): string {
  return session === "Q" ? "Qualifying" : session;
}

function parseTime(value: string | null | undefined): number | null {
  if (!value) {
    return null;
  }
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? null : parsed;
}

function unavailableSelection(reason?: string | null): SessionCountdownSelection {
  return {
    state: "unavailable",
    session: null,
    targetTimeMs: null,
    reason: reason || "Session schedule is not available in this export."
  };
}
