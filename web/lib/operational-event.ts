import type {
  AutopilotStatusEnvelope,
  EventSchedule,
  OperationalEvent
} from "@/lib/dashboard-types";

export function availableOperationalEvent(
  status: AutopilotStatusEnvelope | null | undefined
): OperationalEvent | null {
  if (status?.status !== "available") {
    return null;
  }
  const event = status.data.operational_event;
  return event?.schedule_available && event.sessions.length > 0 ? event : null;
}

export function operationalEventSchedule(event: OperationalEvent): EventSchedule {
  return {
    available: event.schedule_available,
    source: event.calendar_source,
    timezone: event.timezone,
    sessions: event.sessions.map((session) => ({
      session: session.session,
      display_name: session.display_name,
      scheduled_start_utc: session.scheduled_start_utc,
      scheduled_end_utc: session.scheduled_end_utc
    }))
  };
}

export function operationalWeekendLabel(event: OperationalEvent, now: Date): string {
  const starts = event.sessions.map((session) => Date.parse(session.scheduled_start_utc));
  const ends = event.sessions.map((session) => Date.parse(session.scheduled_end_utc));
  const firstStart = Math.min(...starts.filter(Number.isFinite));
  const lastEnd = Math.max(...ends.filter(Number.isFinite));
  if (!Number.isFinite(firstStart) || !Number.isFinite(lastEnd)) {
    return "Monitored weekend";
  }
  if (now.getTime() < firstStart) {
    return "Next weekend";
  }
  if (now.getTime() <= lastEnd) {
    return "Current weekend";
  }
  return "Recently monitored weekend";
}

export function operationalFormatLabel(eventFormat: string): string {
  const normalized = eventFormat.toLowerCase();
  if (normalized.includes("sprint")) {
    return "Sprint weekend";
  }
  if (normalized === "conventional") {
    return "Conventional weekend";
  }
  return "Non-standard weekend";
}
