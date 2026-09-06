import type {
  AutopilotStatusData,
  AutopilotStatusEnvelope,
  EventSchedule,
  OperationalEvent
} from "@/lib/dashboard-types";

export interface OperationalForecastNotice {
  label: string;
  detail: string;
  tone: "neutral" | "warning" | "success";
}

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

export function operationalForecastNotice(
  event: OperationalEvent,
  status: AutopilotStatusData | null | undefined
): OperationalForecastNotice {
  if (!event.supported) {
    return {
      label: "Forecast not supported",
      detail:
        "Apex Pulse does not create predictions for Sprint weekends yet. This weekend remains visible and monitored, but no qualifying forecast will be generated.",
      tone: "warning"
    };
  }
  if (status?.forecast_exists === true) {
    return {
      label: "Forecast generated",
      detail: "The immutable pre-qualifying forecast is available for this monitored weekend.",
      tone: "success"
    };
  }
  if (status?.action_result === "forecast_window_missed") {
    return {
      label: "Forecast not generated",
      detail:
        "The pre-qualifying window closed before a safe forecast could be created. Apex Pulse will not add a retrospective prediction.",
      tone: "warning"
    };
  }
  if (status?.orchestrator_state_after === "TRANSIENT_ERROR" && status.retryable) {
    return {
      label: "Forecast pending",
      detail:
        "Public data is not yet sufficient to create a safe forecast. Monitoring will retry while the pre-qualifying window remains open.",
      tone: "warning"
    };
  }
  if (status?.orchestrator_state_after === "BLOCKED") {
    return {
      label: "Forecast not generated",
      detail:
        "A safe pre-qualifying forecast is not available for this weekend. No retrospective prediction will be created.",
      tone: "warning"
    };
  }
  return {
    label: "Forecast monitoring active",
    detail:
      "Apex Pulse is monitoring the conventional practice-to-qualifying schedule for the next guarded prediction window.",
    tone: "neutral"
  };
}
