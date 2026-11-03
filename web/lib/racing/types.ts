export interface RacingSession {
  code: string;
  name: string;
  startUtc: string;
}
export interface RacingEvent {
  id: string;
  season: number;
  round: number;
  name: string;
  location: string;
  country: string;
  countryCode: string | null;
  circuitId: string | null;
  format: string;
  status?: string;
  date: string | null;
  sessions: RacingSession[];
}
export interface Circuit {
  name: string;
  location?: string;
  countryCode?: string;
  points: number[][];
}
export interface CalendarBundle {
  schemaVersion: 1;
  retrievedAtUtc: string;
  geometryRetrievedAtUtc?: string;
  geometryWarning?: boolean;
  nextSeasonWarning?: boolean;
  nextSeasonAvailable?: boolean;
  events: RacingEvent[];
  circuits: Record<string, Circuit>;
}
export interface WeekendSelection {
  status:
    | "unavailable"
    | "complete"
    | "awaiting_schedule"
    | "scheduled_start"
    | "upcoming";
  event?: RacingEvent;
  session?: RacingSession & { startMs: number };
  sessions?: (RacingSession & { startMs: number })[];
  remainingMs?: number;
  lights?: number;
}
export interface CalendarResponse {
  bundle: CalendarBundle;
  status: "fresh" | "cached";
  checkedAtUtc: string;
  warning?: string;
}
