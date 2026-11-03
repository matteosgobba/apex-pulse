import type { RacingEvent, WeekendSelection } from "./types";
export const START_GRACE_MS: number;
export const LIGHT_THRESHOLDS_MS: number[];
export function lightCount(remainingMs: number): number;
export function countdownParts(remainingMs: number): {
  days: number;
  hours: number;
  minutes: number;
  seconds: number;
};
export function validSessions(
  event: RacingEvent,
): NonNullable<WeekendSelection["sessions"]>;
export function selectWeekend(
  events: RacingEvent[],
  nowMs: number,
): WeekendSelection;
export function projectCircuit(
  points: number[][],
  bounds?: { x: number; y: number; width: number; height: number },
): { path: string; coordinates: number[][] } | null;
export function countryFlag(code: string | null): string;
