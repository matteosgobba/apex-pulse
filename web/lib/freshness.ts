import { getDashboardStaleAfterMinutes } from "@/lib/config";
import { formatDateTime } from "@/lib/formatters";

export type FreshnessState = "fresh" | "aging" | "stale" | "unknown";

export interface FreshnessStatus {
  state: FreshnessState;
  relativeLabel: string;
  exactUtcLabel: string;
  exactLocalLabel: string;
  ageMinutes: number | null;
}

const FRESH_UNDER_MINUTES = 60;

export function evaluateFreshness(
  generatedAt: string | null | undefined,
  options: {
    now?: Date;
    staleAfterMinutes?: number;
  } = {}
): FreshnessStatus {
  if (!generatedAt) {
    return unknownFreshness();
  }
  const parsed = new Date(generatedAt);
  if (Number.isNaN(parsed.getTime())) {
    return unknownFreshness();
  }
  const now = options.now ?? new Date();
  const staleAfterMinutes = options.staleAfterMinutes ?? getDashboardStaleAfterMinutes();
  const rawAgeMinutes = Math.floor((now.getTime() - parsed.getTime()) / 60_000);
  const ageMinutes = Math.max(0, rawAgeMinutes);
  const state =
    ageMinutes < FRESH_UNDER_MINUTES
      ? "fresh"
      : ageMinutes > staleAfterMinutes
        ? "stale"
        : "aging";

  return {
    state,
    relativeLabel: `Updated ${formatRelativeAge(ageMinutes)} ago`,
    exactUtcLabel: `UTC: ${formatUtc(generatedAt)}`,
    exactLocalLabel: `Local: ${formatDateTime(generatedAt)}`,
    ageMinutes
  };
}

function unknownFreshness(): FreshnessStatus {
  return {
    state: "unknown",
    relativeLabel: "Update time unavailable",
    exactUtcLabel: "UTC: Not available",
    exactLocalLabel: "Local: Not available",
    ageMinutes: null
  };
}

function formatRelativeAge(ageMinutes: number): string {
  if (ageMinutes < 1) {
    return "less than a minute";
  }
  if (ageMinutes < 60) {
    return `${ageMinutes} ${ageMinutes === 1 ? "minute" : "minutes"}`;
  }
  const hours = Math.floor(ageMinutes / 60);
  if (hours < 24) {
    return `${hours} ${hours === 1 ? "hour" : "hours"}`;
  }
  const days = Math.floor(hours / 24);
  return `${days} ${days === 1 ? "day" : "days"}`;
}

function formatUtc(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return "Not available";
  }
  return new Intl.DateTimeFormat("en", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    timeZone: "UTC",
    timeZoneName: "short"
  }).format(parsed);
}
