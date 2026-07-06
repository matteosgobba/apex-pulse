export const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";
export const DEFAULT_STALE_AFTER_MINUTES = 180;

export function getApiBaseUrl(): string {
  const configured = process.env.NEXT_PUBLIC_APEX_PULSE_API_BASE_URL?.trim();
  return configured && configured.length > 0
    ? configured.replace(/\/+$/, "")
    : DEFAULT_API_BASE_URL;
}

export function getDashboardStaleAfterMinutes(): number {
  const configured = process.env.NEXT_PUBLIC_APEX_PULSE_DASHBOARD_STALE_AFTER_MINUTES?.trim();
  if (!configured) {
    return DEFAULT_STALE_AFTER_MINUTES;
  }
  const parsed = Number.parseInt(configured, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_STALE_AFTER_MINUTES;
}
