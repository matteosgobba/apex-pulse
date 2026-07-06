export const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

export function getApiBaseUrl(): string {
  const configured = process.env.NEXT_PUBLIC_APEX_PULSE_API_BASE_URL?.trim();
  return configured && configured.length > 0
    ? configured.replace(/\/+$/, "")
    : DEFAULT_API_BASE_URL;
}
