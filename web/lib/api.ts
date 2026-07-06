import { getApiBaseUrl } from "@/lib/config";
import type {
  CurrentEventEnvelope,
  CurrentEventPageData,
  DashboardApiErrorPayload,
  ForecastEnvelope,
  ForecastPageData,
  HealthResponse,
  ManifestEnvelope,
  PracticePageData,
  PracticeStatusEnvelope,
  SafeDashboardError
} from "@/lib/dashboard-types";

const REQUEST_TIMEOUT_MS = 5000;

export class DashboardClientError extends Error {
  readonly code: string;
  readonly status?: number;
  readonly artifactType?: string;

  constructor(error: SafeDashboardError) {
    super(error.message);
    this.name = "DashboardClientError";
    this.code = error.code;
    this.status = error.status;
    this.artifactType = error.artifactType;
  }
}

export async function loadCurrentEventPageData(): Promise<CurrentEventPageData> {
  try {
    const health = await fetchDashboard<HealthResponse>("/api/v1/health");
    const [manifest, currentEvent, practiceStatus] = await Promise.all([
      fetchDashboard<ManifestEnvelope>("/api/v1/dashboard/manifest"),
      fetchDashboard<CurrentEventEnvelope>("/api/v1/dashboard/current-event"),
      fetchDashboard<PracticeStatusEnvelope>("/api/v1/dashboard/current-event/practice-status")
    ]);
    const forecast = await fetchDashboard<ForecastEnvelope>(
      "/api/v1/dashboard/current-event/forecast"
    ).catch(() => null);
    return {
      health,
      manifest,
      currentEvent,
      practiceStatus,
      forecast,
      error: null
    };
  } catch (error) {
    return {
      health: null,
      manifest: null,
      currentEvent: null,
      practiceStatus: null,
      forecast: null,
      error: normalizeClientError(error)
    };
  }
}

export async function loadForecastPageData(): Promise<ForecastPageData> {
  try {
    const health = await fetchDashboard<HealthResponse>("/api/v1/health");
    const [currentEvent, forecast] = await Promise.all([
      fetchDashboard<CurrentEventEnvelope>("/api/v1/dashboard/current-event"),
      fetchDashboard<ForecastEnvelope>("/api/v1/dashboard/current-event/forecast")
    ]);
    return {
      health,
      currentEvent,
      forecast,
      error: null
    };
  } catch (error) {
    return {
      health: null,
      currentEvent: null,
      forecast: null,
      error: normalizeClientError(error)
    };
  }
}

export async function loadPracticePageData(): Promise<PracticePageData> {
  try {
    const health = await fetchDashboard<HealthResponse>("/api/v1/health");
    const [currentEvent, practiceStatus] = await Promise.all([
      fetchDashboard<CurrentEventEnvelope>("/api/v1/dashboard/current-event"),
      fetchDashboard<PracticeStatusEnvelope>("/api/v1/dashboard/current-event/practice-status")
    ]);
    return {
      health,
      currentEvent,
      practiceStatus,
      error: null
    };
  } catch (error) {
    return {
      health: null,
      currentEvent: null,
      practiceStatus: null,
      error: normalizeClientError(error)
    };
  }
}

export async function fetchDashboard<TPayload>(endpoint: string): Promise<TPayload> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(`${getApiBaseUrl()}${endpoint}`, {
      cache: "no-store",
      signal: controller.signal,
      headers: {
        Accept: "application/json"
      }
    });
    const payload = await readJson(response);
    if (!response.ok) {
      throw errorFromResponse(response.status, payload);
    }
    return payload as TPayload;
  } catch (error) {
    if (error instanceof DashboardClientError) {
      throw error;
    }
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new DashboardClientError({
        code: "dashboard_api_timeout",
        message: "The dashboard API did not respond before the request timed out."
      });
    }
    throw new DashboardClientError({
      code: "dashboard_api_unavailable",
      message: "The dashboard API is unavailable. Start the read-only API server first."
    });
  } finally {
    clearTimeout(timeout);
  }
}

export function normalizeClientError(error: unknown): SafeDashboardError {
  if (error instanceof DashboardClientError) {
    return {
      code: error.code,
      message: error.message,
      status: error.status,
      artifactType: error.artifactType
    };
  }
  return {
    code: "dashboard_frontend_error",
    message: "The dashboard could not render the current artifact state safely."
  };
}

async function readJson(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

function errorFromResponse(status: number, payload: unknown): DashboardClientError {
  const apiError = parseApiError(payload);
  return new DashboardClientError({
    code: apiError?.detail.code ?? "dashboard_api_error",
    message:
      apiError?.detail.message ??
      "The dashboard API returned an error while serving artifact data.",
    status,
    artifactType: apiError?.detail.artifact_type
  });
}

function parseApiError(payload: unknown): DashboardApiErrorPayload | null {
  if (!payload || typeof payload !== "object" || !("detail" in payload)) {
    return null;
  }
  const detail = (payload as { detail?: unknown }).detail;
  if (!detail || typeof detail !== "object") {
    return null;
  }
  const candidate = detail as {
    code?: unknown;
    message?: unknown;
    artifact_type?: unknown;
  };
  if (typeof candidate.code !== "string" || typeof candidate.message !== "string") {
    return null;
  }
  return {
    detail: {
      code: candidate.code,
      message: candidate.message,
      artifact_type:
        typeof candidate.artifact_type === "string" ? candidate.artifact_type : undefined
    }
  };
}
