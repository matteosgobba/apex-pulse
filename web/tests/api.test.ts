import { afterEach, describe, expect, test, vi } from "vitest";

import {
  fetchDashboard,
  loadCurrentEventPageData,
  loadForecastPageData,
  loadMonitoringHistoryPageData,
  loadPracticePageData,
  loadSettlementPageData
} from "@/lib/api";
import {
  DEFAULT_API_BASE_URL,
  DEFAULT_STALE_AFTER_MINUTES,
  getApiBaseUrl,
  getDashboardStaleAfterMinutes
} from "@/lib/config";

const originalFetch = global.fetch;
const originalApiBaseUrl = process.env.NEXT_PUBLIC_APEX_PULSE_API_BASE_URL;
const originalStaleAfter = process.env.NEXT_PUBLIC_APEX_PULSE_DASHBOARD_STALE_AFTER_MINUTES;

afterEach(() => {
  global.fetch = originalFetch;
  if (originalApiBaseUrl === undefined) {
    delete process.env.NEXT_PUBLIC_APEX_PULSE_API_BASE_URL;
  } else {
    process.env.NEXT_PUBLIC_APEX_PULSE_API_BASE_URL = originalApiBaseUrl;
  }
  if (originalStaleAfter === undefined) {
    delete process.env.NEXT_PUBLIC_APEX_PULSE_DASHBOARD_STALE_AFTER_MINUTES;
  } else {
    process.env.NEXT_PUBLIC_APEX_PULSE_DASHBOARD_STALE_AFTER_MINUTES = originalStaleAfter;
  }
  vi.restoreAllMocks();
});

describe("dashboard API client", () => {
  test("maps a stable API error response into a safe frontend error state", async () => {
    global.fetch = vi.fn(async () =>
      new Response(
        JSON.stringify({
          detail: {
            code: "dashboard_artifacts_unavailable",
            message: "Dashboard artifacts have not been exported yet. Run dashboard-export first."
          }
        }),
        { status: 503, headers: { "content-type": "application/json" } }
      )
    );

    await expect(fetchDashboard("/api/v1/dashboard/current-event")).rejects.toMatchObject({
      code: "dashboard_artifacts_unavailable",
      status: 503,
      message: "Dashboard artifacts have not been exported yet. Run dashboard-export first."
    });
  });

  test("API base URL is read from configuration", async () => {
    process.env.NEXT_PUBLIC_APEX_PULSE_API_BASE_URL = "http://localhost:9000/";
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify({ status: "ok" }), {
        status: 200,
        headers: { "content-type": "application/json" }
      })
    );
    global.fetch = fetchMock;

    expect(getApiBaseUrl()).toBe("http://localhost:9000");
    await fetchDashboard("/api/v1/health");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:9000/api/v1/health",
      expect.objectContaining({
        cache: "no-store"
      })
    );
  });

  test("default API base URL is local FastAPI", () => {
    delete process.env.NEXT_PUBLIC_APEX_PULSE_API_BASE_URL;

    expect(getApiBaseUrl()).toBe(DEFAULT_API_BASE_URL);
  });

  test("dashboard stale threshold is read from configuration", () => {
    process.env.NEXT_PUBLIC_APEX_PULSE_DASHBOARD_STALE_AFTER_MINUTES = "240";

    expect(getDashboardStaleAfterMinutes()).toBe(240);

    process.env.NEXT_PUBLIC_APEX_PULSE_DASHBOARD_STALE_AFTER_MINUTES = "invalid";

    expect(getDashboardStaleAfterMinutes()).toBe(DEFAULT_STALE_AFTER_MINUTES);
  });

  test("API-unavailable state uses safe frontend error text", async () => {
    global.fetch = vi.fn(async () => {
      throw new TypeError("fetch failed: /private/tmp/secret-dashboard-path");
    });

    await expect(fetchDashboard("/api/v1/dashboard/current-event")).rejects.toMatchObject({
      code: "dashboard_api_unavailable",
      message: "The dashboard API is unavailable. Start the read-only API server first."
    });
  });

  test("operational status failure is additive and does not hide latest dashboard state", async () => {
    global.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/v1/autopilot-status")) {
        return jsonResponse(
          { detail: { code: "not_initialized", message: "Not initialized" } },
          503
        );
      }
      if (url.endsWith("/api/v1/health")) {
        return jsonResponse({
          status: "ok",
          service: "apex-pulse-dashboard-api",
          api_version: "v1",
          dashboard_artifact_status: "complete"
        });
      }
      const artifactType = url.endsWith("/manifest")
        ? "dashboard_manifest"
        : url.endsWith("/practice-status")
          ? "event_practice_status"
          : url.endsWith("/forecast")
            ? "event_forecast"
            : url.endsWith("/settlement")
              ? "event_settlement"
              : url.endsWith("/historical-monitoring")
                ? "historical_monitoring_summary"
                : "current_event";
      return jsonResponse({
        schema_version: "1.0",
        artifact_type: artifactType,
        generated_at_utc: "2026-08-01T00:00:00Z",
        source_artifacts: [],
        source_fingerprints: {},
        status: "complete",
        data: {}
      });
    });

    const data = await loadCurrentEventPageData();

    expect(data.error).toBeNull();
    expect(data.currentEvent?.artifact_type).toBe("current_event");
    expect(data.operationalStatus).toBeNull();
  });

  test("forecast endpoint failure maps to a safe page error", async () => {
    global.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/v1/health")) {
        return jsonResponse({ status: "ok", service: "apex-pulse-dashboard-api", api_version: "v1" });
      }
      if (url.endsWith("/api/v1/dashboard/current-event")) {
        return jsonResponse({ schema_version: "1.0", artifact_type: "current_event", data: {} });
      }
      return jsonResponse(
        {
          detail: {
            code: "dashboard_artifact_not_found",
            message: "The requested dashboard artifact is not available.",
            artifact_type: "event_forecast"
          }
        },
        404
      );
    });

    const data = await loadForecastPageData();

    expect(data.error).toMatchObject({
      code: "dashboard_artifact_not_found",
      artifactType: "event_forecast"
    });
  });

  test("practice endpoint failure maps to a safe page error", async () => {
    global.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/v1/health")) {
        return jsonResponse({ status: "ok", service: "apex-pulse-dashboard-api", api_version: "v1" });
      }
      if (url.endsWith("/api/v1/dashboard/current-event")) {
        return jsonResponse({ schema_version: "1.0", artifact_type: "current_event", data: {} });
      }
      return jsonResponse(
        {
          detail: {
            code: "dashboard_artifact_invalid",
            message: "The requested dashboard artifact failed validation.",
            artifact_type: "event_practice_status"
          }
        },
        500
      );
    });

    const data = await loadPracticePageData();

    expect(data.error).toMatchObject({
      code: "dashboard_artifact_invalid",
      artifactType: "event_practice_status"
    });
  });

  test("settlement endpoint failure maps to a safe page error", async () => {
    global.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/v1/health")) {
        return jsonResponse({ status: "ok", service: "apex-pulse-dashboard-api", api_version: "v1" });
      }
      if (url.endsWith("/api/v1/dashboard/current-event")) {
        return jsonResponse({ schema_version: "1.0", artifact_type: "current_event", data: {} });
      }
      return jsonResponse(
        {
          detail: {
            code: "dashboard_artifact_not_found",
            message: "The requested dashboard artifact is not available.",
            artifact_type: "event_settlement"
          }
        },
        404
      );
    });

    const data = await loadSettlementPageData();

    expect(data.error).toMatchObject({
      code: "dashboard_artifact_not_found",
      artifactType: "event_settlement"
    });
  });

  test("history endpoint failure maps to a safe page error", async () => {
    global.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/v1/health")) {
        return jsonResponse({ status: "ok", service: "apex-pulse-dashboard-api", api_version: "v1" });
      }
      return jsonResponse(
        {
          detail: {
            code: "dashboard_artifact_invalid",
            message: "The requested dashboard artifact failed validation.",
            artifact_type: "historical_monitoring_summary"
          }
        },
        500
      );
    });

    const data = await loadMonitoringHistoryPageData();

    expect(data.error).toMatchObject({
      code: "dashboard_artifact_invalid",
      artifactType: "historical_monitoring_summary"
    });
  });
});

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "content-type": "application/json" }
  });
}
