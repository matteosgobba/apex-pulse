import { afterEach, describe, expect, test, vi } from "vitest";

import { fetchDashboard, loadForecastPageData, loadPracticePageData } from "@/lib/api";
import { DEFAULT_API_BASE_URL, getApiBaseUrl } from "@/lib/config";

const originalFetch = global.fetch;
const originalApiBaseUrl = process.env.NEXT_PUBLIC_APEX_PULSE_API_BASE_URL;

afterEach(() => {
  global.fetch = originalFetch;
  if (originalApiBaseUrl === undefined) {
    delete process.env.NEXT_PUBLIC_APEX_PULSE_API_BASE_URL;
  } else {
    process.env.NEXT_PUBLIC_APEX_PULSE_API_BASE_URL = originalApiBaseUrl;
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
});

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "content-type": "application/json" }
  });
}
