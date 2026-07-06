import { afterEach, describe, expect, test, vi } from "vitest";

import { fetchDashboard } from "@/lib/api";
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
});
