import { act, cleanup, render, screen } from "@testing-library/react";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import {
  DASHBOARD_REFRESH_INTERVAL_MS,
  DashboardAutoRefresh
} from "@/components/dashboard-auto-refresh";
import { ThemeProvider } from "@/components/theme-provider";
import { THEME_STORAGE_KEY } from "@/lib/theme";

const mockedUseRouter = vi.mocked(useRouter);

describe("dashboard automatic refresh", () => {
  let refresh: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.useFakeTimers();
    refresh = vi.fn();
    mockedUseRouter.mockReturnValue({
      back: vi.fn(),
      forward: vi.fn(),
      prefetch: vi.fn(),
      push: vi.fn(),
      refresh,
      replace: vi.fn()
    });
    setVisibility("visible");
    const values = new Map<string, string>();
    Object.defineProperty(window, "localStorage", {
      configurable: true,
      value: {
        clear: () => values.clear(),
        getItem: (key: string) => values.get(key) ?? null,
        key: (index: number) => Array.from(values.keys())[index] ?? null,
        get length() {
          return values.size;
        },
        removeItem: (key: string) => values.delete(key),
        setItem: (key: string, value: string) => values.set(key, String(value))
      } satisfies Storage
    });
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  test("refreshes server data on the configured cadence without a full navigation", async () => {
    render(<DashboardAutoRefresh />);

    await act(async () => {
      vi.advanceTimersByTime(DASHBOARD_REFRESH_INTERVAL_MS);
      await Promise.resolve();
    });

    expect(refresh).toHaveBeenCalledTimes(1);
  });

  test("pauses while hidden and refreshes immediately when visible again", async () => {
    render(<DashboardAutoRefresh />);
    setVisibility("hidden");

    await act(async () => {
      vi.advanceTimersByTime(DASHBOARD_REFRESH_INTERVAL_MS * 2);
    });
    expect(refresh).not.toHaveBeenCalled();

    setVisibility("visible");
    await act(async () => {
      document.dispatchEvent(new Event("visibilitychange"));
      await Promise.resolve();
    });
    expect(refresh).toHaveBeenCalledTimes(1);
  });

  test("refreshes on focus and removes timers and listeners on unmount", async () => {
    const view = render(<DashboardAutoRefresh />);
    await act(async () => {
      window.dispatchEvent(new Event("focus"));
      await Promise.resolve();
    });
    expect(refresh).toHaveBeenCalledTimes(1);

    view.unmount();
    await act(async () => {
      vi.advanceTimersByTime(DASHBOARD_REFRESH_INTERVAL_MS);
      window.dispatchEvent(new Event("focus"));
    });
    expect(refresh).toHaveBeenCalledTimes(1);
  });

  test("router refresh preserves the active theme provider state", async () => {
    window.localStorage.setItem(THEME_STORAGE_KEY, "light");
    document.documentElement.dataset.theme = "light";
    render(
      <ThemeProvider>
        <DashboardAutoRefresh intervalMs={1_000} />
      </ThemeProvider>
    );

    await act(async () => {
      vi.advanceTimersByTime(1_000);
      await Promise.resolve();
    });

    expect(refresh).toHaveBeenCalledTimes(1);
    expect(document.documentElement.dataset.theme).toBe("light");
  });

  test("a forecast-only fixture can become settled partial coverage without navigation", async () => {
    function ControlledDashboardFixture() {
      const [state, setState] = useState("forecast_available");
      useEffect(() => {
        refresh.mockImplementation(() => setState("settled_partial_coverage"));
      }, []);
      return (
        <>
          <DashboardAutoRefresh intervalMs={1_000} />
          <p>{state}</p>
        </>
      );
    }

    render(<ControlledDashboardFixture />);
    expect(screen.getByText("forecast_available")).toBeInTheDocument();

    await act(async () => {
      vi.advanceTimersByTime(1_000);
      await Promise.resolve();
    });

    expect(screen.getByText("settled_partial_coverage")).toBeInTheDocument();
  });
});

function setVisibility(value: DocumentVisibilityState) {
  Object.defineProperty(document, "visibilityState", {
    configurable: true,
    value
  });
}
