import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { hydrateRoot, type Root } from "react-dom/client";
import { renderToString } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import MethodologyPage from "@/app/methodology/page";
import { AppShell } from "@/components/app-shell";
import { CurrentEventPageView } from "@/components/current-event-page";
import { MonitoringHistoryPageView } from "@/components/monitoring-history-page";
import { ThemeProvider } from "@/components/theme-provider";
import { ThemeToggle } from "@/components/theme-toggle";
import type {
  CurrentEventEnvelope,
  CurrentEventPageData,
  ForecastEnvelope,
  HistoricalMonitoringEnvelope,
  MonitoringHistoryPageData,
  PracticeStatusEnvelope
} from "@/lib/dashboard-types";
import {
  DEFAULT_THEME,
  THEME_INITIALIZATION_SCRIPT,
  THEME_STORAGE_KEY,
  type Theme
} from "@/lib/theme";

import readyCurrentEvent from "./fixtures/current-event-ready.json";
import readyForecast from "./fixtures/event-forecast-ready.json";
import validAndLegacyHistory from "./fixtures/historical-monitoring-valid-and-legacy.json";
import readyPracticeStatus from "./fixtures/practice-status-ready.json";

beforeEach(() => {
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
  window.localStorage.clear();
  document.documentElement.dataset.theme = DEFAULT_THEME;
  document.documentElement.style.colorScheme = "";
  document.querySelector('meta[name="theme-color"]')?.remove();
  vi.restoreAllMocks();
});

describe("theme initialization", () => {
  test("dark is the default when no saved preference exists", () => {
    runThemeInitializer();

    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(document.documentElement.style.colorScheme).toBe("dark");
  });

  test("a valid stored light preference initializes light mode", () => {
    window.localStorage.setItem(THEME_STORAGE_KEY, "light");
    runThemeInitializer();

    expect(document.documentElement.dataset.theme).toBe("light");
    expect(themeColorContent()).toBe("#F7F8FA");
  });

  test("a valid stored dark preference initializes dark mode", () => {
    window.localStorage.setItem(THEME_STORAGE_KEY, "dark");
    runThemeInitializer();

    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(themeColorContent()).toBe("#070B12");
  });

  test("an invalid stored preference falls back safely to dark", () => {
    window.localStorage.setItem(THEME_STORAGE_KEY, "system");
    runThemeInitializer();

    expect(document.documentElement.dataset.theme).toBe("dark");
  });
});

describe("theme toggle", () => {
  test("changes dark to light, updates the root, stores the value, and updates its label", () => {
    renderToggle("dark");

    fireEvent.click(screen.getByRole("button", { name: "Switch to light mode" }));

    expect(document.documentElement.dataset.theme).toBe("light");
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe("light");
    expect(screen.getByRole("button", { name: "Switch to dark mode" })).toBeInTheDocument();
  });

  test("changes light to dark and persists dark", () => {
    renderToggle("light");

    fireEvent.click(screen.getByRole("button", { name: "Switch to dark mode" }));

    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe("dark");
    expect(screen.getByRole("button", { name: "Switch to light mode" })).toBeInTheDocument();
  });

  test("preference is restored across an unmount and initialization cycle", () => {
    const first = renderToggle("dark");
    fireEvent.click(screen.getByRole("button", { name: "Switch to light mode" }));
    first.unmount();

    document.documentElement.dataset.theme = "dark";
    runThemeInitializer();
    render(
      <ThemeProvider>
        <ThemeToggle />
      </ThemeProvider>
    );

    expect(document.documentElement.dataset.theme).toBe("light");
    expect(screen.getByRole("button", { name: "Switch to dark mode" })).toBeInTheDocument();
  });

  test("control is present in the shared header with a keyboard and mobile-sized target", () => {
    render(
      <AppShell health={null}>
        <p>Theme-aware page</p>
      </AppShell>
    );

    const toggle = screen.getByRole("button", { name: "Switch to light mode" });
    expect(toggle).toHaveAttribute("type", "button");
    expect(toggle).toHaveAttribute("title", "Switch to light mode");
    expect(toggle).toHaveClass("min-h-11", "min-w-11");
    expect(toggle).not.toHaveClass("hidden");
    expect(screen.getByRole("navigation", { name: "Primary navigation" })).toBeInTheDocument();
  });

  test("light initialization hydrates without a wrong-tree warning", async () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const tree = (
      <ThemeProvider>
        <ThemeToggle />
      </ThemeProvider>
    );
    const container = document.createElement("div");
    container.innerHTML = renderToString(tree);
    document.body.appendChild(container);
    window.localStorage.setItem(THEME_STORAGE_KEY, "light");
    runThemeInitializer();

    let root: Root | null = null;
    await act(async () => {
      root = hydrateRoot(container, tree);
      await Promise.resolve();
    });

    expect(
      consoleError.mock.calls.some((call) =>
        call.some((value) => String(value).toLowerCase().includes("hydration"))
      )
    ).toBe(false);
    expect(document.documentElement.dataset.theme).toBe("light");
    await act(async () => root?.unmount());
    container.remove();
  });
});

describe.each(["dark", "light"] as const)("public routes in %s mode", (theme) => {
  test("current event renders with its preserved forecast semantics", () => {
    setRootTheme(theme);
    render(<CurrentEventPageView data={currentEventData()} now={new Date("2026-07-24T12:00:00Z")} />);

    expect(screen.getByRole("heading", { name: "Predicted starting order" })).toBeInTheDocument();
    expect(document.documentElement.dataset.theme).toBe(theme);
  });

  test("history keeps prospective and legacy evidence separate", () => {
    setRootTheme(theme);
    const data: MonitoringHistoryPageData = {
      health: null,
      historicalMonitoring: validAndLegacyHistory as HistoricalMonitoringEnvelope,
      modelSummary: null,
      error: null
    };
    render(<MonitoringHistoryPageView data={data} />);

    expect(screen.getByText("Valid prospective evidence")).toBeInTheDocument();
    expect(screen.getByText(/Technical archive · 2 legacy descriptive records/i)).toBeInTheDocument();
    expect(document.documentElement.dataset.theme).toBe(theme);
  });

  test("methodology renders with technical details secondary", () => {
    setRootTheme(theme);
    render(<MethodologyPage />);

    expect(screen.getByRole("heading", { name: /pre-qualifying prediction/i })).toBeInTheDocument();
    expect(screen.getByText("Technical lifecycle details").closest("details")).not.toHaveAttribute(
      "open"
    );
    expect(document.documentElement.dataset.theme).toBe(theme);
  });
});

function renderToggle(theme: Theme) {
  setRootTheme(theme);
  return render(
    <ThemeProvider>
      <ThemeToggle />
    </ThemeProvider>
  );
}

function setRootTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.colorScheme = theme;
}

function runThemeInitializer(): void {
  document.querySelector('meta[name="theme-color"]')?.remove();
  const meta = document.createElement("meta");
  meta.name = "theme-color";
  document.head.appendChild(meta);
  Function(THEME_INITIALIZATION_SCRIPT)();
}

function themeColorContent(): string | null {
  return document.querySelector('meta[name="theme-color"]')?.getAttribute("content") ?? null;
}

function currentEventData(): CurrentEventPageData {
  return {
    health: null,
    manifest: null,
    currentEvent: readyCurrentEvent as CurrentEventEnvelope,
    practiceStatus: readyPracticeStatus as PracticeStatusEnvelope,
    forecast: readyForecast as ForecastEnvelope,
    settlement: null,
    historicalMonitoring: null,
    error: null
  };
}
