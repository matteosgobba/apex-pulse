import { render, screen, within } from "@testing-library/react";
import { describe, expect, test } from "vitest";

import { ForecastLeaderboard } from "@/components/forecast-leaderboard";
import { ForecastPageView } from "@/components/forecast-page";
import { PracticePageView } from "@/components/practice-page";
import { dashboardRows } from "@/lib/dashboard-collections";
import type {
  CurrentEventEnvelope,
  ForecastEnvelope,
  ForecastPageData,
  PracticePageData,
  PracticeStatusEnvelope
} from "@/lib/dashboard-types";
import { teamAccent } from "@/lib/team-accent";

import blockedCurrentEvent from "./fixtures/current-event-blocked.json";
import legacyCurrentEvent from "./fixtures/current-event-legacy.json";
import readyCurrentEvent from "./fixtures/current-event-ready.json";
import emptyForecast from "./fixtures/event-forecast-empty.json";
import readyForecast from "./fixtures/event-forecast-ready.json";
import readyPracticeStatus from "./fixtures/practice-status-ready.json";

describe("ForecastPageView", () => {
  test("renders a valid leaderboard with predicted position, driver, team, and gap", () => {
    renderForecastPage();

    expect(screen.getByText("Italy Forecast")).toBeInTheDocument();
    expect(screen.getAllByText("NOR").length).toBeGreaterThan(0);
    expect(screen.getAllByText("McLaren").length).toBeGreaterThan(0);
    expect(screen.getAllByText("+0.000s").length).toBeGreaterThan(0);
  });

  test("interval renders when available", () => {
    renderForecastPage();

    expect(screen.getAllByText("-0.050s to +0.080s").length).toBeGreaterThan(0);
  });

  test("unavailable interval renders safely", () => {
    renderForecastPage();

    expect(screen.getAllByText("Interval not available").length).toBeGreaterThan(0);
  });

  test("handles no forecast rows without crashing", () => {
    renderForecastPage(emptyForecast as ForecastEnvelope);

    expect(screen.getByText("Forecast leaderboard unavailable.")).toBeInTheDocument();
  });

  test("forecast-only FP participant is shown separately from the public leaderboard", () => {
    renderForecastPage();

    expect(screen.getByText("Forecast-Only Audit Rows")).toBeInTheDocument();
    expect(screen.getByText("ARO")).toBeInTheDocument();
    expect(screen.getByText("No Qualifying Lap Rows")).toBeInTheDocument();
    expect(screen.queryByText("Predicted P3")).not.toBeInTheDocument();
  });

  test("settlement-only columns are absent before settlement", () => {
    renderForecastPage();

    expect(screen.queryByText("Actual position")).not.toBeInTheDocument();
    expect(screen.queryByText("Absolute gap error")).not.toBeInTheDocument();
  });

  test("legacy forecast state displays explicit noncanonical warning", () => {
    renderForecastPage(readyForecast as ForecastEnvelope, legacyCurrentEvent as CurrentEventEnvelope);

    expect(screen.getByText("Legacy Descriptive Record")).toBeInTheDocument();
    expect(
      screen.getByText(/Not eligible as valid prospective monitoring evidence/i)
    ).toBeInTheDocument();
  });

  test("mobile-oriented leaderboard rendering preserves essential fields", () => {
    render(
      <ForecastLeaderboard
        rows={dashboardRows((readyForecast as ForecastEnvelope).data.leaderboard)}
      />
    );

    expect(screen.getByText("Predicted P1")).toBeInTheDocument();
    expect(screen.getAllByText("NOR").length).toBeGreaterThan(0);
    expect(screen.getAllByText("McLaren").length).toBeGreaterThan(0);
    expect(screen.getAllByText("+0.000s").length).toBeGreaterThan(0);
  });

  test("team accent mapping has a neutral fallback", () => {
    expect(teamAccent("unknown-team")).toMatch(/^#[0-9A-F]{6}$/);
    expect(teamAccent("unknown-team")).toBe(teamAccent("unknown-team"));
  });
});

describe("PracticePageView", () => {
  test("renders FP1, FP2, FP3, and Q in correct order", () => {
    renderPracticePage();

    const timeline = screen.getByText("FP1 to Qualifying Progression").closest("section");
    expect(timeline).not.toBeNull();
    const headings = within(timeline as HTMLElement)
      .getAllByRole("heading", { level: 3 })
      .map((heading) => heading.textContent);
    expect(headings).toEqual(["FP1", "FP2", "FP3", "Q"]);
  });

  test("ready-to-forecast state shows read-only operator-workflow messaging", () => {
    renderPracticePage();

    expect(screen.getByText(/A forecast can be created through the separate operator workflow/i))
      .toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /forecast/i })).not.toBeInTheDocument();
  });

  test("blocked state shows safe blocking information", () => {
    const blockedPractice = {
      ...(readyPracticeStatus as PracticeStatusEnvelope),
      data: {
        ...(readyPracticeStatus as PracticeStatusEnvelope).data,
        lifecycle_state: "blocked",
        preflight: (blockedCurrentEvent as CurrentEventEnvelope).data.preflight
      }
    } as PracticeStatusEnvelope;

    renderPracticePage(blockedCurrentEvent as CurrentEventEnvelope, blockedPractice);

    expect(screen.getByText(/Preflight blocked the forecast workflow/i)).toBeInTheDocument();
    expect(
      screen.getByText(/exported preflight runbook/i)
    ).toBeInTheDocument();
  });

  test("forecast-available state links appropriately to forecast view", () => {
    const forecastAvailableCurrent = {
      ...(readyCurrentEvent as CurrentEventEnvelope),
      data: {
        ...(readyCurrentEvent as CurrentEventEnvelope).data,
        lifecycle: {
          state: "forecast_available",
          display_label: "Forecast available",
          reason: "forecast_snapshot_available"
        },
        forecast_status: {
          available: true,
          checkpoint: "after_fp3",
          forecast_created_at_utc: "2026-07-06T12:04:00+00:00",
          forecasted_driver_count: 2
        }
      }
    } as CurrentEventEnvelope;

    renderPracticePage(forecastAvailableCurrent);

    const link = screen.getByRole("link", { name: "Open forecast view" });
    expect(link).toHaveAttribute("href", "/forecast");
  });
});

function renderForecastPage(
  forecast: ForecastEnvelope = readyForecast as ForecastEnvelope,
  currentEvent: CurrentEventEnvelope = readyCurrentEvent as CurrentEventEnvelope
): void {
  const data: ForecastPageData = {
    health: null,
    currentEvent,
    forecast,
    error: null
  };
  render(<ForecastPageView data={data} />);
}

function renderPracticePage(
  currentEvent: CurrentEventEnvelope = readyCurrentEvent as CurrentEventEnvelope,
  practiceStatus: PracticeStatusEnvelope = readyPracticeStatus as PracticeStatusEnvelope
): void {
  const data: PracticePageData = {
    health: null,
    currentEvent,
    practiceStatus,
    error: null
  };
  render(<PracticePageView data={data} />);
}
