import { render, screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";

import MethodologyPage from "@/app/methodology/page";
import { CurrentEventPageView } from "@/components/current-event-page";
import type {
  CurrentEventEnvelope,
  CurrentEventPageData,
  ForecastEnvelope,
  HistoricalMonitoringEnvelope,
  HealthResponse,
  PracticeStatusEnvelope,
  SettlementEnvelope
} from "@/lib/dashboard-types";

import blockedCurrentEvent from "./fixtures/current-event-blocked.json";
import emptyCurrentEvent from "./fixtures/current-event-empty.json";
import legacyCurrentEvent from "./fixtures/current-event-legacy.json";
import readyCurrentEvent from "./fixtures/current-event-ready.json";
import readyForecast from "./fixtures/event-forecast-ready.json";
import availableSettlement from "./fixtures/event-settlement-available.json";
import validHistory from "./fixtures/historical-monitoring-valid-and-legacy.json";
import readyPracticeStatus from "./fixtures/practice-status-ready.json";

const HEALTH: HealthResponse = {
  status: "ok",
  service: "apex-pulse-dashboard-api",
  api_version: "v1",
  dashboard_artifact_status: "complete"
};

describe("CurrentEventPageView", () => {
  test("ready-to-forecast current event renders lifecycle badge and read-only readiness messaging", () => {
    renderPage(readyCurrentEvent);

    expect(screen.getByText("Italy")).toBeInTheDocument();
    expect(screen.getByText("Ready to forecast")).toBeInTheDocument();
    expect(screen.getByText("Ready in the operator workflow")).toBeInTheDocument();
    expect(screen.getByText(/does not execute the forecast command/i)).toBeInTheDocument();
  });

  test("blocked current event renders blocking explanation", () => {
    renderPage(blockedCurrentEvent);

    expect(screen.getByText("Belgium")).toBeInTheDocument();
    expect(screen.getByText("Forecast generation is blocked")).toBeInTheDocument();
    expect(screen.getByText(/Missing Required Practice Sessions/i)).toBeInTheDocument();
    expect(
      screen.getAllByText(/review prospective_monitoring_preflight_runbook.md/i)
    ).toHaveLength(2);
    expect(
      screen.getByText(/Next operator action: review prospective_monitoring_preflight_runbook.md/i)
    ).toBeInTheDocument();
  });

  test("legacy current event renders the noncanonical warning", () => {
    renderPage(legacyCurrentEvent);

    expect(screen.getByText("Great Britain")).toBeInTheDocument();
    expect(screen.getByText("Legacy Descriptive Record")).toBeInTheDocument();
    expect(
      screen.getByText(/Not eligible as valid prospective monitoring evidence/i)
    ).toBeInTheDocument();
  });

  test("empty event renders a useful empty state", () => {
    renderPage(emptyCurrentEvent);

    expect(screen.getByText("No Current Event")).toBeInTheDocument();
    expect(
      screen.getByText("No monitored event is currently available.")
    ).toBeInTheDocument();
  });

  test("missing optional KPI values render Not available", () => {
    renderPage(readyCurrentEvent);

    expect(screen.getAllByText("Not available").length).toBeGreaterThanOrEqual(4);
  });

  test("practice session availability renders FP1, FP2, FP3, and Q consistently", () => {
    renderPage(readyCurrentEvent);

    expect(screen.getByRole("heading", { name: "FP1" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "FP2" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "FP3" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Q" })).toBeInTheDocument();
    expect(screen.getByText(/Target Artifact Not Available/i)).toBeInTheDocument();
  });

  test("home-page compact preview renders only when forecast rows exist", () => {
    renderPage(readyCurrentEvent);

    expect(screen.getByText("Forecast Preview")).toBeInTheDocument();
    expect(screen.getByText("Open forecast")).toBeInTheDocument();
    expect(screen.getByText("Open practice status")).toBeInTheDocument();
    expect(screen.getAllByText("NOR").length).toBeGreaterThan(0);
  });

  test("home page displays settlement preview only when relevant", () => {
    renderPage(readyCurrentEvent, {
      settlement: availableSettlement as SettlementEnvelope
    });

    expect(screen.getByText("Settlement Preview")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open settlement" })).toHaveAttribute(
      "href",
      "/settlement"
    );
  });

  test("home page displays monitoring preview only with valid prospective history", () => {
    renderPage(readyCurrentEvent, {
      historicalMonitoring: validHistory as HistoricalMonitoringEnvelope
    });

    expect(screen.getByText("Monitoring History Preview")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open monitoring history" })).toHaveAttribute(
      "href",
      "/monitoring-history"
    );
  });

  test("API-unavailable state renders a polished safe message without raw fetch details", () => {
    renderPage(readyCurrentEvent, {
      currentEvent: null,
      practiceStatus: null,
      forecast: null,
      error: {
        code: "dashboard_api_unavailable",
        message: "The dashboard API is unavailable. Start the read-only API server first."
      }
    });

    expect(screen.getByText("Dashboard API Unavailable")).toBeInTheDocument();
    expect(screen.getByText(/not currently reachable or cannot serve this artifact safely/i))
      .toBeInTheDocument();
    expect(screen.queryByText(/fetch failed/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/private\/tmp/i)).not.toBeInTheDocument();
  });
});

describe("MethodologyPage", () => {
  test("renders key trust and limitations statements", () => {
    render(<MethodologyPage />);

    expect(screen.getByText("Methodology And Trust")).toBeInTheDocument();
    expect(screen.getByText(/pre-qualifying information/i)).toBeInTheDocument();
    expect(screen.getByText(/does not claim private team telemetry/i)).toBeInTheDocument();
    expect(screen.getByText(/Legacy Australia and Great Britain/i)).toBeInTheDocument();
  });

  test("navigation includes settlement and monitoring history", () => {
    render(<MethodologyPage />);

    expect(screen.getByRole("link", { name: "Settlement" })).toHaveAttribute(
      "href",
      "/settlement"
    );
    expect(screen.getByRole("link", { name: "Monitoring History" })).toHaveAttribute(
      "href",
      "/monitoring-history"
    );
  });
});

function renderPage(
  currentEvent: unknown,
  overrides: Partial<CurrentEventPageData> = {}
): void {
  const data: CurrentEventPageData = {
    health: HEALTH,
    manifest: null,
    currentEvent: currentEvent as CurrentEventEnvelope,
    practiceStatus: readyPracticeStatus as PracticeStatusEnvelope,
    forecast: readyForecast as ForecastEnvelope,
    settlement: null,
    historicalMonitoring: null,
    error: null
  };
  render(<CurrentEventPageView data={{ ...data, ...overrides }} />);
}
