import { render, screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";

import MethodologyPage from "@/app/methodology/page";
import { CurrentEventPageView } from "@/components/current-event-page";
import type {
  CurrentEventEnvelope,
  CurrentEventPageData,
  HealthResponse,
  PracticeStatusEnvelope
} from "@/lib/dashboard-types";

import blockedCurrentEvent from "./fixtures/current-event-blocked.json";
import emptyCurrentEvent from "./fixtures/current-event-empty.json";
import legacyCurrentEvent from "./fixtures/current-event-legacy.json";
import readyCurrentEvent from "./fixtures/current-event-ready.json";
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
});

describe("MethodologyPage", () => {
  test("renders key trust and limitations statements", () => {
    render(<MethodologyPage />);

    expect(screen.getByText("Methodology And Trust")).toBeInTheDocument();
    expect(screen.getByText(/pre-qualifying information/i)).toBeInTheDocument();
    expect(screen.getByText(/does not claim private team telemetry/i)).toBeInTheDocument();
    expect(screen.getByText(/Legacy Australia and Great Britain/i)).toBeInTheDocument();
  });
});

function renderPage(currentEvent: unknown): void {
  const data: CurrentEventPageData = {
    health: HEALTH,
    manifest: null,
    currentEvent: currentEvent as CurrentEventEnvelope,
    practiceStatus: readyPracticeStatus as PracticeStatusEnvelope,
    error: null
  };
  render(<CurrentEventPageView data={data} />);
}
