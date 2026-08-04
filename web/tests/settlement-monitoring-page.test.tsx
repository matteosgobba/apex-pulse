import { render, screen, within } from "@testing-library/react";
import { describe, expect, test } from "vitest";

import { MonitoringHistoryPageView } from "@/components/monitoring-history-page";
import { SettlementPageView } from "@/components/settlement-page";
import type {
  CurrentEventEnvelope,
  ForecastEnvelope,
  HistoricalMonitoringEnvelope,
  ModelSummaryEnvelope,
  MonitoringHistoryPageData,
  SettlementEnvelope,
  SettlementPageData
} from "@/lib/dashboard-types";

import legacyCurrentEvent from "./fixtures/current-event-legacy.json";
import readyCurrentEvent from "./fixtures/current-event-ready.json";
import availableSettlement from "./fixtures/event-settlement-available.json";
import legacySettlement from "./fixtures/event-settlement-legacy.json";
import unavailableSettlement from "./fixtures/event-settlement-unavailable.json";
import readyForecast from "./fixtures/event-forecast-ready.json";
import emptyHistory from "./fixtures/historical-monitoring-empty.json";
import validAndLegacyHistory from "./fixtures/historical-monitoring-valid-and-legacy.json";
import modelSummary from "./fixtures/model-summary-basic.json";

describe("SettlementPageView", () => {
  test("renders valid summary metrics", () => {
    renderSettlementPage();

    expect(
      screen.getByRole("heading", { name: "Italy\u00A0🇮🇹 Settlement", level: 1 })
    ).toBeInTheDocument();
    expect(screen.getByText("MAE gap to pole")).toBeInTheDocument();
    expect(screen.getAllByText("0.117 sec").length).toBeGreaterThan(0);
    expect(screen.getByText("Top-3 agreement")).toBeInTheDocument();
    expect(screen.getAllByText("100%").length).toBeGreaterThan(0);
    expect(screen.getByText("Settlement-evaluable drivers")).toBeInTheDocument();
    expect(screen.getByText("Excluded drivers")).toBeInTheDocument();
  });

  test("renders driver predicted-versus-actual comparison", () => {
    renderSettlementPage();

    expect(screen.getByText("Driver Comparison")).toBeInTheDocument();
    expect(screen.getAllByText("NOR").length).toBeGreaterThan(0);
    expect(screen.getAllByText("+0.468s").length).toBeGreaterThan(0);
    expect(screen.getAllByText("+0.234s").length).toBeGreaterThan(0);
  });

  test("position delta is displayed safely", () => {
    renderSettlementPage();

    expect(screen.getAllByText("1 worse").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Matched prediction").length).toBeGreaterThan(0);
  });

  test("partial target coverage is shown as audit-only settlement rows", () => {
    renderSettlementPage();

    expect(screen.getByText("Partial Settlement Coverage")).toBeInTheDocument();
    expect(screen.getByText(/2 settlement-evaluable drivers are shown/i)).toBeInTheDocument();
    expect(screen.getByText(/1 forecast-only rows are retained as audit-only records/i))
      .toBeInTheDocument();
    const comparison = screen.getByText("Driver Comparison").closest("section");
    expect(comparison).not.toBeNull();
    expect(within(comparison as HTMLElement).queryByText("ARO")).not.toBeInTheDocument();
  });

  test("unavailable settlement renders an informative state", () => {
    renderSettlementPage(unavailableSettlement as SettlementEnvelope);

    expect(screen.getByText("Settlement comparison unavailable.")).toBeInTheDocument();
    expect(screen.getByText(/qualifying targets are separately ingested/i)).toBeInTheDocument();
  });

  test("missing optional metrics render Not available", () => {
    renderSettlementPage();

    expect(screen.getAllByText("Not available").length).toBeGreaterThan(0);
    expect(screen.getByText("Mean interval width")).toBeInTheDocument();
  });

  test("legacy settlement record renders explicit noncanonical warning", () => {
    renderSettlementPage(
      legacySettlement as SettlementEnvelope,
      legacyCurrentEvent as CurrentEventEnvelope
    );

    expect(screen.getByText("Legacy Descriptive Record")).toBeInTheDocument();
    expect(
      screen.getByText(/Not eligible as valid prospective monitoring evidence/i)
    ).toBeInTheDocument();
  });
});

describe("MonitoringHistoryPageView", () => {
  test("separates valid prospective evidence from legacy records", () => {
    renderMonitoringPage();

    expect(screen.getByText("Valid prospective evidence")).toBeInTheDocument();
    expect(screen.getByText(/Technical archive · 2 legacy descriptive records/i)).toBeInTheDocument();
    expect(screen.getByText("Historical backtest context")).toBeInTheDocument();
  });

  test("Australia and Great Britain appear only in the legacy section", () => {
    renderMonitoringPage();

    const archive = screen
      .getByText(/Technical archive · 2 legacy descriptive records/i)
      .closest("details");
    expect(archive).not.toBeNull();
    expect(within(archive as HTMLElement).getByText(/Australia/)).toBeInTheDocument();
    expect(within(archive as HTMLElement).getByText(/Great Britain/)).toBeInTheDocument();

    const prospectiveSection = screen.getByText("Valid prospective evidence").closest("section");
    expect(prospectiveSection).not.toBeNull();
    expect(within(prospectiveSection as HTMLElement).queryByText("Australia")).not.toBeInTheDocument();
    expect(
      within(prospectiveSection as HTMLElement).queryByText("Great Britain")
    ).not.toBeInTheDocument();
  });

  test("legacy records do not affect valid prospective aggregate counts shown in UI", () => {
    renderMonitoringPage();

    expect(screen.getByText(/Prospective events:/)).toHaveTextContent("1");
    expect(screen.getByText(/Aggregate gap MAE:/)).toHaveTextContent("0.117s");
  });

  test("historical backtest context is visually distinct from prospective monitoring", () => {
    renderMonitoringPage();

    expect(screen.getByText("Model development evidence")).toBeInTheDocument();
    expect(screen.getByText(/not counted as prospective public predictions/i)).toBeInTheDocument();
    expect(screen.getByText("Walk-forward evaluation")).toBeInTheDocument();
  });

  test("empty monitoring history renders a useful prospective empty state", () => {
    renderMonitoringPage(emptyHistory as HistoricalMonitoringEnvelope);

    expect(screen.getByText("No prospective predictions yet")).toBeInTheDocument();
  });
});

function renderSettlementPage(
  settlement: SettlementEnvelope = availableSettlement as SettlementEnvelope,
  currentEvent: CurrentEventEnvelope = readyCurrentEvent as CurrentEventEnvelope
): void {
  const data: SettlementPageData = {
    health: null,
    currentEvent,
    forecast: readyForecast as ForecastEnvelope,
    settlement,
    error: null
  };
  render(<SettlementPageView data={data} />);
}

function renderMonitoringPage(
  historicalMonitoring: HistoricalMonitoringEnvelope =
    validAndLegacyHistory as HistoricalMonitoringEnvelope
): void {
  const data: MonitoringHistoryPageData = {
    health: null,
    historicalMonitoring,
    modelSummary: modelSummary as ModelSummaryEnvelope,
    error: null
  };
  render(<MonitoringHistoryPageView data={data} />);
}
