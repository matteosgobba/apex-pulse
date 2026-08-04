import { render, screen, within } from "@testing-library/react";
import { describe, expect, test } from "vitest";

import MethodologyPage from "@/app/methodology/page";
import { CurrentEventPageView } from "@/components/current-event-page";
import type {
  AutopilotStatusEnvelope,
  CurrentEventEnvelope,
  CurrentEventPageData,
  ForecastEnvelope,
  HealthResponse,
  PracticeStatusEnvelope,
  SettlementEnvelope
} from "@/lib/dashboard-types";

import emptyCurrentEvent from "./fixtures/current-event-empty.json";
import readyCurrentEvent from "./fixtures/current-event-ready.json";
import availableSettlement from "./fixtures/event-settlement-available.json";
import readyForecast from "./fixtures/event-forecast-ready.json";
import readyPracticeStatus from "./fixtures/practice-status-ready.json";

const HEALTH: HealthResponse = {
  status: "ok",
  service: "apex-pulse-dashboard-api",
  api_version: "v1",
  dashboard_artifact_status: "complete"
};
const TEST_NOW = new Date("2026-07-24T12:00:00Z");

describe("CurrentEventPageView", () => {
  test("homepage renders the artifact event and makes the forecast central", () => {
    renderPage();

    expect(screen.getByRole("heading", { name: "Italy\u00A0🇮🇹", level: 1 })).toBeInTheDocument();
    expect(screen.getByText("Practice data ready")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Predicted starting order" })).toBeInTheDocument();
    expect(screen.getAllByText("NOR").length).toBeGreaterThan(0);
  });

  test("forecast ranking is ordered by predicted position", () => {
    const reversed = {
      ...(readyForecast as ForecastEnvelope),
      data: {
        ...(readyForecast as ForecastEnvelope).data,
        qualifying_eligible_forecast_rows: [
          ...((readyForecast as ForecastEnvelope).data.qualifying_eligible_forecast_rows as [])
        ].reverse()
      }
    } as ForecastEnvelope;
    renderPage({ forecast: reversed });

    const ranking = screen.getByRole("list", { name: "Predicted qualifying ranking" });
    const rows = within(ranking).getAllByRole("listitem");
    expect(within(rows[0]).getByText("NOR")).toBeInTheDocument();
    expect(within(rows[1]).getByText("VER")).toBeInTheDocument();
  });

  test("settled event shows predicted and official positions with partial coverage", () => {
    renderPage({
      currentEvent: settledCurrentEvent(),
      settlement: partialSettlement()
    });

    expect(screen.getByRole("heading", { name: "How the forecast compared" })).toBeInTheDocument();
    expect(screen.getByText("Forecast coverage: 2/3")).toBeInTheDocument();
    expect(screen.getByText(/PER appeared in the official result/i)).toBeInTheDocument();
    expect(screen.getByLabelText("Model overpredicted by 1 position")).toBeInTheDocument();
    const comparison = screen.getByRole("list", {
      name: "Prediction and official result comparison"
    });
    expect(within(comparison).getByText("McLaren")).toBeInTheDocument();
  });

  test("terminal settled artifacts keep neutral age text without a stale warning", () => {
    renderPage({
      currentEvent: {
        ...settledCurrentEvent(),
        generated_at_utc: "2026-07-20T12:00:00Z"
      },
      settlement: partialSettlement()
    });

    expect(screen.getByText(/Settled · Updated 4 days ago/i)).toBeInTheDocument();
    expect(screen.queryByText(/Data may be stale/i)).not.toBeInTheDocument();
  });

  test("active old forecast retains the stale-data warning", () => {
    renderPage({
      currentEvent: {
        ...(readyCurrentEvent as CurrentEventEnvelope),
        generated_at_utc: "2026-07-20T12:00:00Z"
      }
    });

    expect(screen.getByText(/Data may be stale/i)).toBeInTheDocument();
  });

  test("Dutch Sprint is operational while Hungary remains the latest immutable result", () => {
    renderPage({
      operationalStatus: sprintOperationalStatus(),
      currentEvent: hungarianSettledCurrentEvent(),
      settlement: partialSettlement()
    });

    expect(
      screen.getByRole("heading", { name: "Dutch Grand Prix\u00A0🇳🇱", level: 1 })
    ).toBeInTheDocument();
    expect(screen.getByText("Current weekend")).toBeInTheDocument();
    expect(screen.getByText("Sprint weekend")).toBeInTheDocument();
    expect(
      screen.getByText(/does not create predictions for Sprint weekends yet/i)
    ).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Sprint Qualifying" })).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Hungarian Grand Prix\u00A0🇭🇺", level: 2 })
    ).toBeInTheDocument();
    expect(screen.queryByText("UNSUPPORTED_WEEKEND_FORMAT")).not.toBeInTheDocument();
    expect(screen.getByText("Forecast coverage: 2/3")).toBeInTheDocument();
  });

  test("unavailable operational status leaves the latest result functional", () => {
    renderPage({
      operationalStatus: {
        schema_version: "1.0",
        status: "not_initialized",
        data: {}
      },
      currentEvent: hungarianSettledCurrentEvent(),
      settlement: partialSettlement()
    });

    expect(screen.queryByLabelText("Operational Formula 1 weekend")).not.toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Hungarian Grand Prix\u00A0🇭🇺", level: 1 })
    ).toBeInTheDocument();
  });

  test("missing official entrant is never given a retrospective prediction", () => {
    renderPage({
      currentEvent: settledCurrentEvent(),
      settlement: partialSettlement()
    });

    const forecast = screen.getByRole("list", { name: "Predicted qualifying ranking" });
    expect(within(forecast).queryByText("PER")).not.toBeInTheDocument();
    expect(screen.getAllByText(/No retrospective prediction has been added/i)).toHaveLength(1);
  });

  test("missing settlement keeps a prediction-only layout", () => {
    renderPage({ settlement: null });

    expect(screen.getByRole("heading", { name: "Predicted starting order" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "How the forecast compared" })).not.toBeInTheDocument();
  });

  test("missing forecast produces a graceful empty state", () => {
    renderPage({ forecast: null });

    expect(screen.getByText("Prediction unavailable")).toBeInTheDocument();
    expect(screen.getByText(/No forecast rows are present/i)).toBeInTheDocument();
  });

  test("missing current artifact and API error render safe public states", () => {
    const { rerender } = renderPage({
      currentEvent: emptyCurrentEvent as CurrentEventEnvelope,
      forecast: null
    });
    expect(screen.getByText("No current prediction is available")).toBeInTheDocument();

    rerender(
      <CurrentEventPageView
        data={{
          ...baseData(),
          currentEvent: null,
          forecast: null,
          error: { code: "dashboard_api_unavailable", message: "fetch failed /private/tmp" }
        }}
        now={TEST_NOW}
      />
    );
    expect(screen.getByText("Prediction data is unavailable")).toBeInTheDocument();
    expect(screen.queryByText(/private\/tmp/i)).not.toBeInTheDocument();
  });

  test("weekend progression is responsive content without a wide table", () => {
    const { container } = renderPage();

    expect(screen.getByRole("heading", { name: "Practice to qualifying" })).toBeInTheDocument();
    expect(screen.getAllByText("FP1").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Qualifying").length).toBeGreaterThan(0);
    expect(container.querySelector("table")).toBeNull();
  });

  test("contact renders the public profile links and technical details stay secondary", () => {
    renderPage();

    expect(screen.getByText("Built by Matteo Sgobba")).toBeInTheDocument();
    expect(screen.getAllByRole("link", { name: "GitHub" }).length).toBeGreaterThan(0);
    expect(screen.getAllByRole("link", { name: "Email" }).length).toBeGreaterThan(0);
    const linkedIn = screen.getAllByRole("link", { name: "LinkedIn" });
    expect(linkedIn.length).toBeGreaterThan(0);
    expect(linkedIn[0]).toHaveAttribute("href", "https://www.linkedin.com/in/matteosgobba/");
    expect(linkedIn[0]).toHaveAttribute("target", "_blank");
    expect(linkedIn[0]).toHaveAttribute("rel", expect.stringContaining("noopener"));
    expect(screen.getByText("Artifact details").closest("details")).not.toHaveAttribute("open");
  });
});

describe("MethodologyPage", () => {
  test("presents public methodology and limitations with technical details secondary", () => {
    render(<MethodologyPage />);

    expect(screen.getByRole("heading", { name: /pre-qualifying prediction/i })).toBeInTheDocument();
    expect(screen.getByText(/no private team data/i)).toBeInTheDocument();
    expect(screen.getByText(/never rewrites or fills missing predictions/i)).toBeInTheDocument();
    expect(screen.getByText(/legacy noncanonical records/i)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Evaluation metrics" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Public-data limitations" })).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Prospective evidence and backtests" })
    ).toBeInTheDocument();
    expect(screen.getByText("Technical lifecycle details").closest("details")).not.toHaveAttribute(
      "open"
    );
    expect(screen.getByRole("link", { name: "History" })).toHaveAttribute("href", "/history");
  });
});

function renderPage(overrides: Partial<CurrentEventPageData> = {}) {
  return render(
    <CurrentEventPageView data={{ ...baseData(), ...overrides }} now={TEST_NOW} />
  );
}

function baseData(): CurrentEventPageData {
  return {
    health: HEALTH,
    manifest: null,
    currentEvent: readyCurrentEvent as CurrentEventEnvelope,
    practiceStatus: readyPracticeStatus as PracticeStatusEnvelope,
    forecast: readyForecast as ForecastEnvelope,
    settlement: null,
    historicalMonitoring: null,
    error: null
  };
}

function settledCurrentEvent(): CurrentEventEnvelope {
  return {
    ...(readyCurrentEvent as CurrentEventEnvelope),
    data: {
      ...(readyCurrentEvent as CurrentEventEnvelope).data,
      lifecycle: {
        state: "settled_partial_coverage",
        display_label: "Settled with partial coverage",
        reason: "partial_coverage"
      },
      settlement_status: {
        available: true,
        settlement_valid: true,
        scored_driver_count: 2,
        actual_qualifying_driver_count: 3,
        forecast_coverage: "2/3",
        forecast_coverage_percentage: 66.67,
        unforecasted_actual_entrants: [{ driver: "PER", driver_code: "PER" }]
      }
    }
  };
}

function hungarianSettledCurrentEvent(): CurrentEventEnvelope {
  const current = settledCurrentEvent();
  return {
    ...current,
    data: {
      ...current.data,
      event_identity: {
        ...current.data.event_identity,
        season: 2026,
        event: "Hungarian Grand Prix",
        event_slug: "hungarian-grand-prix",
        event_order: 11
      }
    }
  };
}

function sprintOperationalStatus(): AutopilotStatusEnvelope {
  return {
    schema_version: "1.0",
    status: "available",
    data: {
      orchestrator_state_after: "UNSUPPORTED_WEEKEND_FORMAT",
      operational_event: {
        season: 2026,
        event: "Dutch Grand Prix",
        event_slug: "dutch-grand-prix",
        round_number: 12,
        event_format: "sprint_qualifying",
        calendar_source: "fastf1_event_schedule",
        supported: false,
        prediction_support_reason: "Synthetic unsupported format reason.",
        schedule_available: true,
        timezone: "UTC",
        sessions: [
          {
            sequence: 1,
            session: "FP1",
            display_name: "Practice 1",
            scheduled_start_utc: "2026-07-24T10:00:00+00:00",
            scheduled_end_utc: "2026-07-24T11:00:00+00:00",
            end_source: "fastf1_schedule",
            schedule_status: "calendar_time_elapsed_data_not_proven"
          },
          {
            sequence: 2,
            session: "SQ",
            display_name: "Sprint Qualifying",
            scheduled_start_utc: "2026-07-24T14:00:00+00:00",
            scheduled_end_utc: "2026-07-24T15:00:00+00:00",
            end_source: "fastf1_schedule",
            schedule_status: "scheduled"
          },
          {
            sequence: 3,
            session: "S",
            display_name: "Sprint",
            scheduled_start_utc: "2026-07-25T10:00:00+00:00",
            scheduled_end_utc: "2026-07-25T11:00:00+00:00",
            end_source: "fastf1_schedule",
            schedule_status: "scheduled"
          },
          {
            sequence: 4,
            session: "Q",
            display_name: "Qualifying",
            scheduled_start_utc: "2026-07-25T14:00:00+00:00",
            scheduled_end_utc: "2026-07-25T15:30:00+00:00",
            end_source: "fastf1_schedule",
            schedule_status: "scheduled"
          }
        ]
      }
    }
  };
}

function partialSettlement(): SettlementEnvelope {
  const settlement = availableSettlement as SettlementEnvelope;
  const comparisons = (settlement.data.settlement_evaluable_rows as Array<
    Record<string, unknown>
  >).map((row) => ({ ...row, team: null, team_key: null }));
  return {
    ...settlement,
    data: {
      ...settlement.data,
      lifecycle_state: "settled_partial_coverage",
      summary_metrics: {
        ...settlement.data.summary_metrics,
        actual_qualifying_driver_count: 3,
        evaluable_driver_count: 2,
        forecast_coverage: "2/3",
        forecast_coverage_percentage: 66.67,
        forecast_coverage_status: "partial_coverage",
        unforecasted_actual_entrants: [{ driver: "PER", driver_code: "PER" }]
      },
      unforecasted_actual_entrants: [
        {
          driver: "PER",
          driver_code: "PER",
          reason: "pre_q_entry_list_resolution_miss"
        }
      ],
      driver_comparison: comparisons,
      settlement_evaluable_rows: comparisons
    }
  } as SettlementEnvelope;
}
