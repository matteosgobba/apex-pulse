import { act, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";

import { SessionCountdown } from "@/components/session-countdown";
import { TeamMark } from "@/components/team-mark";
import type { EventSchedule } from "@/lib/dashboard-types";
import { formatEventNameWithFlag } from "@/lib/event-display";
import { positionDeltaLabel } from "@/lib/public-view-model";
import {
  countdownParts,
  selectNextSession
} from "@/lib/session-schedule";
import { getTeamIdentity } from "@/lib/team-identity";

const SCHEDULE: EventSchedule = {
  available: true,
  source: "fastf1_cached_session_info",
  timezone: "UTC",
  sessions: [
    { session: "FP1", scheduled_start_utc: "2026-07-24T11:30:00Z" },
    { session: "FP2", scheduled_start_utc: "2026-07-24T15:00:00Z" },
    { session: "FP3", scheduled_start_utc: "2026-07-25T10:30:00Z" },
    { session: "Q", display_name: "Qualifying", scheduled_start_utc: "2026-07-25T14:00:00Z" }
  ]
};

afterEach(() => {
  vi.useRealTimers();
});

describe("session countdown", () => {
  test("selects FP1 before the weekend begins", () => {
    expect(
      selectNextSession(SCHEDULE, "practice_in_progress", new Date("2026-07-24T09:00:00Z"))
        .session?.session
    ).toBe("FP1");
  });

  test("selects the correct next session as the weekend progresses", () => {
    expect(
      selectNextSession(SCHEDULE, "forecast_available", new Date("2026-07-24T13:00:00Z"))
        .session?.session
    ).toBe("FP2");
    expect(
      selectNextSession(SCHEDULE, "forecast_available", new Date("2026-07-25T12:00:00Z"))
        .session?.session
    ).toBe("Q");
  });

  test("never returns negative countdown values and stops at zero", () => {
    expect(countdownParts(Date.parse("2026-07-24T10:00:00Z"), Date.parse("2026-07-24T10:00:01Z")))
      .toEqual({ days: 0, hours: 0, minutes: 0, seconds: 0, complete: true });
  });

  test("settled lifecycle shows completion rather than a negative countdown", () => {
    render(
      <SessionCountdown
        schedule={SCHEDULE}
        lifecycle="settled_partial_coverage"
        initialNow="2026-07-26T10:00:00Z"
      />
    );
    expect(screen.getByRole("heading", { name: "Qualifying complete" })).toBeInTheDocument();
  });

  test("missing schedule renders a graceful unavailable state", () => {
    render(
      <SessionCountdown
        schedule={null}
        lifecycle="forecast_available"
        initialNow="2026-07-24T10:00:00Z"
      />
    );
    expect(screen.getByRole("heading", { name: "Session schedule unavailable" })).toBeInTheDocument();
  });

  test("ticks each second, rolls to the next supplied session, and cleans up", () => {
    vi.useFakeTimers();
    const sprintSchedule: EventSchedule = {
      available: true,
      sessions: [
        {
          session: "SQ",
          display_name: "Sprint Qualifying",
          scheduled_start_utc: "2026-07-24T14:00:00Z"
        },
        {
          session: "S",
          display_name: "Sprint",
          scheduled_start_utc: "2026-07-24T15:00:00Z"
        }
      ]
    };
    const view = render(
      <SessionCountdown
        schedule={sprintSchedule}
        lifecycle="practice_in_progress"
        initialNow="2026-07-24T13:59:59Z"
      />
    );
    const countdown = screen.getByLabelText("Countdown to Sprint Qualifying");
    expect(within(countdown).getByText("01")).toBeInTheDocument();

    act(() => vi.advanceTimersByTime(1_000));

    expect(screen.getByRole("heading", { name: "Sprint" })).toBeInTheDocument();
    expect(vi.getTimerCount()).toBeGreaterThan(0);
    view.unmount();
    expect(vi.getTimerCount()).toBe(0);
  });
});

describe("public identity and position semantics", () => {
  test("known teams use the centralized expected identity", () => {
    const ferrari = getTeamIdentity("ferrari");
    expect(ferrari.displayName).toBe("Ferrari");
    expect(ferrari.primary).toBe("#E80020");
    expect(ferrari.foreground).toBe("#FFFFFF");
    expect(ferrari.logoPath).toBe("/teams/ferrari.svg.webp");
    expect(getTeamIdentity("Red Bull Racing").logoPath).toBe("/teams/redbull.png");
    expect(getTeamIdentity("RB").logoPath).toBe("/teams/racingbulls.png");
  });

  test("team mark renders a mapped logo with accessible alternative text", () => {
    const { container } = render(<TeamMark team={getTeamIdentity("mclaren")} />);

    expect(screen.getByRole("img", { name: "McLaren logo" })).toHaveAttribute(
      "src",
      "/teams/mclaren.svg"
    );
    expect(container.firstChild).toHaveStyle({ backgroundColor: "#2b2b30" });
  });

  test("unknown teams receive a deterministic readable fallback without a logo", () => {
    const first = getTeamIdentity("new_constructor", "New Constructor");
    const second = getTeamIdentity("new_constructor", "New Constructor");
    expect(first).toEqual(second);
    expect(first.known).toBe(false);
    expect(first.logoPath).toBeNull();
    expect(first.monogram).toBe("NC");
    expect(first.foreground).toBe("#FFFFFF");

    render(<TeamMark team={first} />);
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.getByLabelText("New Constructor team mark")).toHaveTextContent("NC");
  });

  test("position delta labels overprediction, underprediction, and exact matches correctly", () => {
    expect(positionDeltaLabel(2, 4)).toMatchObject({
      direction: "over",
      label: "Model overpredicted by 2 positions"
    });
    expect(positionDeltaLabel(5, 3)).toMatchObject({
      direction: "under",
      label: "Model underpredicted by 2 positions"
    });
    expect(positionDeltaLabel(3, 3)).toMatchObject({
      direction: "exact",
      label: "Exact position"
    });
  });
});

describe("event display", () => {
  test.each([
    ["Belgian Grand Prix", null, "Belgian Grand Prix\u00A0🇧🇪"],
    ["São Paulo Grand Prix", null, "São Paulo Grand Prix\u00A0🇧🇷"],
    ["Italy", "Italy", "Italy\u00A0🇮🇹"],
    ["Miami Grand Prix", "United States", "Miami Grand Prix\u00A0🇺🇸"]
  ])("formats %s with the correct host flag", (event, country, expected) => {
    expect(formatEventNameWithFlag(event, country)).toBe(expected);
  });

  test("leaves unknown event names unchanged", () => {
    expect(formatEventNameWithFlag("Synthetic Clean GP Final")).toBe("Synthetic Clean GP Final");
  });

  test("does not append a second flag", () => {
    expect(formatEventNameWithFlag("Hungarian Grand Prix 🇭🇺")).toBe(
      "Hungarian Grand Prix 🇭🇺"
    );
  });
});
