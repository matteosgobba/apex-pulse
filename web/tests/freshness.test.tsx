import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";

import { AppShell } from "@/components/app-shell";
import { DataFreshnessNotice } from "@/components/data-freshness-notice";
import { FreshnessIndicator } from "@/components/freshness-indicator";
import { DEFAULT_STALE_AFTER_MINUTES, getDashboardStaleAfterMinutes } from "@/lib/config";
import { evaluateFreshness } from "@/lib/freshness";

const originalStaleAfter = process.env.NEXT_PUBLIC_APEX_PULSE_DASHBOARD_STALE_AFTER_MINUTES;

afterEach(() => {
  vi.useRealTimers();
  if (originalStaleAfter === undefined) {
    delete process.env.NEXT_PUBLIC_APEX_PULSE_DASHBOARD_STALE_AFTER_MINUTES;
  } else {
    process.env.NEXT_PUBLIC_APEX_PULSE_DASHBOARD_STALE_AFTER_MINUTES = originalStaleAfter;
  }
});

describe("dashboard freshness", () => {
  test("fresh artifact timestamp renders relative and exact accessible time", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-07-06T12:18:00Z"));

    render(<FreshnessIndicator generatedAt="2026-07-06T12:00:00Z" />);

    expect(screen.getByText("Updated 18 minutes ago")).toBeInTheDocument();
    expect(screen.getByText("Fresh")).toBeInTheDocument();
    expect(screen.getByText(/UTC:/)).toBeInTheDocument();
    expect(screen.getByText(/operator artifact-export workflow/i)).toBeInTheDocument();
  });

  test("aging artifact renders non-alarming freshness state", () => {
    const freshness = evaluateFreshness("2026-07-06T10:30:00Z", {
      now: new Date("2026-07-06T12:00:00Z"),
      staleAfterMinutes: 180
    });

    expect(freshness.state).toBe("aging");
    expect(freshness.relativeLabel).toBe("Updated 1 hour ago");
  });

  test("stale artifact renders an understandable public notice", () => {
    const freshness = evaluateFreshness("2026-07-06T08:30:00Z", {
      now: new Date("2026-07-06T12:00:00Z"),
      staleAfterMinutes: 180
    });
    render(<DataFreshnessNotice freshness={freshness} />);

    expect(screen.getByText(/Data may be stale/i)).toBeInTheDocument();
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  test("terminal lifecycle can render old artifact age as settled rather than stale", () => {
    const freshness = evaluateFreshness("2026-07-02T12:00:00Z", {
      now: new Date("2026-07-06T12:00:00Z"),
      staleAfterMinutes: 180
    });
    render(<DataFreshnessNotice freshness={freshness} terminal />);

    expect(screen.getByText("Settled · Updated 4 days ago")).toBeInTheDocument();
    expect(screen.queryByText(/Data may be stale/i)).not.toBeInTheDocument();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  test("missing timestamp renders safe unknown state without stale warning", () => {
    render(<DataFreshnessNotice freshness={evaluateFreshness(null)} />);

    expect(screen.getByText("Update time unavailable")).toBeInTheDocument();
    expect(screen.queryByText(/Data may be stale/i)).not.toBeInTheDocument();
  });

  test("public app shell exposes compact navigation and no operator sidebar", () => {
    const { container } = render(
      <AppShell health={null}>
        <p>Dashboard body</p>
      </AppShell>
    );

    expect(screen.getByRole("navigation", { name: "Primary navigation" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Current Event" })).toHaveAttribute("href", "/");
    expect(container.querySelector("aside")).toBeNull();
  });

  test("configured stale threshold is respected", () => {
    process.env.NEXT_PUBLIC_APEX_PULSE_DASHBOARD_STALE_AFTER_MINUTES = "90";

    expect(getDashboardStaleAfterMinutes()).toBe(90);
    expect(
      evaluateFreshness("2026-07-06T10:20:00Z", {
        now: new Date("2026-07-06T12:00:00Z"),
        staleAfterMinutes: getDashboardStaleAfterMinutes()
      }).state
    ).toBe("stale");
  });

  test("malformed stale threshold falls back safely", () => {
    process.env.NEXT_PUBLIC_APEX_PULSE_DASHBOARD_STALE_AFTER_MINUTES = "not-a-number";

    expect(getDashboardStaleAfterMinutes()).toBe(DEFAULT_STALE_AFTER_MINUTES);
  });
});
