import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  within,
} from "@testing-library/react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { RacingWeekend } from "@/components/racing-weekend";
import { ForecastRanking } from "@/components/forecast-ranking";
import { getTeamIdentity } from "@/lib/team-identity";
import type { PublicRankingRow } from "@/lib/public-view-model";
import { CACHE_KEY } from "@/lib/racing/calendar-source.mjs";
import bootstrap from "@/lib/racing/racing-bootstrap.json";

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date("2026-11-06T15:29:59Z"));
  const saved = new Map<string, string>();
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: {
      getItem: (key: string) => saved.get(key) ?? null,
      setItem: (key: string, value: string) => saved.set(key, value),
      clear: () => saved.clear(),
    },
  });
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});
const flush = async (ms: number) => {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
};
const online = () =>
  vi.stubGlobal(
    "fetch",
    vi
      .fn()
      .mockResolvedValue({
        ok: true,
        json: async () => ({ bundle: bootstrap, status: "fresh" }),
      }),
  );

test("native countdown reaches zero, advances to FP2, and selects a different real circuit", async () => {
  online();
  render(<RacingWeekend />);
  await flush(0);
  expect(
    screen.getByRole("heading", { name: "Practice 1" }),
  ).toBeInTheDocument();
  expect(
    screen.getByRole("img", { name: "5 of 5 countdown lights illuminated" }),
  ).toBeInTheDocument();
  await flush(1000);
  expect(
    screen.getByRole("heading", { name: /scheduled start/i }),
  ).toBeInTheDocument();
  expect(
    screen.getByRole("img", { name: "0 of 5 countdown lights illuminated" }),
  ).toBeInTheDocument();
  await flush(30_000);
  expect(
    screen.getByRole("heading", { name: "Practice 2" }),
  ).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Choose a Grand Prix"), {
    target: {
      value: bootstrap.events.find((event) => event.location === "Las Vegas")!
        .id,
    },
  });
  expect(
    screen.getByRole("heading", { name: "Las Vegas Grand Prix" }),
  ).toBeInTheDocument();
  expect(
    screen.getByRole("img", { name: /Las Vegas.*circuit outline/i }),
  ).toBeInTheDocument();
});

test("browser failure retains the verified calendar and reports degradation", async () => {
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("Offline")));
  render(<RacingWeekend />);
  await flush(0);
  expect(
    screen.getByRole("heading", { name: "São Paulo Grand Prix" }),
  ).toBeInTheDocument();
  expect(screen.getByRole("status")).toHaveTextContent(
    /offline.*last verified/i,
  );
});

test("the light demo returns to the actual countdown without changing its target", async () => {
  online();
  vi.setSystemTime(new Date("2026-11-02T20:00:00Z"));
  render(<RacingWeekend />);
  await flush(0);
  fireEvent.click(screen.getByRole("button", { name: /preview the lights/i }));
  expect(screen.getByText("LIGHTS DEMO / 8 SECONDS")).toBeInTheDocument();
  await flush(8000);
  expect(screen.queryByText("LIGHTS DEMO / 8 SECONDS")).not.toBeInTheDocument();
  expect(
    screen.getByRole("heading", { name: "Practice 1" }),
  ).toBeInTheDocument();
  expect(
    screen.getByRole("img", { name: "1 of 5 countdown lights illuminated" }),
  ).toBeInTheDocument();
});

test("forecast controls preserve original model gaps and the absence of official results", () => {
  const rows: PublicRankingRow[] = Array.from({ length: 9 }, (_, index) => ({
    driverCode: `D${index}`,
    driverName: null,
    team: getTeamIdentity(index === 0 ? "ferrari" : "mclaren"),
    predictedPosition: index + 1,
    predictedGapSec: 0.144 + index,
    intervalLowerSec: null,
    intervalUpperSec: null,
  }));
  render(<ForecastRanking rows={rows} />);
  const ranking = screen.getByRole("list", {
    name: "Predicted qualifying ranking",
  });
  expect(within(ranking).getAllByRole("listitem")).toHaveLength(7);
  expect(within(ranking).getByText("+0.144s")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Versus reality" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: /show full field/i }));
  expect(within(ranking).getAllByRole("listitem")).toHaveLength(9);
  fireEvent.change(screen.getByRole("searchbox"), { target: { value: "D8" } });
  expect(within(ranking).getAllByRole("listitem")).toHaveLength(1);
  fireEvent.click(screen.getByRole("button", { name: "Explore D8" }));
  expect(
    within(screen.getByLabelText("Selected driver details")).getByRole(
      "heading",
      { name: "D8" },
    ),
  ).toBeInTheDocument();
  fireEvent.change(screen.getByRole("searchbox"), {
    target: { value: "no match" },
  });
  expect(screen.getByRole("status")).toHaveTextContent("No drivers match");
});

test("a cold server fallback never overwrites a newer browser calendar", async () => {
  const newer = { ...bootstrap, retrievedAtUtc: "2026-11-06T10:00:00Z" };
  window.localStorage.setItem(CACHE_KEY, JSON.stringify(newer));
  vi.stubGlobal(
    "fetch",
    vi
      .fn()
      .mockResolvedValue({
        ok: true,
        json: async () => ({
          bundle: bootstrap,
          status: "cached",
          warning: "Last verified schedule",
        }),
      }),
  );
  render(<RacingWeekend />);
  await flush(0);
  expect(
    JSON.parse(window.localStorage.getItem(CACHE_KEY)!).retrievedAtUtc,
  ).toBe(newer.retrievedAtUtc);
});
