"use client";

import {
  Fragment,
  useEffect,
  useId,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import bootstrap from "@/lib/racing/racing-bootstrap.json";
import {
  CACHE_KEY,
  REFRESH_INTERVAL_MS,
  validateBundle,
} from "@/lib/racing/calendar-source.mjs";
import {
  countdownParts,
  countryFlag,
  projectCircuit,
  selectWeekend,
  validSessions,
} from "@/lib/racing/race-engine.mjs";
import type { CalendarResponse } from "@/lib/racing/types";

const initial = validateBundle(bootstrap);
const pad = (value: number) => String(value).padStart(2, "0");
const localTime = (value: string, options: Intl.DateTimeFormatOptions) =>
  new Intl.DateTimeFormat(undefined, options).format(new Date(value));

export function RacingWeekend() {
  const [bundle, setBundle] = useState(initial);
  const latestBundle = useRef(initial);
  const [now, setNow] = useState<number | null>(null);
  const [status, setStatus] = useState("Checking calendar…");
  const [warning, setWarning] = useState<string | null>(null);
  const [selected, setSelected] = useState("");
  const [demo, setDemo] = useState<number | null>(null);
  const refresh = useRef<() => void>(() => {});
  const pathId = useId().replace(/:/g, "");
  useEffect(() => {
    let active = true;
    let controller: AbortController | null = null;
    const startup = window.setTimeout(() => {
      try {
        const cached = window.localStorage.getItem(CACHE_KEY);
        if (cached) {
          const candidate = validateBundle(JSON.parse(cached));
          if (
            Date.parse(candidate.retrievedAtUtc) >
            Date.parse(initial.retrievedAtUtc)
          ) {
            latestBundle.current = candidate;
            setBundle(candidate);
          }
        }
      } catch {
        /* Storage is optional; the bundled verified calendar remains available. */
      }
      setNow(Date.now());
      void update();
    }, 0);
    const update = async () => {
      if (controller) return;
      controller = new AbortController();
      const timeout = window.setTimeout(() => controller?.abort(), 25_000);
      setStatus("Checking calendar…");
      try {
        const response = await fetch("/api/calendar", {
          cache: "no-store",
          signal: controller.signal,
        });
        if (!response.ok) throw new Error(`Calendar HTTP ${response.status}`);
        const result: CalendarResponse = await response.json();
        const verified = validateBundle(result.bundle);
        if (!active) return;
        // A browser may hold a newer verified calendar than a cold server's fallback.
        const retained =
          result.status === "cached" &&
          Date.parse(latestBundle.current.retrievedAtUtc) >
            Date.parse(verified.retrievedAtUtc)
            ? latestBundle.current
            : verified;
        latestBundle.current = retained;
        setBundle(retained);
        setStatus(
          result.status === "fresh" ? "Calendar verified" : "Saved calendar",
        );
        setWarning(
          result.warning ??
            (verified.geometryWarning
              ? "Circuit source unavailable; using saved outlines."
              : verified.nextSeasonWarning
                ? "Next season could not be verified; its saved dates may change."
                : null),
        );
        try {
          window.localStorage.setItem(CACHE_KEY, JSON.stringify(retained));
        } catch {
          /* Quota/privacy restrictions do not prevent online updates. */
        }
      } catch {
        if (active) {
          setStatus("Saved calendar");
          setWarning(
            "Calendar offline. Showing the last verified schedule; dates may have changed.",
          );
        }
      } finally {
        window.clearTimeout(timeout);
        controller = null;
      }
    };
    refresh.current = () => {
      void update();
    };
    const clock = window.setInterval(() => setNow(Date.now()), 1000);
    const polling = window.setInterval(() => {
      if (document.visibilityState === "visible") void update();
    }, REFRESH_INTERVAL_MS);
    const focus = () => {
      setNow(Date.now());
      void update();
    };
    const visibility = () => {
      if (document.visibilityState === "visible") focus();
    };
    window.addEventListener("focus", focus);
    window.addEventListener("online", focus);
    document.addEventListener("visibilitychange", visibility);
    return () => {
      window.clearTimeout(startup);
      active = false;
      controller?.abort();
      window.clearInterval(clock);
      window.clearInterval(polling);
      window.removeEventListener("focus", focus);
      window.removeEventListener("online", focus);
      document.removeEventListener("visibilitychange", visibility);
    };
  }, []);
  const auto = now === null ? null : selectWeekend(bundle.events, now);
  const pinned = bundle.events.find((event) => event.id === selected);
  const choice = pinned && now !== null ? selectWeekend([pinned], now) : auto;
  const selection = choice?.event ? choice : auto;
  const event = selection?.event;
  const session = selection?.session;
  const circuit = event?.circuitId
    ? bundle.circuits[event.circuitId]
    : undefined;
  const outline = circuit ? projectCircuit(circuit.points) : null;
  const demoAge = demo !== null && now !== null ? now - demo : null;
  const demoActive = demoAge !== null && demoAge < 8000;
  const lights = demoActive
    ? demoAge < 6000
      ? Math.min(5, 1 + Math.floor(demoAge / 1200))
      : 0
    : (selection?.lights ?? 0);
  const parts = countdownParts(selection?.remainingMs ?? 0);
  const upcoming =
    now === null
      ? []
      : bundle.events.filter((item) => selectWeekend([item], now).event);
  const caption =
    selection?.status === "scheduled_start"
      ? "Scheduled start · lights out"
      : (session?.name ??
        (selection?.status === "complete"
          ? "Awaiting the next published calendar"
          : "Awaiting confirmed schedule"));
  return (
    <div className="weekend-visual">
      <div
        className="track-stage"
        onPointerMove={(event) => {
          if (event.pointerType !== "mouse") return;
          const rect = event.currentTarget.getBoundingClientRect();
          event.currentTarget.style.setProperty(
            "--tilt-y",
            `${((event.clientX - rect.left) / rect.width - 0.5) * 7}deg`,
          );
          event.currentTarget.style.setProperty(
            "--tilt-x",
            `${-((event.clientY - rect.top) / rect.height - 0.5) * 7}deg`,
          );
        }}
        onPointerLeave={(event) => {
          event.currentTarget.style.setProperty("--tilt-x", "0deg");
          event.currentTarget.style.setProperty("--tilt-y", "0deg");
        }}
      >
        <div className="stage-grid" />
        <div className="stage-aura" />
        <div className="stage-top">
          <span>
            <i className="status-square" />
            NEXT ON THE CALENDAR
          </span>
          <span>
            ROUND {event ? pad(event.round) : "—"} / {event?.season ?? "—"}
          </span>
        </div>
        <div className="circuit-identity">
          <span id="circuit-flag" aria-label={event?.country}>
            {countryFlag(event?.countryCode ?? null)}
          </span>
          <div>
            <h2>{event?.name ?? "The next race awaits"}</h2>
            <p>
              {circuit?.name ??
                event?.location ??
                "Checking the public schedule"}
              {event ? ` · ${event.country}` : ""}
            </p>
          </div>
        </div>
        <span className="track-word" aria-hidden="true">
          {event?.countryCode ?? "F1"}
        </span>
        {outline ? (
          <svg
            className="track-svg"
            viewBox="0 0 660 500"
            aria-label={`${circuit?.name} circuit outline`}
            role="img"
          >
            <defs>
              <filter id={`${pathId}-glow`}>
                <feGaussianBlur stdDeviation="6" />
              </filter>
            </defs>
            <path
              d={outline.path}
              fill="none"
              stroke="var(--red)"
              strokeWidth="8"
              opacity=".16"
              filter={`url(#${pathId}-glow)`}
            />
            <path
              d={outline.path}
              fill="none"
              stroke="var(--line)"
              strokeWidth="7"
            />
            <path
              d={outline.path}
              fill="none"
              stroke="var(--red)"
              strokeWidth="2"
            />
            <path
              className="track-flow"
              d={outline.path}
              fill="none"
              stroke="var(--ink)"
              opacity=".5"
              strokeWidth="2"
              strokeDasharray="16 100"
            />
            <circle
              className="race-dot race-dot-one"
              r="4"
              fill="var(--ink)"
              style={{ offsetPath: `path('${outline.path}')` }}
            />
            <circle
              className="race-dot race-dot-two"
              r="3"
              fill="var(--red)"
              style={{ offsetPath: `path('${outline.path}')` }}
            />
          </svg>
        ) : (
          <p className="circuit-unavailable">
            {now === null
              ? "Loading the next circuit…"
              : "Circuit outline not available"}
          </p>
        )}
        <div className="stage-bottom">
          <span>REAL CIRCUIT GEOMETRY</span>
          <span>ILLUSTRATIVE LAP ANIMATION</span>
        </div>
      </div>
      <section
        className="weekend-console"
        aria-label="Next Formula 1 session countdown"
      >
        <div className="console-topline">
          <p className={`eyebrow ${demoActive ? "is-demo" : ""}`}>
            {demoActive ? "LIGHTS DEMO / 8 SECONDS" : "THE NEXT LIGHTS OUT"}
          </p>
          <div
            id="starting-lights"
            aria-label={`${lights} of 5 countdown lights illuminated`}
            role="img"
          >
            {[0, 1, 2, 3, 4].map((index) => (
              <i
                key={index}
                className={`start-light ${index < lights ? "lit" : ""}`}
              />
            ))}
          </div>
        </div>
        <div className="countdown-heading">
          <h3>{caption}</h3>
          {session && now !== null ? (
            <time dateTime={session.startUtc}>
              {localTime(session.startUtc, {
                weekday: "short",
                day: "2-digit",
                month: "short",
                hour: "2-digit",
                minute: "2-digit",
                timeZoneName: "short",
              })}
            </time>
          ) : null}
        </div>
        <div
          className="weekend-clock"
          role="timer"
          aria-label={
            session
              ? `Time until ${session.name}`
              : "Waiting for confirmed session time"
          }
        >
          {Object.entries(parts).map(([label, value], index) => (
            <Fragment key={label}>
              {index > 0 ? <b aria-hidden="true">:</b> : null}
              <div>
                <strong>{session ? pad(value) : "—"}</strong>
                <span>{label.toUpperCase()}</span>
              </div>
            </Fragment>
          ))}
        </div>
        <div className="clock-caption">
          <span>
            YOUR LOCAL TIME · SCHEDULED START, NOT LIVE SESSION STATUS
          </span>
          <button
            type="button"
            onClick={() => {
              if (now !== null) setDemo(now);
            }}
            disabled={demoActive || now === null}
          >
            Preview the lights ↗
          </button>
        </div>
        {event && now !== null && selection?.status !== "awaiting_schedule" ? (
          <div
            className="weekend-sessions"
            style={
              { "--session-count": event.sessions.length } as CSSProperties
            }
          >
            {validSessions(event).map((item) => (
              <div
                key={`${item.code}-${item.startUtc}`}
                className={`weekend-session ${session?.startUtc === item.startUtc ? "is-next" : item.startMs < now ? "start-passed" : ""}`}
              >
                <i />
                <strong>{item.code}</strong>
                <time dateTime={item.startUtc}>
                  {localTime(item.startUtc, {
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </time>
              </div>
            ))}
          </div>
        ) : null}
        <div className="calendar-controls">
          <label>
            <span className="sr-only">Choose a Grand Prix</span>
            <select
              value={pinned && choice?.event ? selected : ""}
              onChange={(event) => setSelected(event.target.value)}
            >
              <option value="">Auto · next Grand Prix</option>
              {upcoming.map((item) => (
                <option key={item.id} value={item.id}>
                  {countryFlag(item.countryCode)} {item.name} {item.season}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            aria-label="Refresh calendar"
            onClick={() => refresh.current()}
          >
            ↻
          </button>
          <a
            href="https://github.com/theOehrly/f1schedule"
            target="_blank"
            rel="noreferrer"
          >
            {status} ↗
          </a>
        </div>
        {warning ? (
          <p className="calendar-refresh-notice" role="status">
            {warning}
          </p>
        ) : null}
        <p className="lights-explanation">
          Lights: 7d → 24h → 6h → 1h → 5m. All out at the scheduled start.
          <br />
          Last verified:{" "}
          {now === null
            ? "—"
            : localTime(bundle.retrievedAtUtc, {
                day: "2-digit",
                month: "short",
                hour: "2-digit",
                minute: "2-digit",
              })}{" "}
          · refresh every 5 minutes.
        </p>
      </section>
    </div>
  );
}
