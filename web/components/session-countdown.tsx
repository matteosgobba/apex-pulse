"use client";

import { useEffect, useMemo, useState } from "react";

import type { EventSchedule, LifecycleState } from "@/lib/dashboard-types";
import {
  countdownParts,
  selectNextSession,
  sessionDisplayName
} from "@/lib/session-schedule";

export function SessionCountdown({
  schedule,
  lifecycle,
  initialNow
}: {
  schedule: EventSchedule | null;
  lifecycle: LifecycleState;
  initialNow?: string;
}) {
  const [nowMs, setNowMs] = useState<number | null>(
    initialNow ? Date.parse(initialNow) : null
  );
  useEffect(() => {
    if (initialNow) {
      return;
    }
    const startup = window.setTimeout(() => setNowMs(Date.now()), 0);
    const timer = window.setInterval(() => setNowMs(Date.now()), 1_000);
    return () => {
      window.clearTimeout(startup);
      window.clearInterval(timer);
    };
  }, [initialNow]);

  const selection = useMemo(
    () => selectNextSession(schedule, lifecycle, new Date(nowMs ?? 0)),
    [schedule, lifecycle, nowMs]
  );
  if (nowMs === null) {
    return <CountdownShell title="Next session" detail="Calculating local session time…" />;
  }
  if (selection.state === "unavailable") {
    return (
      <CountdownShell
        title="Session schedule unavailable"
        detail="No verified session timestamp is present in this export."
      />
    );
  }
  if (selection.state === "complete") {
    return (
      <CountdownShell
        title="Qualifying complete"
        detail="The preserved prediction can now be compared with the official result."
        complete
      />
    );
  }

  const targetTimeMs = selection.targetTimeMs as number;
  const parts = countdownParts(targetTimeMs, nowMs);
  if (parts.complete) {
    return <CountdownShell title="Session starting" detail="Countdown reached zero." complete />;
  }
  const name = sessionDisplayName(selection.session?.session ?? "Session");
  const localDate = new Intl.DateTimeFormat(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZoneName: "short"
  }).format(targetTimeMs);
  return (
    <section
      aria-label={`Countdown to ${name}`}
      className="rounded-3xl border border-apex-border bg-white p-6 shadow-card"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-apex-accent">
            Next session
          </p>
          <h2 className="mt-2 text-2xl font-semibold text-apex-text">{name}</h2>
        </div>
        <p className="text-sm text-slate-500">{localDate} · your local time</p>
      </div>
      <div className="mt-6 grid grid-cols-4 gap-2 sm:gap-4">
        <TimePart value={parts.days} label="Days" />
        <TimePart value={parts.hours} label="Hours" />
        <TimePart value={parts.minutes} label="Minutes" />
        <TimePart value={parts.seconds} label="Seconds" />
      </div>
    </section>
  );
}

function TimePart({ value, label }: { value: number; label: string }) {
  return (
    <div className="rounded-2xl bg-apex-surface px-2 py-4 text-center">
      <div className="text-2xl font-semibold tracking-tight text-apex-text sm:text-4xl">
        {String(value).padStart(2, "0")}
      </div>
      <div className="mt-1 text-[10px] font-medium uppercase tracking-wide text-slate-500 sm:text-xs">
        {label}
      </div>
    </div>
  );
}

function CountdownShell({
  title,
  detail,
  complete = false
}: {
  title: string;
  detail: string;
  complete?: boolean;
}) {
  return (
    <section
      aria-label={title}
      className="rounded-3xl border border-apex-border bg-white p-6 shadow-card"
    >
      <div className="flex items-start gap-4">
        <span
          className={`mt-1 h-3 w-3 shrink-0 rounded-full ${
            complete ? "bg-emerald-500" : "bg-slate-300"
          }`}
        />
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            Weekend timing
          </p>
          <h2 className="mt-2 text-2xl font-semibold text-apex-text">{title}</h2>
          <p className="mt-2 text-sm leading-6 text-slate-600">{detail}</p>
        </div>
      </div>
    </section>
  );
}
