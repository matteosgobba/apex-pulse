"use client";

import { useEffect, useState } from "react";

import { SessionCountdown } from "@/components/session-countdown";
import type { OperationalEvent } from "@/lib/dashboard-types";
import {
  operationalEventSchedule,
  operationalFormatLabel,
  operationalWeekendLabel
} from "@/lib/operational-event";

export function OperationalWeekendCard({
  event,
  initialNow
}: {
  event: OperationalEvent;
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
    const timer = window.setInterval(() => setNowMs(Date.now()), 60_000);
    return () => {
      window.clearTimeout(startup);
      window.clearInterval(timer);
    };
  }, [initialNow]);
  const schedule = operationalEventSchedule(event);
  const supported = event.supported;
  const weekendLabel =
    nowMs === null ? "Monitored weekend" : operationalWeekendLabel(event, new Date(nowMs));

  return (
    <section aria-label="Operational Formula 1 weekend" className="space-y-5">
      <div className="rounded-[2rem] border border-apex-border bg-apex-panel p-6 shadow-card sm:p-8">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-apex-accent">
              {weekendLabel}
            </p>
            <h1 className="mt-3 break-words text-3xl font-semibold tracking-tight text-apex-text sm:text-4xl">
              {event.event}
            </h1>
            <p className="mt-2 text-sm text-apex-secondary">
              {event.season} · Round {event.round_number}
            </p>
          </div>
          <span className="rounded-full border border-apex-border bg-apex-surface px-3 py-1.5 text-xs font-semibold text-apex-secondary">
            {operationalFormatLabel(event.event_format)}
          </span>
        </div>
        <div
          className={`mt-6 rounded-2xl border px-4 py-4 text-sm leading-6 ${
            supported
              ? "border-apex-border bg-apex-surface text-apex-secondary"
              : "border-apex-warning/35 bg-apex-warning/10 text-apex-secondary"
          }`}
        >
          {supported ? (
            <p>
              Apex Pulse is monitoring the conventional practice-to-qualifying schedule for the
              next guarded prediction window.
            </p>
          ) : (
            <p>
              Apex Pulse does not create predictions for Sprint weekends yet. This weekend remains
              visible and monitored, but no qualifying forecast will be generated.
            </p>
          )}
        </div>
      </div>
      <SessionCountdown
        schedule={schedule}
        lifecycle="practice_in_progress"
        initialNow={initialNow}
      />
    </section>
  );
}
