"use client";

import { useEffect, useState } from "react";

import { SessionCountdown } from "@/components/session-countdown";
import type { AutopilotStatusData, OperationalEvent } from "@/lib/dashboard-types";
import { formatEventNameWithFlag } from "@/lib/event-display";
import {
  operationalEventSchedule,
  operationalForecastNotice,
  operationalFormatLabel,
  operationalWeekendLabel
} from "@/lib/operational-event";

export function OperationalWeekendCard({
  event,
  status,
  initialNow
}: {
  event: OperationalEvent;
  status?: AutopilotStatusData | null;
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
  const forecastNotice = operationalForecastNotice(event, status);
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
              {formatEventNameWithFlag(event.event)}
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
            forecastNotice.tone === "warning"
              ? "border-apex-warning/35 bg-apex-warning/10 text-apex-secondary"
              : forecastNotice.tone === "success"
                ? "border-apex-success/35 bg-apex-success/10 text-apex-secondary"
                : "border-apex-border bg-apex-surface text-apex-secondary"
          }`}
        >
          <p className="font-semibold text-apex-text">{forecastNotice.label}</p>
          <p className="mt-1">{forecastNotice.detail}</p>
        </div>
      </div>
      <SessionCountdown
        schedule={schedule}
        lifecycle="practice_in_progress"
        initialNow={initialNow}
        forecastAvailable={event.supported ? (status?.forecast_exists ?? undefined) : false}
      />
    </section>
  );
}
