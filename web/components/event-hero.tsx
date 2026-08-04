import { ApexPulseLogo } from "@/components/apex-pulse-logo";
import { DataFreshnessNotice } from "@/components/data-freshness-notice";
import type { PublicEventViewModel } from "@/lib/public-view-model";

export function EventHero({
  event,
  primary = true
}: {
  event: PublicEventViewModel;
  primary?: boolean;
}) {
  const place = [event.circuit, event.location].filter(Boolean).join(" · ");
  const EventHeading = primary ? "h1" : "h2";
  return (
    <section className="overflow-hidden rounded-[2rem] border border-white/10 bg-apex-ink text-apex-onStrong shadow-hero">
      <div className="grid gap-10 px-6 py-9 md:px-10 md:py-12 lg:grid-cols-[1fr_0.8fr] lg:items-end">
        <div>
          <ApexPulseLogo priority className="mb-10 max-h-24 max-w-[260px] sm:max-w-[320px]" />
          <p className="mb-4 text-xs font-semibold uppercase tracking-[0.16em] text-apex-accent">
            Latest Apex Pulse prediction / result
          </p>
          <div className="flex flex-wrap items-center gap-3">
            <span className="rounded-full bg-white/10 px-3 py-1 text-xs font-semibold text-apex-onStrong">
              {event.season ?? "Season unavailable"}
            </span>
            <span className="text-sm text-apex-onStrongMuted">{event.lifecycleLabel}</span>
          </div>
          <EventHeading className="mt-5 max-w-4xl text-4xl font-semibold tracking-tight sm:text-5xl lg:text-6xl">
            {event.eventName}
          </EventHeading>
          {place ? <p className="mt-3 text-base text-apex-onStrongMuted">{place}</p> : null}
        </div>
        <div className="rounded-2xl border border-apex-border bg-apex-panel p-5 text-apex-text shadow-card">
          <p className="text-sm font-semibold text-apex-accent">{event.checkpointLabel}</p>
          <p className="mt-2 text-sm leading-6 text-apex-secondary">
            Generated before qualifying from the practice data available at this checkpoint.
          </p>
          <p className="mt-4 text-sm leading-6 text-apex-secondary">{event.lifecycleDetail}</p>
          <div className="mt-5">
            <DataFreshnessNotice
              freshness={event.freshness}
              terminal={
                event.lifecycle === "settled" ||
                event.lifecycle === "settled_partial_coverage"
              }
            />
          </div>
        </div>
      </div>
    </section>
  );
}
