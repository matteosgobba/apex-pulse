import { ApexPulseLogo } from "@/components/apex-pulse-logo";
import { DataFreshnessNotice } from "@/components/data-freshness-notice";
import type { PublicEventViewModel } from "@/lib/public-view-model";

export function EventHero({ event }: { event: PublicEventViewModel }) {
  const place = [event.circuit, event.location].filter(Boolean).join(" · ");
  return (
    <section className="overflow-hidden rounded-[2rem] bg-apex-ink text-white shadow-hero">
      <div className="grid gap-10 px-6 py-9 md:px-10 md:py-12 lg:grid-cols-[1fr_0.8fr] lg:items-end">
        <div>
          <ApexPulseLogo priority className="mb-10 max-h-24 max-w-[260px] sm:max-w-[320px]" />
          <div className="flex flex-wrap items-center gap-3">
            <span className="rounded-full bg-white/10 px-3 py-1 text-xs font-semibold text-white">
              {event.season ?? "Season unavailable"}
            </span>
            <span className="text-sm text-slate-300">{event.lifecycleLabel}</span>
          </div>
          <h1 className="mt-5 max-w-4xl text-4xl font-semibold tracking-tight sm:text-5xl lg:text-6xl">
            {event.eventName}
          </h1>
          {place ? <p className="mt-3 text-base text-slate-300">{place}</p> : null}
        </div>
        <div className="rounded-2xl bg-white p-5 text-apex-text shadow-sm">
          <p className="text-sm font-semibold text-apex-accent">{event.checkpointLabel}</p>
          <p className="mt-2 text-sm leading-6 text-slate-600">
            Generated before qualifying from the practice data available at this checkpoint.
          </p>
          <p className="mt-4 text-sm leading-6 text-slate-700">{event.lifecycleDetail}</p>
          <div className="mt-5">
            <DataFreshnessNotice freshness={event.freshness} />
          </div>
        </div>
      </div>
    </section>
  );
}
