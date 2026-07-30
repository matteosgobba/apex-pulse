import { ContactSection } from "@/components/contact-section";
import { EventHero } from "@/components/event-hero";
import { EventMetrics } from "@/components/event-metrics";
import { ForecastRanking } from "@/components/forecast-ranking";
import { MethodologyPreview } from "@/components/methodology-preview";
import { PredictionOfficialComparison } from "@/components/prediction-official-comparison";
import { SessionCountdown } from "@/components/session-countdown";
import { TechnicalDetails } from "@/components/technical-details";
import { WeekendTimeline } from "@/components/weekend-timeline";
import type { CurrentEventPageData } from "@/lib/dashboard-types";
import { adaptCurrentEvent } from "@/lib/public-view-model";

export function CurrentEventPageView({
  data,
  now
}: {
  data: CurrentEventPageData;
  now?: Date;
}) {
  if (data.error) {
    return (
      <PublicEmptyState
        title="Prediction data is unavailable"
        detail="The exported dashboard artifacts could not be loaded. This public interface remains read-only and will not attempt to regenerate them."
      />
    );
  }
  const event = adaptCurrentEvent(data, { now });
  if (!event.available) {
    return (
      <PublicEmptyState
        title="No current prediction is available"
        detail="A valid current or recently settled prospective event is not present in the exported artifacts."
      />
    );
  }

  return (
    <div className="space-y-16">
      <EventHero event={event} />
      <div className="grid gap-5 lg:grid-cols-[0.85fr_1.15fr]">
        <SessionCountdown
          schedule={event.schedule}
          lifecycle={event.lifecycle}
          initialNow={now?.toISOString()}
        />
        <WeekendTimeline
          schedule={event.schedule}
          sessions={event.sessions}
          lifecycle={event.lifecycle}
          now={now}
        />
      </div>
      <ForecastRanking rows={event.ranking} />
      {event.hasSettlement ? (
        <>
          <PredictionOfficialComparison
            rows={event.comparison}
            coverage={event.metrics.coverage}
            unforecastedEntrants={event.unforecastedEntrants}
          />
          <EventMetrics metrics={event.metrics} />
        </>
      ) : null}
      <MethodologyPreview />
      <TechnicalDetails items={event.technical} />
      <ContactSection />
    </div>
  );
}

function PublicEmptyState({ title, detail }: { title: string; detail: string }) {
  return (
    <section className="mx-auto max-w-2xl rounded-[2rem] border border-dashed border-apex-border bg-apex-panel p-10 text-center shadow-card">
      <p className="text-xs font-semibold uppercase tracking-[0.16em] text-apex-accent">
        Apex Pulse
      </p>
      <h1 className="mt-4 text-3xl font-semibold text-apex-text">{title}</h1>
      <p className="mt-3 text-sm leading-6 text-apex-secondary">{detail}</p>
    </section>
  );
}
