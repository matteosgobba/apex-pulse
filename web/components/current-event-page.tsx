import { ContactSection } from "@/components/contact-section";
import { EventHero } from "@/components/event-hero";
import { EventMetrics } from "@/components/event-metrics";
import { ForecastRanking } from "@/components/forecast-ranking";
import { MethodologyPreview } from "@/components/methodology-preview";
import { OperationalWeekendCard } from "@/components/operational-weekend-card";
import { PredictionOfficialComparison } from "@/components/prediction-official-comparison";
import { TechnicalDetails } from "@/components/technical-details";
import { WeekendTimeline } from "@/components/weekend-timeline";
import type { CurrentEventPageData } from "@/lib/dashboard-types";
import { adaptCurrentEvent, adaptLatestCompletedForecast } from "@/lib/public-view-model";
import { availableOperationalEvent } from "@/lib/operational-event";

export function CurrentEventPageView({
  data,
  now
}: {
  data: CurrentEventPageData;
  now?: Date;
}) {
  const operationalEvent = availableOperationalEvent(data.operationalStatus);
  if (data.error && !operationalEvent) {
    return (
      <PublicEmptyState
        title="Prediction data is unavailable"
        detail="The exported dashboard artifacts could not be loaded. This public interface remains read-only and will not attempt to regenerate them."
      />
    );
  }
  const event = adaptCurrentEvent(data, { now });
  const historicalForecast = adaptLatestCompletedForecast(data.historicalMonitoring, {
    excludeEventSlug: operationalEvent?.event_slug,
    now
  });
  const displayedForecastEvent =
    event.available && event.hasForecast
      ? event
      : (historicalForecast ?? (event.available ? event : null));
  if (!displayedForecastEvent && !operationalEvent) {
    return (
      <PublicEmptyState
        title="No current prediction is available"
        detail="A valid current or recently settled prospective event is not present in the exported artifacts."
      />
    );
  }
  const operationalIsDifferentEvent =
    displayedForecastEvent !== null &&
    operationalEvent?.event_slug !== undefined &&
    operationalEvent.event_slug !== displayedForecastEvent.eventSlug;

  return (
    <div className="space-y-16">
      {operationalEvent ? (
        <OperationalWeekendCard
          event={operationalEvent}
          status={data.operationalStatus?.data}
          initialNow={now?.toISOString()}
        />
      ) : null}
      {displayedForecastEvent ? (
        <>
          <EventHero
            event={displayedForecastEvent}
            primary={!operationalEvent}
            contextLabel={
              operationalIsDifferentEvent
                ? "Last completed prediction"
                : "Latest Apex Pulse prediction / result"
            }
          />
          <div>
            <WeekendTimeline
              schedule={displayedForecastEvent.schedule}
              sessions={displayedForecastEvent.sessions}
              lifecycle={displayedForecastEvent.lifecycle}
              now={now}
            />
          </div>
          <ForecastRanking rows={displayedForecastEvent.ranking} />
          {displayedForecastEvent.hasSettlement ? (
            <>
              <PredictionOfficialComparison
                rows={displayedForecastEvent.comparison}
                coverage={displayedForecastEvent.metrics.coverage}
                unforecastedEntrants={displayedForecastEvent.unforecastedEntrants}
              />
              <EventMetrics metrics={displayedForecastEvent.metrics} />
            </>
          ) : null}
          <TechnicalDetails items={displayedForecastEvent.technical} />
        </>
      ) : (
        <PublicEmptyState
          title="No completed prediction is available"
          detail="This weekend is being reported from the operational monitor, but no immutable pre-qualifying forecast is available to display."
        />
      )}
      <MethodologyPreview />
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
