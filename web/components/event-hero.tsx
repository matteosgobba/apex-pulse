import { DataFreshnessNotice } from "@/components/data-freshness-notice";
import { formatEventNameWithFlag } from "@/lib/event-display";
import type { PublicEventViewModel } from "@/lib/public-view-model";

export function EventHero({
  event,
  primary = true,
  contextLabel = "Latest Apex Pulse prediction / result",
}: {
  event: PublicEventViewModel;
  primary?: boolean;
  contextLabel?: string;
}) {
  const Heading = primary ? "h1" : "h2";
  return (
    <section className="forecast-event">
      <div>
        <p className="eyebrow red-type">{contextLabel}</p>
        <Heading>
          {formatEventNameWithFlag(event.eventName, event.schedule?.country)}
        </Heading>
        <p className="muted">
          {[event.season, event.circuit, event.location]
            .filter(Boolean)
            .join(" · ")}
        </p>
      </div>
      <div className="forecast-context">
        <span className="status-pill">{event.lifecycleLabel}</span>
        <strong>{event.checkpointLabel}</strong>
        <p>
          {event.hasForecast
            ? "Generated before qualifying from the practice data available at this checkpoint."
            : "No pre-qualifying forecast is available for this event."}
        </p>
        <p>{event.lifecycleDetail}</p>
        <DataFreshnessNotice
          freshness={event.freshness}
          terminal={
            event.lifecycle === "settled" ||
            event.lifecycle === "settled_partial_coverage"
          }
        />
      </div>
    </section>
  );
}
