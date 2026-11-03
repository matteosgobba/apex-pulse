import { formatDecimal, formatPercent } from "@/lib/formatters";
import type { PublicEventViewModel } from "@/lib/public-view-model";

export function EventMetrics({
  metrics,
}: {
  metrics: PublicEventViewModel["metrics"];
}) {
  const cards = [
    {
      label: "Mean gap error",
      value: seconds(metrics.maeGapSec),
      hint: "Average absolute error in gap to pole",
    },
    {
      label: "Position error",
      value: formatDecimal(metrics.meanAbsolutePositionError, 2),
      hint: "Average positions missed per driver",
    },
    {
      label: "Top 5 agreement",
      value: formatPercent(metrics.top5Agreement),
      hint: "Predicted and official top-five overlap",
    },
    {
      label: "Top 10 agreement",
      value: formatPercent(metrics.top10Agreement),
      hint: "Predicted and official top-ten overlap",
    },
    {
      label: "Forecast coverage",
      value:
        metrics.coverage ??
        (metrics.coveragePercentage === null
          ? "Not available"
          : `${metrics.coveragePercentage.toFixed(1)}%`),
      hint: "Official entrants with a preserved forecast",
    },
  ].filter((card) => card.value !== "Not available");

  if (cards.length === 0) {
    return null;
  }
  return (
    <section aria-labelledby="metrics-title" className="integrated-metrics">
      <div className="section-heading">
        <div>
          <p className="eyebrow">
            <span className="section-number">02 /</span>THE CHEQUERED FLAG
          </p>
          <h2 id="metrics-title">
            The forecast meets
            <br />
            the finish line.
          </h2>
        </div>
        <p className="section-intro">
          A prediction is only half the story.
          <br />
          Here is how this forecast
          <br />
          compared with qualifying.
        </p>
      </div>
      <div className="integrated-metric-grid">
        {cards.map((card) => (
          <article key={card.label} className="metric" title={card.hint}>
            <p className="text-sm font-medium text-apex-muted">{card.label}</p>
            <p className="mt-3 text-2xl font-semibold text-apex-text">
              {card.value}
            </p>
            <p className="mt-2 text-xs leading-5 text-apex-muted">
              {card.hint}
            </p>
          </article>
        ))}
      </div>
    </section>
  );
}

function seconds(value: number | null): string {
  return typeof value === "number" && Number.isFinite(value)
    ? `${value.toFixed(3)}s`
    : "Not available";
}
