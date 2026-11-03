import Link from "next/link";
import type { PublicEventViewModel } from "@/lib/public-view-model";

export function EventMetrics({
  metrics,
}: {
  metrics: PublicEventViewModel["metrics"];
}) {
  if (
    [
      metrics.maeGapSec,
      metrics.meanAbsolutePositionError,
      metrics.top5Agreement,
      metrics.top10Agreement,
    ].every((value) => value === null)
  )
    return null;

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
            <span className="muted-type">the finish line.</span>
          </h2>
        </div>
        <p className="section-intro">
          A prediction is only half the story.
          <br /> Here is how this forecast
          <br /> compared with qualifying.
        </p>
      </div>
      <div className="metric-grid">
        <article className="metric">
          <div className="metric-label">
            MEAN GAP ERROR <span aria-hidden="true">↘</span>
          </div>
          <strong>
            {number(metrics.maeGapSec, 3)}
            {metrics.maeGapSec !== null ? <span>s</span> : null}
          </strong>
          <p>Average absolute gap error</p>
          <div className="mini-wave" aria-hidden="true">
            <svg viewBox="0 0 240 36">
              <path d="m0 25 15-4 15 6 15-14 15 7 15-6 15 11 15-16 15 6 15-12 15 10 15-6 15 8 15-13 15 4 15-8" />
            </svg>
          </div>
        </article>
        <article className="metric">
          <div className="metric-label">
            POSITION ERROR <span aria-hidden="true">↔</span>
          </div>
          <strong>{number(metrics.meanAbsolutePositionError, 2)}</strong>
          <p>Average positions missed per driver</p>
          <div className="metric-track" aria-hidden="true">
            {Array.from({ length: 18 }, (_, i) => (
              <i key={i} />
            ))}
          </div>
        </article>
        <AgreementMetric
          label="TOP 5 AGREEMENT"
          value={metrics.top5Agreement}
          count={5}
          hint="Top-five overlap, regardless of order"
        />
        <AgreementMetric
          label="TOP 10 AGREEMENT"
          value={metrics.top10Agreement}
          count={10}
          hint="Top-ten overlap, regardless of order"
        />
      </div>
      <p className="metrics-note">
        This weekend only ·{" "}
        {metrics.coverage ? `${metrics.coverage} entrants covered · ` : ""}Not
        an estimate of overall model performance. Decorative graphics are not
        historical trends.
      </p>
      <Link className="metrics-history-link text-link" href="/history">
        Explore the track record <span>↗</span>
      </Link>
    </section>
  );
}

function number(value: number | null, digits: number): string {
  return value !== null && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function AgreementMetric({
  label,
  value,
  count,
  hint,
}: {
  label: string;
  value: number | null;
  count: number;
  hint: string;
}) {
  const lit =
    value === null ? 0 : Math.round(Math.max(0, Math.min(1, value)) * count);
  return (
    <article className="metric">
      <div className="metric-label">
        {label} <span aria-hidden="true">◎</span>
      </div>
      <strong>
        {value === null ? "—" : (value * 100).toFixed(0)}
        {value !== null ? <span>%</span> : null}
      </strong>
      <p>{hint}</p>
      <div
        className={count === 5 ? "five-lights" : "ten-blocks"}
        aria-hidden="true"
      >
        {Array.from({ length: count }, (_, i) => (
          <i key={i} className={i < lit ? "is-lit" : ""} />
        ))}
      </div>
    </article>
  );
}
