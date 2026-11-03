import Link from "next/link";

const STEPS = [
  {
    number: "01",
    title: "Listen to the laps.",
    body: "Practice timing, representative laps, relative pace and session context become interpretable features.",
    tag: "PRACTICE → FEATURES",
  },
  {
    number: "02",
    title: "Find the signal.",
    body: "A tabular model estimates qualifying gaps. Historical features use only information available before the weekend.",
    tag: "FEATURES → FORECAST",
  },
  {
    number: "03",
    title: "Face the stopwatch.",
    body: "The preserved forecast is compared with the qualifying result. Errors, missing drivers and limitations stay visible.",
    tag: "FORECAST → EVIDENCE",
  },
];
export function MethodologyPreview() {
  return (
    <section id="methodology" aria-labelledby="methodology-preview-title">
      <div className="method-header">
        <div>
          <p className="eyebrow">
            <span className="section-number">03 /</span> UNDER THE HOOD
          </p>
          <h2 id="methodology-preview-title">
            Built on data.
            <br />
            Driven by curiosity.
          </h2>
        </div>
        <Link href="/methodology" className="text-link">
          Data, limitations & methodology <span>↗</span>
        </Link>
      </div>
      <div className="method-steps">
        {STEPS.map((step, index) => (
          <article key={step.number}>
            <p className="step-number">
              {step.number}
              <span>↗</span>
            </p>
            <svg
              className="method-icon"
              viewBox="0 0 80 40"
              fill="none"
              aria-hidden="true"
            >
              {index === 0 ? (
                <>
                  <path d="M0 32h80M4 29l10-7 10 3 10-17 10 14 10-8 10 7 12-18" />
                  <circle cx="34" cy="8" r="3" />
                </>
              ) : index === 1 ? (
                <>
                  <path d="M4 7h24v26H4zM52 7h24v26H52zM28 20h24m-7-6 7 6-7 6" />
                  <path d="m10 23 5-6 6 3m37 5 6-11 6 6" />
                </>
              ) : (
                <path d="M2 34h70M10 34V23h12v11M35 34V12h12v22M60 34V5h12v29" />
              )}
            </svg>
            <h3>{step.title}</h3>
            <p>{step.body}</p>
            <span className="step-tag">{step.tag}</span>
          </article>
        ))}
      </div>
      <p className="metrics-note">
        No crystal ball. No private team data. Just a transparent machine
        learning project exploring the pace before qualifying.
      </p>
    </section>
  );
}
