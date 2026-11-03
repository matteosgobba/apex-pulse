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
            <span className="muted-type">Driven by curiosity.</span>
          </h2>
        </div>
        <p className="section-intro">
          No crystal ball. No private team data.
          <br /> Just a transparent machine learning project
          <br /> exploring the pace before qualifying.
        </p>
      </div>
      <div className="method-steps">
        {STEPS.map((step, index) => (
          <article key={step.number}>
            <p className="step-number">
              {step.number}
              <span>{index === 2 ? "⚑" : "↗"}</span>
            </p>
            <svg
              className="method-icon"
              viewBox="0 0 80 40"
              fill="none"
              aria-hidden="true"
            >
              {index === 0 ? (
                <path d="M0 30h12l6-16 9 17 12-26 10 20 6-10 7 15h18" />
              ) : index === 1 ? (
                <>
                  <path d="M10 5v30m30-30v30M70 5v30M10 10l30 20 30-20M10 30 40 10l30 20" />
                  <circle cx="10" cy="10" r="4" />
                  <circle cx="40" cy="30" r="4" />
                  <circle cx="70" cy="10" r="4" />
                </>
              ) : (
                <>
                  <path d="M10 35H72" />
                  <path d="M20 35V22H32V35M40 35V12H52V35M60 35V5H72V35" />
                </>
              )}
            </svg>
            <h3>{step.title}</h3>
            <p>{step.body}</p>
            <span className="step-tag">{step.tag}</span>
          </article>
        ))}
      </div>
      <Link href="/methodology" className="methodology-link text-link">
        Data, limitations & methodology <span>↗</span>
      </Link>
    </section>
  );
}
