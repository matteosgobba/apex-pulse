import Link from "next/link";
import { RacingWeekend } from "@/components/racing-weekend";

export function RacingHero() {
  return (
    <>
      <section className="hero" id="home" aria-labelledby="racing-title">
        <div className="hero-copy">
          <p className="eyebrow">
            <span className="pulse-dot" /> A DIFFERENT KIND OF RACE INTELLIGENCE
          </p>
          <h1 id="racing-title">
            READ THE PACE.
            <br />
            <span className="outline-type">ANTICIPATE</span>
            <br />
            THE GRID<span className="red-type">.</span>
          </h1>
          <p className="hero-description">
            Free practice tells a story. We turn it into a forecast.
            <br />
            Machine learning. Public F1 data. Every prediction on record.
          </p>
          <div className="hero-actions">
            <a className="button button-red" href="#forecast">
              Explore the forecast <span>↗</span>
            </a>
            <Link href="/methodology" className="text-link">
              How it works <span>↗</span>
            </Link>
          </div>
          <p className="hero-footnote">
            <span className="tiny-check">✓</span> PUBLIC DATA <span>/</span>{" "}
            PRESERVED FORECASTS <span>/</span> NO HINDSIGHT
          </p>
        </div>
        <RacingWeekend />
      </section>
      <div className="ticker" aria-hidden="true">
        <div className="ticker-track">
          {[0, 1, 2, 3].map((index) => (
            <span key={index}>
              DATA IN. LIGHTS OUT. <b>✳</b> READ THE PACE. <b>✳</b> FIND YOUR
              EDGE. <b>✳</b> EVERY LAP COUNTS. <b>✳</b>{" "}
            </span>
          ))}
        </div>
      </div>
    </>
  );
}
