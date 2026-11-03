import Link from "next/link";
import { RacingWeekend } from "@/components/racing-weekend";

export function RacingHero() {
  return (
    <>
      <section className="hero" id="home" aria-labelledby="racing-title">
        <div className="hero-copy">
          <p className="eyebrow">
            <span className="pulse-dot" /> MACHINE LEARNING. RACING INSTINCT.
          </p>
          <h1 id="racing-title">
            READ THE PACE.
            <br />
            <span className="outline-type">ANTICIPATE</span>
            <br />
            THE GRID<span className="red-type">.</span>
          </h1>
          <p className="hero-description">
            The story of qualifying starts in practice.
            <br />
            We turn the laps into a forecast. You see the bigger picture.
          </p>
          <div className="hero-actions">
            <a className="button button-red" href="#forecast">
              Explore the forecast <span>↗</span>
            </a>
            <Link href="#methodology" className="text-link">
              How it works <span>↓</span>
            </Link>
          </div>
          <p className="hero-footnote">
            <span className="tiny-check">✓</span> Public data <span>/</span>{" "}
            Preserved predictions <span>/</span> Open methodology
          </p>
        </div>
        <RacingWeekend />
      </section>
      <div className="ticker" aria-hidden="true">
        <div className="ticker-track">
          {[0, 1, 2, 3].map((index) => (
            <span key={index}>
              DATA IN. LIGHTS OUT. <b>✳</b> FIND YOUR EDGE. <b>✳</b> EVERY
              MILLISECOND MATTERS. <b>✳</b>{" "}
            </span>
          ))}
        </div>
      </div>
    </>
  );
}
