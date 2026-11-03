import Link from "next/link";
import { contactLinks, SITE_CONTACT } from "@/lib/site-config";

export function SiteFooter() {
  return (
    <footer className="ap-main">
      <Link href="/" className="footer-brand">
        APEXPULSE
      </Link>
      <span>READ THE PACE. ANTICIPATE THE GRID.</span>
      <nav aria-label="Footer navigation">
        <Link href="/history">Prediction history</Link>
        <Link href="/methodology">Methodology</Link>
        {contactLinks().map((link) => (
          <a
            key={link.label}
            href={link.href}
            target={link.external ? "_blank" : undefined}
            rel={link.external ? "noopener noreferrer" : undefined}
          >
            {link.label}
          </a>
        ))}
      </nav>
      <p className="footer-disclaimer">
        Built by {SITE_CONTACT.name}. An independent machine learning project.
        Not affiliated with Formula 1, the FIA or any team. Circuit outlines:{" "}
        <a href="https://github.com/bacinger/f1-circuits">Tomislav Bacinger</a>,{" "}
        <a href="/CIRCUITS-LICENSE.md">MIT license</a>. Animation is
        illustrative.
      </p>
    </footer>
  );
}
