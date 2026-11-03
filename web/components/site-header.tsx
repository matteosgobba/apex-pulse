import Link from "next/link";
import { ApexPulseLogo } from "@/components/apex-pulse-logo";
import { ThemeToggle } from "@/components/theme-toggle";
import { SiteNavigation } from "@/components/site-navigation";
import { MotionToggle } from "@/components/racing-motion";
import { SITE_CONTACT } from "@/lib/site-config";
import type { HealthResponse } from "@/lib/dashboard-types";

export function SiteHeader({ health }: { health: HealthResponse | null }) {
  const available = Boolean(
    health &&
      !["unavailable", "invalid"].includes(health.dashboard_artifact_status),
  );
  return (
    <header className="header">
      <Link href="/" aria-label="Apex Pulse home" className="brand">
        <ApexPulseLogo compact priority className="brand-symbol" />
        <span>
          APEX<span className="brand-pulse">PULSE</span>
          <small>QUALIFYING INTELLIGENCE</small>
        </span>
      </Link>
      <SiteNavigation />
      <div className="header-actions">
        <a
          className="icon-button github-link"
          href={SITE_CONTACT.github}
          target="_blank"
          rel="noopener noreferrer"
          aria-label="Apex Pulse on GitHub"
          title="View the GitHub repository"
        >
          <svg
            width="17"
            height="17"
            viewBox="0 0 24 24"
            fill="currentColor"
            aria-hidden="true"
          >
            <path d="M12 .8a11.2 11.2 0 0 0-3.54 21.82c.56.1.77-.24.77-.54v-2.1c-3.13.68-3.79-1.33-3.79-1.33-.51-1.3-1.25-1.65-1.25-1.65-1.02-.7.08-.69.08-.69 1.13.08 1.72 1.16 1.72 1.16 1 1.72 2.63 1.22 3.27.93.1-.73.39-1.22.71-1.5-2.5-.29-5.13-1.25-5.13-5.54 0-1.23.44-2.23 1.16-3.02-.12-.29-.5-1.43.11-2.98 0 0 .95-.31 3.08 1.15a10.75 10.75 0 0 1 5.6 0c2.14-1.46 3.08-1.15 3.08-1.15.61 1.55.23 2.69.11 2.98.72.79 1.16 1.79 1.16 3.02 0 4.3-2.64 5.25-5.15 5.53.4.35.76 1.03.76 2.08v3.11c0 .3.2.65.77.54A11.2 11.2 0 0 0 12 .8Z" />
          </svg>
        </a>
        <MotionToggle />
        <ThemeToggle />
        <Link href="/#forecast" className="header-cta">
          Enter the pit wall <span>↗</span>
        </Link>
        <span className="sr-only">
          {available ? "Exported data available" : "Exported data unavailable"}
        </span>
      </div>
    </header>
  );
}
