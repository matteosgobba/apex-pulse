import Link from "next/link";

import { ApexPulseLogo } from "@/components/apex-pulse-logo";
import { ThemeToggle } from "@/components/theme-toggle";
import type { HealthResponse } from "@/lib/dashboard-types";

const NAVIGATION = [
  { href: "/", label: "Current Event" },
  { href: "/history", label: "History" },
  { href: "/methodology", label: "Methodology" },
  { href: "/#about", label: "About" }
];

export function SiteHeader({ health }: { health: HealthResponse | null }) {
  const available = Boolean(
    health && !["unavailable", "invalid"].includes(health.dashboard_artifact_status)
  );
  return (
    <header className="sticky top-0 z-50 border-b border-apex-border/80 bg-apex-bg/95 backdrop-blur-xl">
      <div className="mx-auto flex min-h-16 max-w-7xl flex-wrap items-center justify-between gap-x-4 gap-y-1 px-4 py-2 md:flex-nowrap md:py-0 lg:px-8">
        <Link
          href="/"
          aria-label="Apex Pulse home"
          className="flex items-center rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-apex-accent"
        >
          <span className="flex h-10 w-24 items-center justify-center overflow-hidden rounded-lg bg-apex-ink px-2 ring-1 ring-white/10 sm:w-28">
            <ApexPulseLogo compact priority className="max-h-8" />
          </span>
        </Link>
        <nav
          aria-label="Primary navigation"
          className="order-3 w-full min-w-0 md:order-none md:w-auto"
        >
          <ul className="flex items-center justify-between gap-1 text-xs font-medium text-apex-secondary sm:justify-start sm:text-sm md:gap-2">
            {NAVIGATION.map((item) => (
              <li key={item.href}>
                <Link
                  href={item.href}
                  className="inline-flex rounded-full px-2 py-2 transition-colors hover:bg-apex-surface hover:text-apex-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-apex-accent sm:px-3"
                >
                  {item.label}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
        <div className="flex shrink-0 items-center gap-2">
          <ThemeToggle />
          <div
            className="hidden items-center gap-2 text-xs text-apex-muted md:flex"
            aria-label={available ? "Exported data available" : "Exported data unavailable"}
          >
            <span
              className={`h-2 w-2 rounded-full ${
                available ? "bg-apex-success" : "bg-apex-warning"
              }`}
            />
            {available ? "Data available" : "Data unavailable"}
          </div>
        </div>
      </div>
    </header>
  );
}
