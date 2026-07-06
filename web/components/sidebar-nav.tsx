"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_ITEMS = [
  { href: "/", label: "Current Event" },
  { href: "/forecast", label: "Forecast" },
  { href: "/practice", label: "Practice Status" },
  { href: "/methodology", label: "Methodology" }
];

export function SidebarNav() {
  const pathname = usePathname();
  return (
    <aside className="border-b border-apex-border bg-apex-panel/80 px-4 py-4 lg:min-h-dvh lg:w-72 lg:border-b-0 lg:border-r lg:px-5">
      <div className="flex items-center justify-between gap-4 lg:block">
        <Link href="/" className="block">
          <p className="text-lg font-bold tracking-wide text-apex-text">Apex Pulse</p>
          <p className="mt-1 text-xs font-semibold uppercase tracking-[0.18em] text-apex-muted">
            Qualifying Intelligence
          </p>
        </Link>
      </div>
      <nav className="mt-5 flex gap-2 overflow-x-auto lg:flex-col lg:overflow-visible">
        {NAV_ITEMS.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className={`whitespace-nowrap rounded-md border px-3 py-2 text-sm font-medium transition ${
              pathname === item.href
                ? "border-apex-accent/50 bg-apex-accent/10 text-apex-text"
                : "border-transparent text-apex-muted hover:border-apex-border hover:text-apex-text"
            }`}
          >
            {item.label}
          </Link>
        ))}
      </nav>
      <div className="mt-6 hidden rounded-lg border border-apex-border bg-apex-bg/50 p-4 text-sm leading-6 text-apex-muted lg:block">
        Read-only interface over exported monitoring artifacts. No ingestion, forecast, settlement,
        or training operation is triggered from the web app.
      </div>
    </aside>
  );
}
