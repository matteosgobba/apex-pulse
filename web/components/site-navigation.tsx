"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAVIGATION = [
  { href: "/#forecast", label: "The forecast" },
  { href: "/history", label: "Track record" },
  { href: "/methodology", label: "Under the hood" },
];

export function SiteNavigation() {
  const pathname = usePathname();
  return (
    <nav aria-label="Primary navigation">
      {NAVIGATION.map((item) => {
        const active = pathname === item.href.split("#")[0];
        return (
          <Link
            key={item.href}
            href={item.href}
            className={`nav-link ${active ? "active" : ""}`}
            aria-current={active ? "page" : undefined}
          >
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}
