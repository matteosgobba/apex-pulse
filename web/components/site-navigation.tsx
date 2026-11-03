"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAVIGATION = [
  { href: "/", label: "Current Event" },
  { href: "/history", label: "History" },
  { href: "/methodology", label: "Methodology" },
  { href: "/#about", label: "About" },
];

export function SiteNavigation() {
  const pathname = usePathname();
  return (
    <nav aria-label="Primary navigation">
      {NAVIGATION.map((item) => {
        const active = pathname === item.href;
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
