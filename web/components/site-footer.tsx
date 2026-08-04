import Link from "next/link";

import { ApexPulseLogo } from "@/components/apex-pulse-logo";
import { contactLinks, SITE_CONTACT } from "@/lib/site-config";

export function SiteFooter() {
  return (
    <footer className="mt-20 border-t border-white/10 bg-apex-ink text-apex-onStrong">
      <div className="mx-auto grid max-w-7xl gap-10 px-4 py-12 md:grid-cols-[1.4fr_0.8fr_0.8fr] lg:px-8">
        <div>
          <ApexPulseLogo className="max-h-20 max-w-[220px]" />
          <p className="mt-4 max-w-md text-sm leading-6 text-apex-onStrongMuted">
            Machine-learning qualifying predictions built from public Formula 1 practice data.
          </p>
          <p className="mt-3 text-sm text-apex-onStrongMuted/75">
            Created by {SITE_CONTACT.name}.
          </p>
        </div>
        <div>
          <h2 className="text-sm font-semibold text-apex-onStrong">Explore</h2>
          <nav
            aria-label="Footer navigation"
            className="mt-4 grid gap-3 text-sm text-apex-onStrongMuted"
          >
            <Link className="transition-colors hover:text-apex-onStrong" href="/">
              Current event
            </Link>
            <Link className="transition-colors hover:text-apex-onStrong" href="/history">
              Prediction history
            </Link>
            <Link className="transition-colors hover:text-apex-onStrong" href="/methodology">
              Methodology
            </Link>
          </nav>
        </div>
        <div>
          <h2 className="text-sm font-semibold text-apex-onStrong">Connect</h2>
          <div className="mt-4 grid gap-3 text-sm text-apex-onStrongMuted">
            {contactLinks().map((link) => (
              <a
                key={link.label}
                href={link.href}
                target={link.external ? "_blank" : undefined}
                rel={link.external ? "noopener noreferrer" : undefined}
                className="transition-colors hover:text-apex-onStrong"
              >
                {link.label}
              </a>
            ))}
          </div>
        </div>
      </div>
    </footer>
  );
}
