import Link from "next/link";

import { ApexPulseLogo } from "@/components/apex-pulse-logo";
import { contactLinks, SITE_CONTACT } from "@/lib/site-config";

export function SiteFooter() {
  return (
    <footer className="mt-20 bg-apex-ink text-white">
      <div className="mx-auto grid max-w-7xl gap-10 px-4 py-12 md:grid-cols-[1.4fr_0.8fr_0.8fr] lg:px-8">
        <div>
          <ApexPulseLogo className="max-h-20 max-w-[220px]" />
          <p className="mt-4 max-w-md text-sm leading-6 text-slate-300">
            Machine-learning qualifying predictions built from public Formula 1 practice data.
          </p>
          <p className="mt-3 text-sm text-slate-400">Created by {SITE_CONTACT.name}.</p>
        </div>
        <div>
          <h2 className="text-sm font-semibold text-white">Explore</h2>
          <nav aria-label="Footer navigation" className="mt-4 grid gap-3 text-sm text-slate-300">
            <Link className="hover:text-white" href="/">
              Current event
            </Link>
            <Link className="hover:text-white" href="/history">
              Prediction history
            </Link>
            <Link className="hover:text-white" href="/methodology">
              Methodology
            </Link>
          </nav>
        </div>
        <div>
          <h2 className="text-sm font-semibold text-white">Connect</h2>
          <div className="mt-4 grid gap-3 text-sm text-slate-300">
            {contactLinks().map((link) => (
              <a
                key={link.label}
                href={link.href}
                target={link.external ? "_blank" : undefined}
                rel={link.external ? "noreferrer" : undefined}
                className="hover:text-white"
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
