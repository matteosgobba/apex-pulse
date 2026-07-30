import { ApexPulseLogo } from "@/components/apex-pulse-logo";
import { contactLinks, SITE_CONTACT } from "@/lib/site-config";

export function ContactSection() {
  return (
    <section id="about" aria-labelledby="about-title" className="scroll-mt-24">
      <div className="grid overflow-hidden rounded-[2rem] bg-apex-ink text-white lg:grid-cols-[0.8fr_1.2fr]">
        <div className="flex min-h-64 items-center justify-center bg-black/20 p-8">
          <ApexPulseLogo className="max-h-28 max-w-[300px]" />
        </div>
        <div className="p-7 sm:p-10">
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-red-300">
            About the project
          </p>
          <h2 id="about-title" className="mt-3 text-3xl font-semibold">
            Built by {SITE_CONTACT.name}
          </h2>
          <p className="mt-4 max-w-2xl text-sm leading-7 text-slate-300">
            Apex Pulse is an applied machine-learning portfolio project focused on one clear
            question: what can public free-practice data tell us about the qualifying order?
          </p>
          <div className="mt-7 flex flex-wrap gap-3">
            {contactLinks().map((link) => (
              <a
                key={link.label}
                href={link.href}
                target={link.external ? "_blank" : undefined}
                rel={link.external ? "noreferrer" : undefined}
                className="rounded-full border border-white/20 px-4 py-2 text-sm font-semibold transition hover:border-white/50 hover:bg-white/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-300"
              >
                {link.label}
              </a>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
