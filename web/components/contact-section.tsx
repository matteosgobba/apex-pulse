import { contactLinks, SITE_CONTACT } from "@/lib/site-config";
export function ContactSection() {
  return (
    <section id="about" aria-labelledby="about-title" className="about-section">
      <div className="about-mark" aria-hidden="true">
        AP<span>↗</span>
      </div>
      <div>
        <p className="eyebrow">INDEPENDENT PROJECT. REAL CURIOSITY.</p>
        <h2 id="about-title">Built by {SITE_CONTACT.name}</h2>
        <p>
          Apex Pulse turns public free-practice data into pre-qualifying
          forecasts of the Formula 1 qualifying order, with each prediction
          frozen before qualifying begins.
        </p>
        <div className="contact-links">
          {contactLinks().map((link) => (
            <a
              key={link.label}
              href={link.href}
              target={link.external ? "_blank" : undefined}
              rel={link.external ? "noopener noreferrer" : undefined}
            >
              {link.label} <span aria-hidden="true">↗</span>
            </a>
          ))}
        </div>
      </div>
    </section>
  );
}
