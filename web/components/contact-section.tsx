import { contactLinks, SITE_CONTACT } from "@/lib/site-config";
export function ContactSection() {
  return (
    <section id="about" aria-labelledby="about-title" className="about-section">
      <div className="about-mark" aria-hidden="true">
        AP<span>↗</span>
      </div>
      <div>
        <p className="eyebrow">
          AN INDEPENDENT PROJECT BY {SITE_CONTACT.name.toUpperCase()}
        </p>
        <h2 id="about-title">
          A little obsession.
          <br />A lot of race data.
        </h2>
        <p>
          Exploring what machine learning can read between the laps.
          <br />
          Built in the open, for the love of the sport.
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
      <a
        className="button button-outline"
        href={SITE_CONTACT.github}
        target="_blank"
        rel="noopener noreferrer"
      >
        Explore the project <span>↗</span>
      </a>
    </section>
  );
}
