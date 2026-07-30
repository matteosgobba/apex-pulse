export interface ContactLink {
  label: string;
  href: string;
  external: boolean;
}

const LINKEDIN_URL = process.env.NEXT_PUBLIC_APEX_PULSE_LINKEDIN_URL ?? null;

export const SITE_CONTACT = {
  name: "Matteo Sgobba",
  email: "matteosgobba03@gmail.com",
  github: "https://github.com/matteosgobba/apex-pulse",
  linkedin: validLinkedInUrl(LINKEDIN_URL) ? LINKEDIN_URL : null
} as const;

export function contactLinks(): ContactLink[] {
  const links: ContactLink[] = [
    { label: "GitHub", href: SITE_CONTACT.github, external: true },
    { label: "Email", href: `mailto:${SITE_CONTACT.email}`, external: false }
  ];
  if (SITE_CONTACT.linkedin) {
    links.splice(1, 0, {
      label: "LinkedIn",
      href: SITE_CONTACT.linkedin,
      external: true
    });
  }
  return links;
}

function validLinkedInUrl(value: string | null): value is string {
  if (!value) {
    return false;
  }
  try {
    const url = new URL(value);
    return url.protocol === "https:" && ["linkedin.com", "www.linkedin.com"].includes(url.hostname);
  } catch {
    return false;
  }
}
