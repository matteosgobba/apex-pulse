import Image from "next/image";

export const HEADER_LOGO_PATHS = {
  light: "/brand/apex-pulse-logo-simple-black.png",
  dark: "/brand/apex-pulse-logo-simple-white.png"
} as const;

export function ApexPulseLogo({
  compact = false,
  priority = false,
  className = ""
}: {
  compact?: boolean;
  priority?: boolean;
  className?: string;
}) {
  if (compact) {
    return (
      <span
        className={`relative inline-block h-12 w-[110px] shrink-0 ${className}`}
        aria-label="Apex Pulse"
      >
        <Image
          src={HEADER_LOGO_PATHS.light}
          alt=""
          fill
          priority={priority}
          sizes="112px"
          data-logo-theme="light"
          className="apex-theme-logo apex-theme-logo-light object-contain"
        />
        <Image
          src={HEADER_LOGO_PATHS.dark}
          alt=""
          fill
          priority={priority}
          sizes="112px"
          data-logo-theme="dark"
          className="apex-theme-logo apex-theme-logo-dark object-contain"
        />
      </span>
    );
  }

  return (
    <Image
      src="/brand/apex-pulse-logo.png"
      alt="Apex Pulse"
      width={260}
      height={106}
      priority={priority}
      className={`h-auto w-auto object-contain ${className}`}
    />
  );
}
