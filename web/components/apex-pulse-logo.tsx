import Image from "next/image";

export function ApexPulseLogo({
  compact = false,
  priority = false,
  className = ""
}: {
  compact?: boolean;
  priority?: boolean;
  className?: string;
}) {
  return (
    <Image
      src={
        compact
          ? "/brand/apex-pulse-logo-simple.png"
          : "/brand/apex-pulse-logo.png"
      }
      alt="Apex Pulse"
      width={compact ? 110 : 260}
      height={compact ? 48 : 106}
      priority={priority}
      className={`h-auto w-auto object-contain ${className}`}
    />
  );
}
