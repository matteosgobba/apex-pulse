import Image from "next/image";

import type { TeamIdentity } from "@/lib/team-identity";

export function TeamMark({
  team,
  size = "md"
}: {
  team: TeamIdentity;
  size?: "sm" | "md";
}) {
  const sizeClass = size === "sm" ? "h-8 w-8 text-[9px]" : "h-10 w-10 text-[10px]";
  return (
    <span
      className={`inline-flex shrink-0 items-center justify-center overflow-hidden rounded-xl border border-white/10 font-bold tracking-wide shadow-sm ring-1 ring-apex-border/40 ${sizeClass}`}
      style={{ backgroundColor: team.primary, color: team.foreground }}
      title={team.logoPath ? team.displayName : `${team.displayName} team mark`}
      aria-label={team.displayName}
    >
      {team.logoPath ? (
        <Image src={team.logoPath} alt="" width={40} height={40} className="object-contain" />
      ) : (
        team.monogram
      )}
    </span>
  );
}
