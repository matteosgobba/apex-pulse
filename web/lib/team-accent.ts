import { getTeamIdentity } from "@/lib/team-identity";

export function teamAccent(teamKey: string | null | undefined): string {
  return getTeamIdentity(teamKey).primary;
}
