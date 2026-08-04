export interface TeamIdentity {
  key: string;
  displayName: string;
  primary: string;
  secondary: string;
  foreground: "#FFFFFF" | "#0F172A";
  monogram: string;
  logoPath: string | null;
  known: boolean;
}

type TeamDefinition = Omit<TeamIdentity, "key" | "known" | "logoPath"> & {
  aliases: string[];
  logoPath?: string | null;
};

const TEAMS: Record<string, TeamDefinition> = {
  ferrari: team("Ferrari", "#E80020", "#FFF0F2", "#FFFFFF", "FER", ["scuderia ferrari"], "/teams/ferrari.svg.webp"),
  mclaren: team("McLaren", "#FF8000", "#FFF3E7", "#0F172A", "MCL", ["mclaren mercedes"], "/teams/mclaren.svg"),
  mercedes: team("Mercedes", "#00A19C", "#E7F7F6", "#0F172A", "MER", [
    "mercedes amg",
    "mercedes-amg"
  ], "/teams/mercedes.svg"),
  red_bull: team("Red Bull Racing", "#3671C6", "#EAF1FC", "#FFFFFF", "RBR", [
    "red bull",
    "red bull racing",
    "oracle red bull racing"
  ], "/teams/redbull.png"),
  audi: team("Audi", "#C0002A", "#FBE8EC", "#FFFFFF", "AUD", ["stake", "sauber"], "/teams/audi.webp"),
  racing_bulls: team("Racing Bulls", "#5E8BFF", "#EDF2FF", "#0F172A", "RB", [
    "rb",
    "visa cash app rb",
    "vcarb"
  ], "/teams/racingbulls.png"),
  alpine: team("Alpine", "#2173D3", "#EAF2FC", "#FFFFFF", "ALP", ["alpine f1 team"], "/teams/alpine.png"),
  haas: team("Haas", "#767A81", "#F0F1F2", "#FFFFFF", "HAS", ["haas f1 team"], "/teams/haas.png"),
  aston_martin: team("Aston Martin", "#006F62", "#E4F1EF", "#FFFFFF", "AMR", [
    "aston martin aramco"
  ], "/teams/astonmartin.png"),
  cadillac: team("Cadillac", "#161A22", "#EBEDF0", "#FFFFFF", "CAD", [
    "cadillac f1 team"
  ], "/teams/cadillac.png"),
  williams: team("Williams", "#005AFF", "#E8F0FF", "#FFFFFF", "WIL", ["williams racing"], "/teams/williams.svg")
};

const ALIASES = Object.entries(TEAMS).reduce<Record<string, string>>((index, [key, value]) => {
  index[normalizeTeamKey(key)] = key;
  index[normalizeTeamKey(value.displayName)] = key;
  value.aliases.forEach((alias) => {
    index[normalizeTeamKey(alias)] = key;
  });
  return index;
}, {});

const FALLBACK_COLORS = ["#334155", "#475569", "#3F4C5F", "#465569", "#26364C"];

export function getTeamIdentity(
  teamKey: string | null | undefined,
  teamName?: string | null
): TeamIdentity {
  const normalized = normalizeTeamKey(teamKey ?? teamName ?? "");
  const canonicalKey = ALIASES[normalized] ?? ALIASES[normalizeTeamKey(teamName ?? "")];
  if (canonicalKey) {
    const definition = TEAMS[canonicalKey];
    return {
      key: canonicalKey,
      displayName: definition.displayName,
      primary: definition.primary,
      secondary: definition.secondary,
      foreground: definition.foreground,
      monogram: definition.monogram,
      logoPath: definition.logoPath ?? null,
      known: true
    };
  }

  const displayName = teamName?.trim() || teamKey?.trim() || "Unknown team";
  const primary = FALLBACK_COLORS[stableIndex(normalized || displayName, FALLBACK_COLORS.length)];
  return {
    key: normalized || "unknown",
    displayName,
    primary,
    secondary: "#F1F5F9",
    foreground: "#FFFFFF",
    monogram: fallbackMonogram(displayName),
    logoPath: null,
    known: false
  };
}

export function normalizeTeamKey(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/&/g, "and")
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function team(
  displayName: string,
  primary: string,
  secondary: string,
  foreground: TeamIdentity["foreground"],
  monogram: string,
  aliases: string[],
  logoPath: string
): TeamDefinition {
  return { displayName, primary, secondary, foreground, monogram, aliases, logoPath };
}

function stableIndex(value: string, length: number): number {
  let hash = 0;
  for (const character of value) {
    hash = (hash * 31 + character.charCodeAt(0)) >>> 0;
  }
  return hash % length;
}

function fallbackMonogram(value: string): string {
  const words = value.match(/[A-Za-z0-9]+/g) ?? [];
  if (words.length > 1) {
    return words
      .slice(0, 3)
      .map((word) => word[0])
      .join("")
      .toUpperCase();
  }
  return (words[0] ?? "TEAM").slice(0, 3).toUpperCase();
}
