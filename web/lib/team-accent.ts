const TEAM_ACCENTS: Record<string, string> = {
  alpine: "#64c4ff",
  aston_martin: "#4fd1b0",
  audi: "#c7ccd4",
  cadillac: "#d6b66a",
  ferrari: "#ff6b6b",
  haas: "#d9dee7",
  mclaren: "#ffb45c",
  mercedes: "#7ee4d8",
  racing_bulls: "#8eb8ff",
  red_bull: "#8f9dff",
  williams: "#6da8ff"
};

const FALLBACK_ACCENT = "#9ca8b8";

export function teamAccent(teamKey: string | null | undefined): string {
  if (!teamKey) {
    return FALLBACK_ACCENT;
  }
  return TEAM_ACCENTS[teamKey] ?? FALLBACK_ACCENT;
}
