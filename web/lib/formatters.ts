export function formatDateTime(value: string | null | undefined): string {
  if (!value) {
    return "Not available";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return "Not available";
  }
  return new Intl.DateTimeFormat("en", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZoneName: "short"
  }).format(parsed);
}

export function formatInteger(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value)
    ? new Intl.NumberFormat("en").format(value)
    : "Not available";
}

export function formatSeconds(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value)
    ? `${value.toFixed(3)} sec`
    : "Not available";
}

export function formatSignedGap(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "Not available";
  }
  const normalized = Math.abs(value) < 0.0005 ? 0 : value;
  const sign = normalized >= 0 ? "+" : "-";
  return `${sign}${Math.abs(normalized).toFixed(3)}s`;
}

export function formatPercent(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value)
    ? `${(value * 100).toFixed(0)}%`
    : "Not available";
}

export function formatText(value: string | number | null | undefined): string {
  if (value === null || value === undefined || value === "") {
    return "Not available";
  }
  return String(value);
}

export function humanizeToken(value: string | null | undefined): string {
  if (!value) {
    return "Not available";
  }
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}
