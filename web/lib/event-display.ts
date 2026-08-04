const COUNTRY_FLAGS: Record<string, string> = {
  australia: "🇦🇺",
  austria: "🇦🇹",
  azerbaijan: "🇦🇿",
  bahrain: "🇧🇭",
  belgium: "🇧🇪",
  brazil: "🇧🇷",
  canada: "🇨🇦",
  china: "🇨🇳",
  hungary: "🇭🇺",
  italy: "🇮🇹",
  japan: "🇯🇵",
  mexico: "🇲🇽",
  monaco: "🇲🇨",
  netherlands: "🇳🇱",
  qatar: "🇶🇦",
  saudi_arabia: "🇸🇦",
  singapore: "🇸🇬",
  spain: "🇪🇸",
  united_arab_emirates: "🇦🇪",
  united_kingdom: "🇬🇧",
  united_states: "🇺🇸"
};

const COUNTRY_ALIASES: Record<string, keyof typeof COUNTRY_FLAGS> = {
  brazil: "brazil",
  great_britain: "united_kingdom",
  mexico: "mexico",
  the_netherlands: "netherlands",
  uae: "united_arab_emirates",
  uk: "united_kingdom",
  united_states_of_america: "united_states",
  usa: "united_states"
};

const EVENT_FLAGS: Record<string, string> = {
  abu_dhabi: "🇦🇪",
  abu_dhabi_grand_prix: "🇦🇪",
  australia: "🇦🇺",
  australian_grand_prix: "🇦🇺",
  austria: "🇦🇹",
  austrian_grand_prix: "🇦🇹",
  azerbaijan: "🇦🇿",
  azerbaijan_grand_prix: "🇦🇿",
  bahrain: "🇧🇭",
  bahrain_grand_prix: "🇧🇭",
  belgium: "🇧🇪",
  belgian_grand_prix: "🇧🇪",
  brazil: "🇧🇷",
  brazilian_grand_prix: "🇧🇷",
  british_grand_prix: "🇬🇧",
  canada: "🇨🇦",
  canadian_grand_prix: "🇨🇦",
  china: "🇨🇳",
  chinese_grand_prix: "🇨🇳",
  dutch_grand_prix: "🇳🇱",
  emilia_romagna: "🇮🇹",
  emilia_romagna_grand_prix: "🇮🇹",
  great_britain: "🇬🇧",
  great_britain_grand_prix: "🇬🇧",
  hungary: "🇭🇺",
  hungarian_grand_prix: "🇭🇺",
  italian_grand_prix: "🇮🇹",
  italy: "🇮🇹",
  japan: "🇯🇵",
  japanese_grand_prix: "🇯🇵",
  las_vegas: "🇺🇸",
  las_vegas_grand_prix: "🇺🇸",
  mexico: "🇲🇽",
  mexico_city: "🇲🇽",
  mexico_city_grand_prix: "🇲🇽",
  mexican_grand_prix: "🇲🇽",
  miami: "🇺🇸",
  miami_grand_prix: "🇺🇸",
  monaco: "🇲🇨",
  monaco_grand_prix: "🇲🇨",
  netherlands: "🇳🇱",
  qatar: "🇶🇦",
  qatar_grand_prix: "🇶🇦",
  sao_paulo: "🇧🇷",
  sao_paulo_grand_prix: "🇧🇷",
  saudi_arabia: "🇸🇦",
  saudi_arabian_grand_prix: "🇸🇦",
  singapore: "🇸🇬",
  singapore_grand_prix: "🇸🇬",
  spain: "🇪🇸",
  spain_grand_prix: "🇪🇸",
  spanish_grand_prix: "🇪🇸",
  united_states: "🇺🇸",
  united_states_grand_prix: "🇺🇸"
};

export function formatEventNameWithFlag(
  eventName: string | null | undefined,
  country?: string | null
): string {
  const name = eventName?.trim() ?? "";
  if (!name || containsFlagEmoji(name)) {
    return name;
  }

  const normalizedCountry = normalizeDisplayKey(country ?? "");
  const countryKey = COUNTRY_ALIASES[normalizedCountry] ?? normalizedCountry;
  const flag = COUNTRY_FLAGS[countryKey] ?? EVENT_FLAGS[normalizeDisplayKey(name)];
  return flag ? `${name}\u00A0${flag}` : name;
}

function containsFlagEmoji(value: string): boolean {
  return /[\u{1F1E6}-\u{1F1FF}]{2}/u.test(value);
}

function normalizeDisplayKey(value: string): string {
  return value
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .trim()
    .toLowerCase()
    .replace(/&/g, "and")
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}
