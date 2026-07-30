export const THEMES = ["dark", "light"] as const;

export type Theme = (typeof THEMES)[number];

export const DEFAULT_THEME: Theme = "dark";
export const THEME_STORAGE_KEY = "apex-pulse-theme";
export const THEME_CHANGE_EVENT = "apex-pulse-theme-change";

export const THEME_COLORS: Record<Theme, string> = {
  dark: "#070B12",
  light: "#F7F8FA"
};

export function isTheme(value: unknown): value is Theme {
  return value === "dark" || value === "light";
}

export function resolveTheme(value: unknown): Theme {
  return isTheme(value) ? value : DEFAULT_THEME;
}

export const THEME_INITIALIZATION_SCRIPT = `
(() => {
  const key = ${JSON.stringify(THEME_STORAGE_KEY)};
  const fallback = ${JSON.stringify(DEFAULT_THEME)};
  let theme = fallback;
  try {
    const stored = window.localStorage.getItem(key);
    if (stored === "dark" || stored === "light") theme = stored;
  } catch {}
  const root = document.documentElement;
  root.dataset.theme = theme;
  root.style.colorScheme = theme;
  const themeColor = theme === "dark" ? ${JSON.stringify(THEME_COLORS.dark)} : ${JSON.stringify(THEME_COLORS.light)};
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.setAttribute("content", themeColor);
})();
`;
