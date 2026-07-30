"use client";

import { useTheme } from "@/components/theme-provider";

export function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();
  const nextTheme = theme === "dark" ? "light" : "dark";
  const label = `Switch to ${nextTheme} mode`;

  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={toggleTheme}
      className="inline-flex min-h-11 min-w-11 items-center justify-center gap-2 rounded-full border border-apex-border bg-apex-surface px-3 text-apex-secondary transition-colors hover:border-apex-muted hover:bg-apex-panel hover:text-apex-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-apex-accent focus-visible:ring-offset-2 focus-visible:ring-offset-apex-bg"
    >
      {theme === "dark" ? <SunIcon /> : <MoonIcon />}
      <span className="hidden text-xs font-semibold xl:inline">
        {theme === "dark" ? "Light" : "Dark"}
      </span>
    </button>
  );
}

function SunIcon() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      className="h-5 w-5"
    >
      <circle cx="12" cy="12" r="3.5" />
      <path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M18.7 5.3l-1.4 1.4M6.7 17.3l-1.4 1.4" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      className="h-5 w-5"
    >
      <path d="M20.4 15.2A8.5 8.5 0 0 1 8.8 3.6 8.5 8.5 0 1 0 20.4 15.2Z" />
    </svg>
  );
}
