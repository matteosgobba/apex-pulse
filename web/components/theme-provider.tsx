"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useSyncExternalStore,
  type ReactNode
} from "react";

import {
  DEFAULT_THEME,
  resolveTheme,
  THEME_CHANGE_EVENT,
  THEME_COLORS,
  THEME_STORAGE_KEY,
  type Theme
} from "@/lib/theme";

interface ThemeContextValue {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

export function ThemeProvider({ children }: { children: ReactNode }) {
  const theme = useSyncExternalStore(subscribeToTheme, readDocumentTheme, readServerTheme);

  const setTheme = useCallback((nextTheme: Theme) => {
    applyTheme(nextTheme, true);
  }, []);

  const toggleTheme = useCallback(() => {
    setTheme(theme === "dark" ? "light" : "dark");
  }, [setTheme, theme]);

  const value = useMemo(
    () => ({ theme, setTheme, toggleTheme }),
    [setTheme, theme, toggleTheme]
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeContextValue {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error("useTheme must be used within ThemeProvider");
  }
  return context;
}

function readServerTheme(): Theme {
  return DEFAULT_THEME;
}

function readDocumentTheme(): Theme {
  if (typeof document === "undefined") {
    return DEFAULT_THEME;
  }
  return resolveTheme(document.documentElement.dataset.theme);
}

function subscribeToTheme(onStoreChange: () => void): () => void {
  if (typeof window === "undefined") {
    return () => undefined;
  }

  const onThemeChange = () => onStoreChange();
  const onStorage = (event: StorageEvent) => {
    if (event.key !== THEME_STORAGE_KEY) {
      return;
    }
    applyTheme(resolveTheme(event.newValue), false);
    onStoreChange();
  };

  window.addEventListener(THEME_CHANGE_EVENT, onThemeChange);
  window.addEventListener("storage", onStorage);
  return () => {
    window.removeEventListener(THEME_CHANGE_EVENT, onThemeChange);
    window.removeEventListener("storage", onStorage);
  };
}

function applyTheme(theme: Theme, persist: boolean): void {
  if (typeof document === "undefined") {
    return;
  }

  document.documentElement.dataset.theme = theme;
  document.documentElement.style.colorScheme = theme;
  document
    .querySelector('meta[name="theme-color"]')
    ?.setAttribute("content", THEME_COLORS[theme]);

  if (persist) {
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      // The visible theme still changes when storage is unavailable.
    }
  }

  window.dispatchEvent(new Event(THEME_CHANGE_EVENT));
}
