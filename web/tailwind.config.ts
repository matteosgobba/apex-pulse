import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        apex: {
          bg: "rgb(var(--apex-background) / <alpha-value>)",
          panel: "rgb(var(--apex-surface) / <alpha-value>)",
          elevated: "rgb(var(--apex-surface-elevated) / <alpha-value>)",
          panelSoft: "rgb(var(--apex-surface-muted) / <alpha-value>)",
          surface: "rgb(var(--apex-surface-muted) / <alpha-value>)",
          border: "rgb(var(--apex-border) / <alpha-value>)",
          muted: "rgb(var(--apex-muted) / <alpha-value>)",
          secondary: "rgb(var(--apex-secondary) / <alpha-value>)",
          text: "rgb(var(--apex-foreground) / <alpha-value>)",
          ink: "rgb(var(--apex-ink) / <alpha-value>)",
          accent: "rgb(var(--apex-brand) / <alpha-value>)",
          accentSoft: "rgb(var(--apex-brand-soft) / <alpha-value>)",
          success: "rgb(var(--apex-success) / <alpha-value>)",
          successSoft: "rgb(var(--apex-success-soft) / <alpha-value>)",
          successText: "rgb(var(--apex-success-text) / <alpha-value>)",
          warning: "rgb(var(--apex-warning) / <alpha-value>)",
          warningSoft: "rgb(var(--apex-warning-soft) / <alpha-value>)",
          warningText: "rgb(var(--apex-warning-text) / <alpha-value>)",
          danger: "rgb(var(--apex-danger) / <alpha-value>)",
          dangerSoft: "rgb(var(--apex-danger-soft) / <alpha-value>)",
          dangerText: "rgb(var(--apex-danger-text) / <alpha-value>)",
          onStrong: "rgb(var(--apex-on-strong) / <alpha-value>)",
          onStrongMuted: "rgb(var(--apex-on-strong-muted) / <alpha-value>)",
          amber: "#D97706",
          violet: "#6D5BD0"
        }
      },
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "sans-serif"
        ]
      },
      boxShadow: {
        panel: "var(--shadow-panel)",
        card: "var(--shadow-card)",
        hero: "var(--shadow-hero)"
      }
    }
  },
  plugins: []
};

export default config;
