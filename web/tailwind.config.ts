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
          bg: "#F7F8FA",
          panel: "#FFFFFF",
          panelSoft: "#F1F5F9",
          surface: "#F1F5F9",
          border: "#E2E8F0",
          muted: "#64748B",
          text: "#0F172A",
          ink: "#111827",
          accent: "#E10600",
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
        panel: "0 16px 50px rgba(15, 23, 42, 0.08)",
        card: "0 8px 30px rgba(15, 23, 42, 0.055)",
        hero: "0 24px 70px rgba(15, 23, 42, 0.16)"
      }
    }
  },
  plugins: []
};

export default config;
