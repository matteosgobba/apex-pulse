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
          bg: "#080b10",
          panel: "#10151f",
          panelSoft: "#151c28",
          border: "#263244",
          muted: "#9ca8b8",
          text: "#eef4fb",
          accent: "#6ee7f9",
          amber: "#f5c56b",
          violet: "#a7a5ff"
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
        panel: "0 24px 80px rgba(0, 0, 0, 0.24)"
      }
    }
  },
  plugins: []
};

export default config;
