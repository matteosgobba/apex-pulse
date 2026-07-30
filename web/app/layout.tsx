import type { Metadata, Viewport } from "next";

import "@/app/globals.css";
import {
  DEFAULT_THEME,
  THEME_COLORS,
  THEME_INITIALIZATION_SCRIPT
} from "@/lib/theme";

export const metadata: Metadata = {
  title: {
    default: "Apex Pulse — Formula 1 Qualifying Predictions",
    template: "%s — Apex Pulse"
  },
  description:
    "Machine-learning Formula 1 qualifying predictions built from public free-practice data.",
  icons: {
    icon: "/brand/apex-pulse-logo-simple.png"
  }
};

export const viewport: Viewport = {
  colorScheme: "dark light",
  themeColor: THEME_COLORS.dark
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" data-theme={DEFAULT_THEME} suppressHydrationWarning>
      <head>
        <script
          id="apex-pulse-theme-init"
          dangerouslySetInnerHTML={{ __html: THEME_INITIALIZATION_SCRIPT }}
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
