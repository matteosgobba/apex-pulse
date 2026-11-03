import type { Metadata, Viewport } from "next";
import { Analytics } from "@vercel/analytics/next";

import "@/app/globals.css";
import "@/app/racing.css";
import {
  DEFAULT_THEME,
  THEME_COLORS,
  THEME_INITIALIZATION_SCRIPT
} from "@/lib/theme";

export const metadata: Metadata = {
  title: "Apex Pulse",
  description:
    "Machine-learning Formula 1 qualifying predictions built from public free-practice data.",
  icons: {
    icon: {
      url: "/brand/favicon.svg",
      type: "image/svg+xml",
      sizes: "any"
    }
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
      <body>
        {children}
        <Analytics />
      </body>
    </html>
  );
}
