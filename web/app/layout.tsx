import type { Metadata } from "next";

import "@/app/globals.css";

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

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
