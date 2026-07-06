import type { Metadata } from "next";

import "@/app/globals.css";

export const metadata: Metadata = {
  title: "Apex Pulse Dashboard",
  description: "Read-only Formula 1 qualifying forecast monitoring dashboard."
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
