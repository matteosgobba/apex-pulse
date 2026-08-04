import type { ReactNode } from "react";

import { DashboardAutoRefresh } from "@/components/dashboard-auto-refresh";
import { SiteFooter } from "@/components/site-footer";
import { SiteHeader } from "@/components/site-header";
import { ThemeProvider } from "@/components/theme-provider";
import type { HealthResponse } from "@/lib/dashboard-types";

export function AppShell({
  children,
  health,
  autoRefresh = false
}: {
  children: ReactNode;
  health: HealthResponse | null;
  generatedAt?: string | null;
  autoRefresh?: boolean;
}) {
  return (
    <ThemeProvider>
      {autoRefresh ? <DashboardAutoRefresh /> : null}
      <div className="min-h-dvh bg-apex-bg text-apex-text transition-colors">
        <SiteHeader health={health} />
        <main className="mx-auto w-full max-w-7xl px-4 py-6 sm:py-10 lg:px-8">{children}</main>
        <SiteFooter />
      </div>
    </ThemeProvider>
  );
}
