import type { ReactNode } from "react";

import { DashboardAutoRefresh } from "@/components/dashboard-auto-refresh";
import { SiteFooter } from "@/components/site-footer";
import { SiteHeader } from "@/components/site-header";
import { RacingMotion } from "@/components/racing-motion";
import { ThemeProvider } from "@/components/theme-provider";
import type { HealthResponse } from "@/lib/dashboard-types";

export function AppShell({
  children,
  health,
  autoRefresh = false,
  landing = false,
}: {
  children: ReactNode;
  health: HealthResponse | null;
  generatedAt?: string | null;
  autoRefresh?: boolean;
  landing?: boolean;
}) {
  return (
    <ThemeProvider>
      <RacingMotion>
        {autoRefresh ? <DashboardAutoRefresh /> : null}
        <a className="skip-link" href="#main-content">
          Skip to content
        </a>
        <div className="min-h-dvh bg-apex-bg text-apex-text transition-colors">
          <SiteHeader health={health} />
          <main
            id="main-content"
            className={`ap-main ${landing ? "" : "page-main"}`}
          >
            {children}
          </main>
          <SiteFooter />
        </div>
      </RacingMotion>
    </ThemeProvider>
  );
}
