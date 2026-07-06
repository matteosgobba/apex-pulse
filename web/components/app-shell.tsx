import type { ReactNode } from "react";

import { SidebarNav } from "@/components/sidebar-nav";
import { StaleDataBanner } from "@/components/stale-data-banner";
import { Topbar } from "@/components/topbar";
import type { HealthResponse } from "@/lib/dashboard-types";

export function AppShell({
  children,
  health,
  generatedAt
}: {
  children: ReactNode;
  health: HealthResponse | null;
  generatedAt?: string | null;
}) {
  return (
    <div className="min-h-dvh bg-apex-bg text-apex-text">
      <div className="lg:flex">
        <SidebarNav />
        <div className="min-w-0 flex-1">
          <Topbar health={health} generatedAt={generatedAt} />
          <StaleDataBanner generatedAt={generatedAt} />
          <main className="mx-auto w-full max-w-7xl px-4 py-6 lg:px-8">{children}</main>
        </div>
      </div>
    </div>
  );
}
