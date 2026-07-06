import type { DashboardRows } from "@/lib/dashboard-types";

export function dashboardRows<TRow>(rows: DashboardRows<TRow> | null | undefined): TRow[] {
  return Array.isArray(rows) ? rows : [];
}
