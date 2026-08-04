"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef } from "react";

export const DASHBOARD_REFRESH_INTERVAL_MS = 60_000;

export function DashboardAutoRefresh({
  intervalMs = DASHBOARD_REFRESH_INTERVAL_MS
}: {
  intervalMs?: number;
}) {
  const router = useRouter();
  const refreshQueued = useRef(false);

  const refresh = useCallback(() => {
    if (document.visibilityState === "hidden" || refreshQueued.current) {
      return;
    }
    refreshQueued.current = true;
    router.refresh();
    window.queueMicrotask(() => {
      refreshQueued.current = false;
    });
  }, [router]);

  useEffect(() => {
    const timer = window.setInterval(refresh, intervalMs);
    const onVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        refresh();
      }
    };
    document.addEventListener("visibilitychange", onVisibilityChange);
    window.addEventListener("focus", refresh);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisibilityChange);
      window.removeEventListener("focus", refresh);
    };
  }, [intervalMs, refresh]);

  return null;
}
