import type { SafeDashboardError } from "@/lib/dashboard-types";

export function ErrorState({ error }: { error: SafeDashboardError }) {
  const showLocalGuidance = process.env.NODE_ENV !== "production";
  return (
    <section className="rounded-lg border border-rose-300/40 bg-rose-300/10 p-6">
      <p className="text-sm font-semibold uppercase tracking-[0.16em] text-rose-100">
        Dashboard API Unavailable
      </p>
      <h2 className="mt-3 text-xl font-semibold text-apex-text">
        The dashboard API is not currently reachable or cannot serve this artifact safely.
      </h2>
      <p className="mt-3 text-sm leading-6 text-rose-50/90">{error.message}</p>
      <p className="mt-3 text-sm text-rose-100/80">
        Code: <span className="font-mono">{error.code}</span>
        {error.artifactType ? (
          <>
            {" "}
            Artifact: <span className="font-mono">{error.artifactType}</span>
          </>
        ) : null}
      </p>
      {showLocalGuidance ? (
        <p className="mt-4 text-sm leading-6 text-slate-300">
          Local development: run the dashboard export and read-only API, then refresh this page. The
          frontend does not generate artifacts or run the prediction workflow.
        </p>
      ) : null}
    </section>
  );
}
