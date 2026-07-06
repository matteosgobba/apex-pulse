import type { SafeDashboardError } from "@/lib/dashboard-types";

export function ErrorState({ error }: { error: SafeDashboardError }) {
  return (
    <section className="rounded-lg border border-rose-300/40 bg-rose-300/10 p-6">
      <p className="text-sm font-semibold uppercase tracking-[0.16em] text-rose-100">
        Dashboard Data Unavailable
      </p>
      <h2 className="mt-3 text-xl font-semibold text-apex-text">{error.message}</h2>
      <p className="mt-3 text-sm text-rose-100/80">
        Code: <span className="font-mono">{error.code}</span>
        {error.artifactType ? (
          <>
            {" "}
            Artifact: <span className="font-mono">{error.artifactType}</span>
          </>
        ) : null}
      </p>
      <p className="mt-4 text-sm leading-6 text-slate-300">
        Run the dashboard export and read-only API locally, then refresh this page. The frontend does
        not generate artifacts or run the prediction workflow.
      </p>
    </section>
  );
}
