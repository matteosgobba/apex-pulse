import { ErrorState } from "@/components/error-state";
import { LegacyWarning } from "@/components/legacy-warning";
import { LifecycleBadge } from "@/components/lifecycle-badge";
import { PracticeTimeline } from "@/components/practice-timeline";
import { WorkflowStatusPanel } from "@/components/workflow-status-panel";
import type { PracticePageData } from "@/lib/dashboard-types";
import { formatInteger, formatText, humanizeToken } from "@/lib/formatters";

export function PracticePageView({ data }: { data: PracticePageData }) {
  if (data.error) {
    return <ErrorState error={data.error} />;
  }

  const current = data.currentEvent?.data;
  const practice = data.practiceStatus?.data;
  const identity = practice?.event_identity ?? current?.event_identity;
  const lifecycleState = current?.lifecycle?.state ?? practice?.lifecycle_state;
  const legacy =
    lifecycleState === "legacy_descriptive_only" || current?.legacy_status?.legacy_noncanonical;

  return (
    <div className="space-y-6">
      {legacy ? <LegacyWarning /> : null}
      <section className="rounded-lg border border-apex-border bg-apex-panel p-6 shadow-panel">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="min-w-0">
            <p className="text-sm font-semibold uppercase tracking-[0.18em] text-apex-muted">
              {formatText(identity?.season)}
            </p>
            <h1 className="mt-2 break-words text-3xl font-semibold text-apex-text md:text-5xl">
              {formatText(identity?.event)} Practice Status
            </h1>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-300">
              This page represents exported session availability and monitoring readiness. This
              dashboard is read-only. It does not ingest sessions, run preflight, or create
              forecasts.
            </p>
          </div>
          <div className="flex shrink-0 flex-col gap-3">
            <LifecycleBadge
              state={lifecycleState}
              label={current?.lifecycle?.display_label ?? humanizeToken(lifecycleState)}
            />
            <div className="rounded-md border border-apex-border bg-apex-bg/45 p-3 text-sm">
              <p className="text-apex-muted">Event order</p>
              <p className="mt-1 font-mono text-lg text-apex-text">
                {formatInteger(identity?.event_order)}
              </p>
            </div>
          </div>
        </div>
      </section>
      <PracticeTimeline sessions={practice?.sessions ?? []} />
      <WorkflowStatusPanel currentEvent={current} practiceStatus={practice} />
    </div>
  );
}
