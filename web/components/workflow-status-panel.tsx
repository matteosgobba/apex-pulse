import Link from "next/link";

import { StatusCard } from "@/components/status-card";
import type { CurrentEventData, LifecycleState, PracticeStatusData } from "@/lib/dashboard-types";
import { formatDateTime, formatInteger } from "@/lib/formatters";

export function WorkflowStatusPanel({
  currentEvent,
  practiceStatus
}: {
  currentEvent: CurrentEventData | null | undefined;
  practiceStatus: PracticeStatusData | null | undefined;
}) {
  const state =
    currentEvent?.lifecycle?.state ?? practiceStatus?.lifecycle_state ?? "no_event_available";
  const readiness = practiceStatus?.monitoring_readiness;
  const preflight = practiceStatus?.preflight ?? currentEvent?.preflight;
  const forecast = currentEvent?.forecast_status;
  const settlement = currentEvent?.settlement_status;

  return (
    <div className="space-y-4">
      <section className="rounded-lg border border-apex-border bg-apex-panel/85 p-5">
        <h2 className="text-lg font-semibold text-apex-text">Operator Workflow Status</h2>
        <p className="mt-2 text-sm leading-6 text-slate-300">{workflowMessage(state)}</p>
        {state === "forecast_available" || state === "awaiting_qualifying_targets" ? (
          <Link
            href="/forecast"
            className="mt-4 inline-flex rounded-md border border-apex-accent/50 bg-apex-accent/10 px-4 py-2 text-sm font-semibold text-apex-text transition hover:bg-apex-accent/15"
          >
            Open forecast view
          </Link>
        ) : null}
      </section>
      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <StatusCard
          title="Monitoring readiness"
          status={readiness?.status}
          items={[
            { label: "Chronology", value: readiness?.chronological_order_status },
            { label: "Target isolation", value: readiness?.target_isolation_status },
            { label: "Forecastable", value: formatInteger(readiness?.forecastable_event_count) },
            { label: "Settleable", value: formatInteger(readiness?.settleable_event_count) }
          ]}
        />
        <StatusCard
          title="Preflight"
          status={preflight?.status}
          tone={preflight?.forecast_allowed ? "good" : "danger"}
          detail={blockedDetail(state, preflight?.next_required_command)}
          items={[
            { label: "Allowed", value: preflight?.forecast_allowed },
            { label: "Blocking", value: preflight?.blocking_check_count },
            { label: "Warnings", value: preflight?.warning_check_count }
          ]}
        />
        <StatusCard
          title="Forecast"
          status={forecast?.available ? "Available" : "Not available"}
          tone={forecast?.available ? "good" : "neutral"}
          items={[
            { label: "Checkpoint", value: forecast?.checkpoint },
            { label: "Drivers", value: forecast?.forecasted_driver_count },
            { label: "Generated", value: formatDateTime(forecast?.forecast_created_at_utc) }
          ]}
        />
        <StatusCard
          title="Settlement"
          status={settlement?.available ? "Available" : "Not available"}
          tone={settlement?.settlement_valid ? "good" : "neutral"}
          detail={
            state === "settled"
              ? "Qualifying targets have been incorporated. Detailed comparison arrives in a future dashboard update."
              : null
          }
          items={[
            { label: "Valid", value: settlement?.settlement_valid },
            { label: "Scored", value: settlement?.scored_driver_count },
            { label: "Excluded", value: settlement?.excluded_driver_count }
          ]}
        />
      </section>
      {practiceStatus?.notes && practiceStatus.notes.length > 0 ? (
        <section className="rounded-lg border border-apex-border bg-apex-bg/45 p-4">
          <h2 className="text-sm font-semibold uppercase tracking-[0.14em] text-apex-muted">
            Artifact Notes
          </h2>
          <ul className="mt-3 space-y-2 text-sm leading-6 text-slate-300">
            {practiceStatus.notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </section>
      ) : null}
    </div>
  );
}

function workflowMessage(state: LifecycleState): string {
  const messages: Record<LifecycleState, string> = {
    no_event_available:
      "No monitored event is available in the exported dashboard artifacts.",
    practice_in_progress:
      "Practice evidence is still developing. More validated inputs may be needed before forecasting.",
    ready_to_forecast:
      "Validated inputs are available. A forecast can be created through the separate operator workflow; this dashboard does not create it.",
    forecast_available:
      "A forecast snapshot exists. Use the forecast view to inspect the exported leaderboard.",
    awaiting_qualifying_targets:
      "A forecast snapshot exists and qualifying target ingestion is pending in the operator workflow.",
    settled:
      "Qualifying targets have been incorporated. Detailed settlement comparison is planned for a later dashboard milestone.",
    settled_partial_coverage:
      "Qualifying targets have been incorporated for the entrants covered by the preserved forecast.",
    blocked:
      "Preflight blocked the forecast workflow. Only safe artifact-level blocking context is shown here.",
    legacy_descriptive_only:
      "This is a noncanonical legacy descriptive record and is not valid prospective monitoring evidence."
  };
  return messages[state];
}

function blockedDetail(state: LifecycleState, command: string | null | undefined): string | null {
  if (state !== "blocked" && state !== "legacy_descriptive_only") {
    return null;
  }
  return command ? "Next operator context is available in the exported preflight runbook." : null;
}
