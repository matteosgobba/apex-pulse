import Link from "next/link";
import type { AutopilotStatusData } from "@/lib/dashboard-types";

export function ForecastNotes({
  status,
}: {
  status?: AutopilotStatusData | null;
}) {
  const missedWeekends = (status?.weekend_history ?? [])
    .filter(
      (weekend) =>
        !weekend.forecast_exists &&
        (weekend.action_result === "forecast_window_missed" ||
          weekend.orchestrator_state_after === "UNSUPPORTED_WEEKEND_FORMAT"),
    )
    .slice(-4)
    .reverse();
  const policy = status?.prediction_policy;
  if (!missedWeekends.length && !policy) return null;

  return (
    <details className="forecast-notes">
      <summary>
        <span>Weekend & data notes</span>
        <span className="notes-summary-end">
          {missedWeekends.length ? (
            <small>
              {missedWeekends.length}{" "}
              {missedWeekends.length === 1 ? "weekend" : "weekends"} without a
              forecast
            </small>
          ) : null}
          <span className="summary-plus" aria-hidden="true">
            +
          </span>
        </span>
      </summary>
      <div className="forecast-notes-body">
        {missedWeekends.length ? (
          <div>
            <h3>Recent weekends without a forecast</h3>
            <ul>
              {missedWeekends.map((weekend) => (
                <li key={`${weekend.season}/${weekend.event_slug}`}>
                  <strong>{weekend.event}</strong>
                  <span>
                    {weekend.orchestrator_state_after ===
                    "UNSUPPORTED_WEEKEND_FORMAT"
                      ? "This weekend format is not supported by the FP3 model."
                      : weekend.last_data_issue
                        ? "Practice timing data could not be loaded before qualifying. No retrospective prediction was added."
                        : "The safe forecast window closed before a prediction was created."}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
        {policy ? (
          <p>
            {policy.status !== "ready"
              ? "Prediction policy requires attention before the next forecast."
              : policy.mode === "uniform_only"
                ? "Prediction policy: Uniform rollback is active."
                : typeof policy.transition_prior_events === "number"
                  ? `Prediction policy: season-aware selection is enabled after ${policy.transition_prior_events} eligible, settled events in the current season. Earlier forecasts use Uniform.`
                  : "Prediction policy: season-aware selection is configured. Earlier forecasts use Uniform."}
          </p>
        ) : null}
        <Link href="/methodology">Data sources & methodology ↗</Link>
      </div>
    </details>
  );
}
