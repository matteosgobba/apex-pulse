import type { LegacyDescriptiveRecord } from "@/lib/dashboard-types";
import { formatEventNameWithFlag } from "@/lib/event-display";
import { formatInteger, formatSeconds, formatText, humanizeToken } from "@/lib/formatters";

export function LegacyRecordsPanel({ records }: { records: LegacyDescriptiveRecord[] }) {
  return (
    <section className="rounded-lg border border-amber-300/45 bg-amber-300/10 p-5">
      <p className="text-sm font-semibold uppercase tracking-[0.16em] text-amber-100">
        Legacy Descriptive Records
      </p>
      <h2 className="mt-2 text-xl font-semibold text-apex-text">
        Excluded from valid prospective monitoring evidence
      </h2>
      <p className="mt-2 text-sm leading-6 text-amber-50">
        These records are displayed descriptively only. They do not contribute to prospective event
        counts, aggregate MAE, coverage, model-performance claims, or future candidate evidence.
      </p>
      <div className="mt-5 grid gap-3 md:grid-cols-2">
        {records.length > 0 ? (
          records.map((record) => (
            <article
              key={`${record.event_identity?.season}-${record.event_identity?.event_slug}`}
              className="rounded-lg border border-amber-200/30 bg-apex-bg/45 p-4"
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.12em] text-amber-100/80">
                    {formatInteger(record.event_identity?.season)}
                  </p>
                  <h3 className="mt-1 text-lg font-semibold text-apex-text">
                    {formatEventNameWithFlag(record.event_identity?.event) || formatText(null)}
                  </h3>
                </div>
                <span className="rounded-full bg-amber-300/10 px-2 py-1 text-xs font-semibold text-amber-100">
                  Legacy
                </span>
              </div>
              <dl className="mt-4 grid gap-2 text-sm">
                <Row label="Eligible" value="No" />
                <Row label="Lifecycle" value={humanizeToken(record.lifecycle_state)} />
                <Row label="Reason" value={humanizeToken(record.exclusion_reason)} />
                <Row label="Forecast rows" value={formatInteger(record.descriptive_metrics?.forecast_rows)} />
                <Row label="Scored rows" value={formatInteger(record.descriptive_metrics?.scored_rows)} />
                <Row label="Descriptive MAE" value={formatSeconds(record.descriptive_metrics?.mae_gap_sec)} />
              </dl>
            </article>
          ))
        ) : (
          <p className="text-sm text-amber-50">No legacy descriptive records are exported.</p>
        )}
      </div>
    </section>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start justify-between gap-4">
      <dt className="text-amber-100/75">{label}</dt>
      <dd className="max-w-[60%] break-words text-right text-amber-50">{value}</dd>
    </div>
  );
}
