import type { SessionStatus } from "@/lib/dashboard-types";
import { formatDateTime, humanizeToken } from "@/lib/formatters";

export function PracticeSessionCard({
  session,
  index
}: {
  session: SessionStatus;
  index: number;
}) {
  return (
    <article className="relative rounded-lg border border-apex-border bg-apex-panel/85 p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-apex-muted">
            Step {index + 1}
          </p>
          <h3 className="mt-1 text-2xl font-semibold text-apex-text">{session.session}</h3>
        </div>
        <span
          className={`rounded-full px-2.5 py-1 text-xs font-semibold ${
            session.available
              ? "bg-emerald-300/10 text-emerald-100"
              : "bg-slate-400/10 text-slate-300"
          }`}
          aria-label={`${session.session} ${session.available ? "available" : "unavailable"}`}
        >
          {session.available ? "Available" : "Unavailable"}
        </span>
      </div>
      <dl className="mt-4 grid gap-3 text-sm">
        <Row label="Status" value={humanizeToken(session.status)} />
        <Row label="Timestamp" value={formatDateTime(session.last_known_timestamp)} mono />
        <Row label="Reason" value={humanizeToken(session.reason)} />
      </dl>
    </article>
  );
}

function Row({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-start justify-between gap-4">
      <dt className="text-apex-muted">{label}</dt>
      <dd
        className={`max-w-[60%] break-words text-right text-slate-100 ${
          mono ? "font-mono" : ""
        }`}
      >
        {value}
      </dd>
    </div>
  );
}
