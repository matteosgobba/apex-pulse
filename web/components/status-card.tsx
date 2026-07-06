import { formatText } from "@/lib/formatters";

export interface StatusCardItem {
  label: string;
  value: string | number | boolean | null | undefined;
}

export function StatusCard({
  title,
  status,
  detail,
  items = [],
  tone = "neutral"
}: {
  title: string;
  status: string | null | undefined;
  detail?: string | null;
  items?: StatusCardItem[];
  tone?: "neutral" | "good" | "warn" | "danger";
}) {
  const toneClass = {
    neutral: "border-apex-border",
    good: "border-emerald-300/40",
    warn: "border-amber-300/40",
    danger: "border-rose-300/40"
  }[tone];

  return (
    <section className={`rounded-lg border ${toneClass} bg-apex-panel/85 p-4 shadow-panel`}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-apex-text">{title}</h3>
          <p className="mt-1 text-sm text-apex-muted">{formatText(status)}</p>
        </div>
        <span className="mt-1 h-2.5 w-2.5 shrink-0 rounded-full bg-apex-accent" aria-hidden="true" />
      </div>
      {detail ? <p className="mt-3 text-sm leading-6 text-slate-300">{detail}</p> : null}
      {items.length > 0 ? (
        <dl className="mt-4 grid gap-3 text-sm">
          {items.map((item) => (
            <div key={item.label} className="flex items-start justify-between gap-3">
              <dt className="text-apex-muted">{item.label}</dt>
              <dd className="max-w-[58%] text-right font-medium text-slate-100">
                {formatText(
                  typeof item.value === "boolean" ? (item.value ? "Yes" : "No") : item.value
                )}
              </dd>
            </div>
          ))}
        </dl>
      ) : null}
    </section>
  );
}
