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
    <section
      className={`min-w-0 rounded-lg border ${toneClass} bg-apex-panel/85 p-4 shadow-panel`}
    >
      <div className="flex min-w-0 items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="text-sm font-semibold text-apex-text">{title}</h3>
          <p className="mt-1 break-words text-sm text-apex-muted [overflow-wrap:anywhere]">
            {formatText(status)}
          </p>
        </div>
        <span className="mt-1 h-2.5 w-2.5 shrink-0 rounded-full bg-apex-accent" aria-hidden="true" />
      </div>
      {detail ? (
        <p className="mt-3 break-words text-sm leading-6 text-slate-300 [overflow-wrap:anywhere]">
          {detail}
        </p>
      ) : null}
      {items.length > 0 ? (
        <dl className="mt-4 grid gap-3 text-sm">
          {items.map((item) => (
            <div key={item.label} className="flex min-w-0 items-start justify-between gap-3">
              <dt className="min-w-0 break-words text-apex-muted [overflow-wrap:anywhere]">
                {item.label}
              </dt>
              <dd className="min-w-0 max-w-[58%] break-words text-right font-medium text-slate-100 [overflow-wrap:anywhere]">
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
