export function TechnicalDetails({
  items,
  title = "Artifact details"
}: {
  items: Array<{ label: string; value: string | number | boolean | null | undefined }>;
  title?: string;
}) {
  const available = items.filter(
    (item) => item.value !== null && item.value !== undefined && item.value !== ""
  );
  return (
    <details className="group rounded-2xl border border-apex-border bg-apex-panel">
      <summary className="cursor-pointer list-none px-5 py-4 text-sm font-semibold text-apex-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-apex-accent">
        <span className="flex items-center justify-between gap-4">
          {title}
          <span aria-hidden="true" className="text-apex-muted transition-transform group-open:rotate-45">
            +
          </span>
        </span>
      </summary>
      <dl className="grid gap-x-8 gap-y-4 border-t border-apex-border px-5 py-5 sm:grid-cols-2 lg:grid-cols-3">
        {available.map((item) => (
          <div key={item.label}>
            <dt className="text-xs font-medium text-apex-muted">{item.label}</dt>
            <dd className="mt-1 break-all text-sm text-apex-text">
              {typeof item.value === "boolean" ? (item.value ? "Yes" : "No") : String(item.value)}
            </dd>
          </div>
        ))}
      </dl>
    </details>
  );
}
