export function TableEmptyState({
  title,
  message
}: {
  title: string;
  message: string;
}) {
  return (
    <section className="rounded-lg border border-dashed border-apex-border bg-apex-panel/70 p-6 text-center">
      <p className="text-sm font-semibold uppercase tracking-[0.16em] text-apex-muted">
        Unavailable
      </p>
      <h2 className="mt-3 text-xl font-semibold text-apex-text">{title}</h2>
      <p className="mx-auto mt-2 max-w-2xl text-sm leading-6 text-slate-300">{message}</p>
    </section>
  );
}
