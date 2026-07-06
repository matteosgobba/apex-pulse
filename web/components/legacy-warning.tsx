export function LegacyWarning() {
  return (
    <section className="rounded-lg border border-amber-300/45 bg-amber-300/10 p-4">
      <p className="text-sm font-semibold uppercase tracking-[0.16em] text-amber-100">
        Legacy Descriptive Record
      </p>
      <p className="mt-2 text-sm leading-6 text-amber-50">
        Not eligible as valid prospective monitoring evidence. This event may be displayed for
        descriptive continuity, but it is quarantined from prospective aggregate evidence.
      </p>
    </section>
  );
}
