import Link from "next/link";

import type { BacktestContext, ModelSummaryData } from "@/lib/dashboard-types";
import { formatInteger, formatText, humanizeToken } from "@/lib/formatters";

export function BacktestContextPanel({
  context,
  modelSummary
}: {
  context: BacktestContext | null | undefined;
  modelSummary: ModelSummaryData | null | undefined;
}) {
  const backtestSummary = modelSummary?.backtest_summary;
  return (
    <section className="rounded-lg border border-apex-border bg-apex-panel/85 p-5">
      <p className="text-sm font-semibold uppercase tracking-[0.16em] text-apex-muted">
        Historical Backtest Context
      </p>
      <h2 className="mt-2 text-xl font-semibold text-apex-text">
        Separate from prospective monitoring
      </h2>
      <p className="mt-2 text-sm leading-6 text-slate-300">
        Backtests provide historical context for the modeling workflow. They are not prospective
        monitoring performance and are not blended with valid monitored-event evidence.
      </p>
      <dl className="mt-5 grid gap-3 text-sm sm:grid-cols-2 lg:grid-cols-4">
        <Item label="Strategy" value={humanizeToken(context?.preferred_backtest_strategy)} />
        <Item label="Selection mode" value={humanizeToken(context?.champion_selection_mode)} />
        <Item label="Backtest events" value={formatInteger(context?.n_events)} />
        <Item label="Successful folds" value={formatInteger(context?.n_folds_successful)} />
        <Item
          label="Dataset rows"
          value={formatInteger(
            typeof backtestSummary?.dataset_rows === "number" ? backtestSummary.dataset_rows : null
          )}
        />
        <Item
          label="Checkpoints"
          value={
            Array.isArray(backtestSummary?.checkpoints)
              ? backtestSummary.checkpoints.map((item) => formatText(String(item))).join(", ")
              : "Not available"
          }
        />
      </dl>
      <Link
        href="/methodology"
        className="mt-5 inline-flex rounded-md border border-apex-accent/50 bg-apex-accent/10 px-4 py-2 text-sm font-semibold text-apex-text transition hover:bg-apex-accent/15"
      >
        Methodology
      </Link>
    </section>
  );
}

function Item({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-apex-border bg-apex-bg/45 p-3">
      <dt className="text-apex-muted">{label}</dt>
      <dd className="mt-1 break-words font-medium text-slate-100">{value}</dd>
    </div>
  );
}
