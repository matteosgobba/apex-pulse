"use client";

import { useState, type CSSProperties, type ReactNode } from "react";
import { TeamMark } from "@/components/team-mark";
import { formatSignedGap } from "@/lib/formatters";
import type {
  PublicComparisonRow,
  PublicRankingRow,
} from "@/lib/public-view-model";

export function ForecastRanking({
  rows,
  comparison = [],
  context,
}: {
  rows: PublicRankingRow[];
  comparison?: PublicComparisonRow[];
  context?: ReactNode;
}) {
  const [query, setQuery] = useState("");
  const [expanded, setExpanded] = useState(false);
  const [selectedCode, setSelectedCode] = useState<string | null>(null);
  const hasResults = comparison.some((row) => row.actualPosition !== null);
  const selected =
    rows.find((row) => row.driverCode === selectedCode) ?? rows[0];
  const actual = comparison.find(
    (row) => row.driverCode === selected?.driverCode,
  );
  const matching = rows.filter((row) =>
    `${row.driverCode} ${row.driverName ?? ""} ${row.team.displayName}`
      .toLowerCase()
      .includes(query.toLowerCase().trim()),
  );
  const visible = expanded || query.trim() ? matching : matching.slice(0, 7);
  return (
    <section id="forecast" aria-labelledby="forecast-title">
      <div className="section-heading">
        <div>
          <p className="eyebrow">
            <span className="section-number">01 /</span> THE PIT WALL
          </p>
          <h2 id="forecast-title">
            Less noise.
            <br />
            <span className="muted-type">More racing intelligence.</span>
          </h2>
        </div>
        <p className="section-intro">
          One clear view of the model’s call.
          <br /> Explore a driver. Check the outcome.
          <br /> See where the prediction stands.
        </p>
      </div>
      {context ? <div className="forecast-context-block">{context}</div> : null}
      {rows.length === 0 ? (
        <div className="leaderboard-panel p-8">
          <h3>Prediction unavailable</h3>
          <p>No forecast rows are present in the current dashboard export.</p>
        </div>
      ) : (
        <div className="pit-grid">
          <div className="leaderboard-panel unified-ranking">
            <div className="table-toolbar">
              <span className="ranking-caption">QUALIFYING FORECAST</span>
              <label className="search-box">
                <svg
                  width="15"
                  height="15"
                  viewBox="0 0 20 20"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  aria-hidden="true"
                >
                  <circle cx="8.5" cy="8.5" r="5.5" />
                  <path d="m13 13 4 4" />
                </svg>
                <span className="sr-only">Search driver or team</span>
                <input
                  type="search"
                  placeholder="Driver or team…"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                />
              </label>
            </div>
            <p className="comparison-key">
              {hasResults ? (
                <>
                  In predicted order. <span>↑ Higher</span> <span>↓ Lower</span>{" "}
                  <span>= Matched</span>
                  <span className="sr-only">
                    Arrows compare the official qualifying position with the
                    prediction.
                  </span>
                </>
              ) : (
                "Awaiting official qualifying results. Predictions are shown below."
              )}
            </p>
            <div className="ranking-labels" aria-hidden="true">
              <span>PREDICTED</span>
              <span>DRIVER / TEAM</span>
              <span className="model-gap-heading">MODEL GAP</span>
              <span>OFFICIAL</span>
              <span>CHANGE</span>
            </div>
            <ol
              aria-label="Predicted qualifying ranking"
              className="racing-ranking"
            >
              {visible.map((row) => {
                const result = comparison.find(
                  (item) => item.driverCode === row.driverCode,
                );
                return (
                  <li
                    key={row.driverCode}
                    className={`ranking-row ${selected?.driverCode === row.driverCode ? "selected" : ""}`}
                    style={
                      { "--team-color": row.team.primary } as CSSProperties
                    }
                  >
                    <span
                      className="rank-position"
                      aria-label={`Predicted position ${row.predictedPosition ?? "unavailable"}`}
                    >
                      {row.predictedPosition === null
                        ? "—"
                        : `P${row.predictedPosition}`}
                    </span>
                    <button
                      type="button"
                      className="driver-button"
                      aria-label={`Explore ${row.driverCode}`}
                      aria-pressed={selected?.driverCode === row.driverCode}
                      onClick={() => setSelectedCode(row.driverCode)}
                    >
                      <TeamMark team={row.team} size="sm" />
                      <span>
                        <strong>{row.driverCode}</strong>
                        <small>{row.team.displayName}</small>
                      </span>
                    </button>
                    <span
                      className={`row-gap ${row.predictedPosition === 1 ? "pole" : ""}`}
                    >
                      <span className="inline-gap-label" aria-hidden="true">
                        Gap{" "}
                      </span>
                      <span
                        aria-label={`Model gap ${formatSignedGap(row.predictedGapSec)}`}
                      >
                        {formatSignedGap(row.predictedGapSec)}
                      </span>
                    </span>
                    <span
                      className="official-position"
                      aria-label={`Official position ${result?.actualPosition ?? "unavailable"}`}
                    >
                      {result?.actualPosition
                        ? `P${result.actualPosition}`
                        : "—"}
                    </span>
                    <span className="position-change-cell">
                      <PositionChange
                        predicted={row.predictedPosition}
                        actual={result?.actualPosition}
                        awaitingResults={!hasResults}
                      />
                    </span>
                  </li>
                );
              })}
            </ol>
            {!visible.length ? (
              <p className="no-results" role="status">
                No drivers match your search.
              </p>
            ) : null}
            <div className="table-bottom">
              <span className="mono small muted">
                {visible.length} / {rows.length} DRIVERS
              </span>
              {rows.length > 7 ? (
                <button
                  className="text-link"
                  type="button"
                  onClick={() => setExpanded((value) => !value)}
                >
                  {expanded ? "Show top 7 ↑" : "Show full field ↓"}
                </button>
              ) : null}
            </div>
            <p className="table-disclosure">
              Model gap = estimated seconds to the pole benchmark. Arrows
              compare the official position with the prediction; rows stay in
              predicted order.
            </p>
          </div>
          {selected ? (
            <aside
              className="driver-panel"
              aria-label="Selected driver details"
              style={{ "--team-color": selected.team.primary } as CSSProperties}
            >
              <div className="driver-panel-top">
                <span className="eyebrow muted">DRIVER SPOTLIGHT</span>
                <span className="mono small muted">
                  P{selected.predictedPosition ?? "—"}
                </span>
              </div>
              <span className="driver-monogram" aria-hidden="true">
                {selected.driverCode}
              </span>
              <div className="driver-title">
                <p className="team-rule">
                  {selected.team.displayName.toUpperCase()}
                </p>
                <h3>{selected.driverName ?? selected.driverCode}</h3>
                <p>
                  {selected.driverCode} / PREDICTED P
                  {selected.predictedPosition ?? "—"}
                </p>
              </div>
              <div className="detail-values">
                <div>
                  <span>Model gap to pole</span>
                  <strong>{formatSignedGap(selected.predictedGapSec)}</strong>
                </div>
                <div>
                  <span>Official position</span>
                  <strong>
                    {actual?.actualPosition ? `P${actual.actualPosition}` : "—"}
                  </strong>
                </div>
              </div>
              <div className="pace-comparison">
                <div>
                  <span>PREDICTED GAP</span>
                  <span>{formatSignedGap(selected.predictedGapSec)}</span>
                </div>
                <div className="pace-rail">
                  <span
                    style={{
                      width: `${Math.min(100, (Math.max(0, selected.predictedGapSec ?? 0) / 3) * 100)}%`,
                    }}
                  />
                </div>
                {actual?.actualGapSec !== null &&
                actual?.actualGapSec !== undefined ? (
                  <>
                    <div>
                      <span>OFFICIAL GAP</span>
                      <span>{formatSignedGap(actual.actualGapSec)}</span>
                    </div>
                    <div className="pace-rail actual">
                      <span
                        style={{
                          width: `${Math.min(100, (Math.max(0, actual.actualGapSec) / 3) * 100)}%`,
                        }}
                      />
                    </div>
                  </>
                ) : null}
              </div>
              <p className="driver-insight">
                <span className="insight-symbol">↳</span>
                <span>
                  {actual?.actualPosition
                    ? `Predicted P${selected.predictedPosition} → qualified P${actual.actualPosition}. ${selected.predictedPosition === actual.actualPosition ? "An exact position match." : "See the position change in the ranking."}`
                    : "This forecast was preserved before qualifying. Official results appear only after settlement."}
                </span>
              </p>
              <p className="detail-note">
                Public practice data. No private team telemetry. Estimates are
                model outputs, not guarantees.
              </p>
            </aside>
          ) : null}
        </div>
      )}
    </section>
  );
}

function PositionChange({
  predicted,
  actual,
  awaitingResults,
}: {
  predicted: number | null;
  actual?: number | null;
  awaitingResults: boolean;
}) {
  if (predicted === null || actual === null || actual === undefined) {
    return (
      <span
        className="position-change is-pending"
        aria-label="Position change unavailable"
      >
        —
        <small>
          {actual == null
            ? awaitingResults
              ? "Awaiting"
              : "No result"
            : "No rank"}
        </small>
      </span>
    );
  }
  const delta = predicted - actual;
  const direction = delta === 0 ? "matched" : delta > 0 ? "higher" : "lower";
  const label =
    delta === 0
      ? "Exact position match"
      : `Qualified ${Math.abs(delta)} ${Math.abs(delta) === 1 ? "place" : "places"} ${direction} than predicted`;
  return (
    <span
      className={`position-change is-${direction}`}
      aria-label={label}
      title={label}
    >
      <strong>
        {delta === 0 ? "=" : `${delta > 0 ? "↑" : "↓"} ${Math.abs(delta)}`}
      </strong>
      <small>{delta === 0 ? "Matched" : delta > 0 ? "Higher" : "Lower"}</small>
    </span>
  );
}
