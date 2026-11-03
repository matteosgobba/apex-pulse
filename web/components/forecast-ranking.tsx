"use client";

import { useState, type CSSProperties } from "react";
import { TeamMark } from "@/components/team-mark";
import { formatSignedGap } from "@/lib/formatters";
import type {
  PublicComparisonRow,
  PublicRankingRow,
} from "@/lib/public-view-model";

export function ForecastRanking({
  rows,
  comparison = [],
}: {
  rows: PublicRankingRow[];
  comparison?: PublicComparisonRow[];
}) {
  const [query, setQuery] = useState("");
  const [expanded, setExpanded] = useState(false);
  const [selectedCode, setSelectedCode] = useState<string | null>(null);
  const [reality, setReality] = useState(false);
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
            More racing intelligence.
          </h2>
        </div>
        <p className="section-intro">
          One clear view of the model’s call.
          <br />
          Explore a driver. Check the outcome.
          <br />
          See where the prediction stands.
        </p>
      </div>
      {rows.length === 0 ? (
        <div className="leaderboard-panel p-8">
          <h3>Prediction unavailable</h3>
          <p>No forecast rows are present in the current dashboard export.</p>
        </div>
      ) : (
        <div className="pit-grid">
          <div className="leaderboard-panel">
            <div className="table-toolbar">
              <div className="segmented" aria-label="Forecast view">
                <button
                  type="button"
                  className={!reality ? "selected" : ""}
                  aria-pressed={!reality}
                  onClick={() => setReality(false)}
                >
                  Prediction
                </button>
                <button
                  type="button"
                  className={reality ? "selected" : ""}
                  aria-pressed={reality}
                  disabled={!comparison.length}
                  onClick={() => setReality(true)}
                >
                  Versus reality
                </button>
              </div>
              <label className="search-box">
                <span aria-hidden="true">⌕</span>
                <span className="sr-only">Search driver or team</span>
                <input
                  type="search"
                  placeholder="Driver or team…"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                />
              </label>
            </div>
            <div className="ranking-labels" aria-hidden="true">
              <span>POS</span>
              <span>DRIVER / TEAM</span>
              <span>MODEL GAP</span>
              <span>{reality ? "ACTUAL" : "PACE SPREAD"}</span>
            </div>
            <ol
              aria-label="Predicted qualifying ranking"
              className="racing-ranking"
            >
              {visible.map((row) => {
                const result = comparison.find(
                  (item) => item.driverCode === row.driverCode,
                );
                const spread =
                  row.intervalUpperSec !== null && row.intervalLowerSec !== null
                    ? Math.max(0, row.intervalUpperSec - row.intervalLowerSec)
                    : null;
                return (
                  <li
                    key={row.driverCode}
                    className={`ranking-row ${selected?.driverCode === row.driverCode ? "selected" : ""}`}
                    style={
                      { "--team-color": row.team.primary } as CSSProperties
                    }
                  >
                    <span className="rank-position">
                      {row.predictedPosition === null
                        ? "—"
                        : String(row.predictedPosition).padStart(2, "0")}
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
                      {formatSignedGap(row.predictedGapSec)}
                    </span>
                    <span className="spread-cell">
                      {reality ? (
                        <span title="Official qualifying position">
                          {result?.actualPosition
                            ? `P${result.actualPosition}`
                            : "—"}
                        </span>
                      ) : spread !== null ? (
                        <span
                          className="spread-bar"
                          title={`Prediction interval width: ${spread.toFixed(3)} seconds`}
                        >
                          <i
                            style={{
                              width: `${Math.min(100, (spread / 3) * 100)}%`,
                            }}
                          />
                        </span>
                      ) : (
                        <span className="muted">—</span>
                      )}
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
              Model gap = estimated seconds to the actual pole benchmark. Rank 1
              may have a non-zero estimate.
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
                <span className="mono small red-type">↗</span>
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
                    ? `The preserved forecast placed ${selected.driverCode} P${selected.predictedPosition}; official qualifying position: P${actual.actualPosition}.`
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
