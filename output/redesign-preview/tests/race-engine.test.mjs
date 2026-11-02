import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { selectWeekend, lightCount, countdownParts, projectCircuit, countryFlag, START_GRACE_MS } from "../race-engine.mjs";

const data = JSON.parse(await readFile(new URL("../data/weekends.json", import.meta.url), "utf8"));
const brazil = data.events.find(event => event.countryCode === "BR");
const vegas = data.events.find(event => event.location === "Las Vegas");
const time = value => Date.parse(value);

test("calendar conversion agrees with the published São Paulo UTC schedule", () => {
  assert.deepEqual(brazil.sessions.map(s => s.startUtc), ["2026-11-06T15:30:00Z", "2026-11-06T19:00:00Z", "2026-11-07T14:30:00Z", "2026-11-07T18:00:00Z", "2026-11-08T17:00:00Z"]);
});
test("the next GP on 2 November is São Paulo, counting down to FP1", () => {
  const result = selectWeekend(data.events, time("2026-11-02T19:00:00Z"));
  assert.equal(result.event.id, brazil.id);
  assert.equal(result.session.code, "FP1");
});
test("Las Vegas track-local Thursday evening converts to Friday UTC", () => {
  assert.equal(vegas.sessions[0].startUtc, "2026-11-20T00:30:00Z");
});
test("each session gets a zero state, then advances through qualifying and race", () => {
  for (let i = 0; i < brazil.sessions.length; i++) {
    const target = time(brazil.sessions[i].startUtc);
    assert.equal(selectWeekend(data.events, target - 1).session.code, brazil.sessions[i].code);
    assert.equal(selectWeekend(data.events, target).status, "scheduled_start");
    const next = selectWeekend(data.events, target + START_GRACE_MS);
    assert.equal(next.session.code, i === 4 ? "FP1" : brazil.sessions[i + 1].code);
    assert.equal(next.event.id, i === 4 ? vegas.id : brazil.id);
  }
});
test("missing dates and missing race starts stay unavailable; end of calendar is explicit", () => {
  assert.equal(selectWeekend([], Date.now()).status, "unavailable");
  assert.equal(selectWeekend([{ sessions: [{ code: "FP1", startUtc: "invalid" }] }], Date.now()).status, "unavailable");
  assert.equal(selectWeekend([{ sessions: [{ code: "R", startUtc: "2026-11-08T17:00:00" }] }], Date.now()).status, "unavailable");
  assert.equal(selectWeekend(data.events, time("2027-01-01T00:00:00Z")).status, "complete");
});
test("Sprint sessions follow the actual timestamps instead of inventing FP2 and FP3", () => {
  const sprint = data.events.find(event => event.format.includes("sprint"));
  assert.ok(sprint.sessions.some(session => session.code === "SPRINT"));
  const first = time(sprint.sessions[0].startUtc);
  const next = selectWeekend([sprint], first + START_GRACE_MS);
  assert.equal(next.session.code, "SQ");
});
test("lights accumulate with proximity and all go out at zero", () => {
  const seconds = [8*86400, 6*86400, 20*3600, 4*3600, 30*60, 3*60, 0, -1];
  assert.deepEqual(seconds.map(s => lightCount(s*1000)), [0,1,2,3,4,5,0,0]);
  assert.equal(lightCount(NaN), 0);
  assert.deepEqual(countdownParts(1), { days:0, hours:0, minutes:0, seconds:1 });
  assert.deepEqual(countdownParts(-1000), { days:0, hours:0, minutes:0, seconds:0 });
});
test("the four remaining GPs have distinct valid, bounded real circuit paths", () => {
  const remaining = data.events.filter(event => event.round >= brazil.round);
  const paths = remaining.map(event => {
    const outline = projectCircuit(data.circuits[event.circuitId].points);
    assert.ok(outline.coordinates.every(([x,y]) => x>=95 && x<=565.01 && y>=90 && y<=430.01));
    assert.match(outline.path, /^M.* Z$/);
    return outline.path;
  });
  assert.equal(new Set(paths).size, remaining.length);
  assert.equal(projectCircuit([[0,0],[0,0],[0,0],[0,0]]), null);
  assert.equal(projectCircuit([[0,0],[1,1],[2,2],[NaN,3]]), null);
  assert.equal(countryFlag("BR"), "🇧🇷");
  assert.equal(countryFlag("unknown"), "");
});
