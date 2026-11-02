"use strict";

// This isolated prototype reads a sanitized historical export, never the production API.
const teamAssets = {
  ferrari: ["ferrari.svg.webp", "#ee4145"], mclaren: ["mclaren.svg", "#ff962d"],
  mercedes: ["mercedes.svg", "#65cfc1"], red_bull: ["redbull.png", "#5385df"],
  audi: ["audi.webp", "#d8485c"], racing_bulls: ["racingbulls.png", "#83a1ff"],
  alpine: ["alpine.png", "#52a4e9"], haas: ["haas.png", "#b3b6bd"],
  aston_martin: ["astonmartin.png", "#7ab6a1"], cadillac: ["cadillac.png", "#b9b8b1"],
  williams: ["williams.svg", "#628cff"]
};
const driverNames = {
  HAM: ["Lewis", "Hamilton"], LEC: ["Charles", "Leclerc"], NOR: ["Lando", "Norris"],
  ANT: ["Kimi", "Antonelli"], PIA: ["Oscar", "Piastri"], RUS: ["George", "Russell"],
  VER: ["Max", "Verstappen"], HAD: ["Isack", "Hadjar"], HUL: ["Nico", "Hülkenberg"],
  LAW: ["Liam", "Lawson"], SAI: ["Carlos", "Sainz"], ALO: ["Fernando", "Alonso"],
  ALB: ["Alex", "Albon"], STR: ["Lance", "Stroll"], GAS: ["Pierre", "Gasly"],
  OCO: ["Esteban", "Ocon"], BEA: ["Oliver", "Bearman"], BOT: ["Valtteri", "Bottas"],
  PER: ["Sergio", "Pérez"], COL: ["Franco", "Colapinto"], BOR: ["Gabriel", "Bortoleto"],
  LIN: ["Arvid", "Lindblad"]
};
const state = { rows: [], selected: "HAM", expanded: false, view: "forecast", query: "" };
const $ = (selector) => document.querySelector(selector);
const gap = (value) => Number.isFinite(value) ? `+${value.toFixed(3)}s` : "Unavailable";
const text = (selector, value) => { $(selector).textContent = value; };
const position = (value) => Number.isFinite(value) ? `P${String(value).padStart(2, "0")}` : "—";
const cell = (className, content) => {
  const td = document.createElement("td");
  td.className = className;
  if (content !== undefined) td.textContent = content;
  return td;
};

function renderRanking() {
  const focusedDriver = document.activeElement?.closest?.(".driver-button")?.getAttribute("aria-label");
  const matches = state.rows.filter((row) => {
    const name = driverNames[row.driver_code]?.join(" ") ?? row.driver_code;
    return `${row.driver_code} ${row.team} ${name}`.toLowerCase().includes(state.query);
  });
  const visible = state.expanded || state.query ? matches : matches.slice(0, 7);
  const body = $("#ranking-body");
  body.replaceChildren();
  for (const row of visible) {
    const tr = document.createElement("tr");
    tr.classList.toggle("selected", row.driver_code === state.selected);
    const team = teamAssets[row.team_key];
    tr.style.setProperty("--team-color", team?.[1] ?? "#a1a1a1");
    tr.append(cell("rank-cell", String(row.predicted_position).padStart(2, "0")));
    const driverCell = cell("driver-cell");
    const button = document.createElement("button");
    button.className = "driver-button";
    button.setAttribute("aria-label", `Inspect ${row.driver_code}, ${row.team}`);
    button.setAttribute("aria-pressed", String(row.driver_code === state.selected));
    const avatar = document.createElement("span");
    avatar.className = "team-avatar";
    avatar.dataset.team = row.team_key;
    if (team) {
      const img = document.createElement("img");
      img.src = `assets/teams/${team[0]}`;
      img.alt = "";
      img.width = 32;
      img.height = 32;
      img.decoding = "async";
      img.addEventListener("error", () => { avatar.textContent = row.team.slice(0, 2).toUpperCase(); });
      avatar.append(img);
    } else avatar.textContent = row.team.slice(0, 2).toUpperCase();
    const labels = document.createElement("span");
    const code = document.createElement("strong");
    code.textContent = row.driver_code;
    const name = document.createElement("small");
    name.textContent = row.team;
    labels.append(code, name);
    button.append(avatar, labels);
    button.addEventListener("click", () => selectDriver(row.driver_code));
    driverCell.append(button);
    tr.append(driverCell, cell(`row-gap${row.predicted_position === 1 ? " pole" : ""}`, gap(row.predicted_gap_to_pole_sec)));
    if (state.view === "forecast") {
      const spread = cell("spread-cell");
      const bar = document.createElement("span");
      bar.className = "spread-bar";
      bar.setAttribute("aria-hidden", "true");
      const fill = document.createElement("i");
      const max = Math.max(...state.rows.map((r) => r.predicted_gap_to_pole_sec), 0.001);
      fill.style.width = `${Math.max(4, row.predicted_gap_to_pole_sec / max * 100)}%`;
      bar.append(fill);
      spread.append(bar);
      tr.append(spread);
    } else {
      const result = cell("result-cell");
      const actual = document.createElement("strong");
      actual.textContent = position(row.actual_position);
      const delta = document.createElement("span");
      const difference = row.predicted_position - row.actual_position;
      delta.textContent = difference === 0 ? " =" : ` ${difference > 0 ? "↑" : "↓"}${Math.abs(difference)}`;
      if (difference === 0) delta.className = "exact";
      delta.setAttribute("aria-label", difference === 0 ? "Exact position" : `Official result ${Math.abs(difference)} positions ${difference > 0 ? "higher" : "lower"} than predicted`);
      result.append(actual, delta);
      tr.append(result);
    }
    body.append(tr);
  }
  if (visible.length === 0) {
    const tr = document.createElement("tr");
    const empty = cell("no-results", "No drivers match. Try a driver code, surname or team.");
    empty.colSpan = 4;
    tr.append(empty);
    body.append(tr);
  }
  text("#driver-count", `${visible.length} / ${state.rows.length} DRIVERS · AFTER FP3`);
  $("#show-all").hidden = Boolean(state.query);
  $("#show-all").innerHTML = state.expanded ? 'Show top 7 <span>↑</span>' : 'Show full field <span>↓</span>';
  text("#last-heading", state.view === "forecast" ? "PACE SPREAD" : "OFFICIAL / Δ");
  if (focusedDriver) {
    Array.from(body.querySelectorAll(".driver-button"))
      .find((button) => button.getAttribute("aria-label") === focusedDriver)?.focus({ preventScroll: true });
  }
}

function selectDriver(code) {
  const row = state.rows.find((r) => r.driver_code === code);
  if (!row) return;
  state.selected = code;
  const names = driverNames[code] ?? ["Driver", code];
  const color = teamAssets[row.team_key]?.[1] ?? "#aaaaaa";
  $("#driver-detail").style.setProperty("--team-color", color);
  $("#driver-detail").style.borderTopColor = color;
  text("#detail-position", position(row.predicted_position));
  text("#detail-code", code);
  text("#detail-team", row.team.toUpperCase());
  text("#detail-name", names[1]);
  text("#detail-firstname", `${names[0]} / ${code}`);
  $("#detail-gap").replaceChildren(document.createTextNode(`+${row.predicted_gap_to_pole_sec.toFixed(3)}`));
  const unit = document.createElement("span");
  unit.textContent = "s";
  $("#detail-gap").append(unit);
  text("#detail-actual", position(row.actual_position));
  text("#detail-model-label", gap(row.predicted_gap_to_pole_sec));
  text("#detail-actual-label", gap(row.actual_gap_to_pole_sec));
  const scale = Math.max(row.predicted_gap_to_pole_sec, row.actual_gap_to_pole_sec, 0.001) * 1.15;
  $("#model-bar").style.width = `${row.predicted_gap_to_pole_sec / scale * 100}%`;
  $("#actual-bar").style.width = `${Math.max(1, row.actual_gap_to_pole_sec / scale * 100)}%`;
  const delta = row.predicted_position - row.actual_position;
  text("#detail-insight", delta === 0
    ? `The model and official result agree on ${position(row.actual_position)} for ${names[1]}.`
    : `The model placed ${names[1]} ${position(row.predicted_position)}. The official result was ${position(row.actual_position)}, ${Math.abs(delta)} ${Math.abs(delta) === 1 ? "place" : "places"} ${delta > 0 ? "higher" : "lower"}.`);
  renderRanking();
}

function setView(view) {
  state.view = view;
  document.querySelectorAll("[data-view]").forEach((button) => {
    const active = button.dataset.view === view;
    button.classList.toggle("selected", active);
    button.setAttribute("aria-pressed", String(active));
  });
  renderRanking();
}

document.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
$("#driver-search").addEventListener("input", (event) => { state.query = event.target.value.trim().toLowerCase(); renderRanking(); });
$("#show-all").addEventListener("click", () => { state.expanded = !state.expanded; renderRanking(); });

const checkpointMessages = {
  FP1: "FP1 practice data is available in the archive. No after-FP1 forecast is included; the table keeps the preserved after-FP3 estimate.",
  FP2: "FP2 practice data is available in the archive. No after-FP2 forecast is included; the table keeps the preserved after-FP3 estimate.",
  FP3: "Showing the preserved after-FP3 prediction. Earlier checkpoint forecasts are not included in this snapshot.",
  Q: "Qualifying is complete. Compare the preserved after-FP3 prediction with the official result below."
};
document.querySelectorAll("[data-session]").forEach((button) => button.addEventListener("click", () => {
  document.querySelectorAll("[data-session]").forEach((b) => {
    b.classList.toggle("selected", b === button);
    b.setAttribute("aria-pressed", String(b === button));
  });
  text("#checkpoint-notice", checkpointMessages[button.dataset.session]);
  setView(button.dataset.session === "Q" ? "comparison" : "forecast");
}));

$("#theme-toggle").addEventListener("click", () => {
  const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = theme;
  $("#theme-toggle").setAttribute("aria-label", `Switch to ${theme === "dark" ? "light" : "dark"} mode`);
  $("#theme-toggle").innerHTML = theme === "light"
    ? '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M20 15.5A8.5 8.5 0 0 1 8.5 4 8.5 8.5 0 1 0 20 15.5Z"/></svg>'
    : '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.4 1.4m11.2 11.2L19 19M5 19l1.4-1.4M17.6 6.4 19 5"/></svg>';
  $("meta[name=theme-color]").content = theme === "dark" ? "#101112" : "#f5f3ee";
});

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
let paused = reducedMotion.matches;
let userMotionPreference = false;
function updateMotion() {
  document.documentElement.classList.toggle("motion-paused", paused);
  $("#motion-toggle").setAttribute("aria-pressed", String(paused));
  $("#motion-toggle").setAttribute("aria-label", paused ? "Resume animations" : "Pause animations");
  $("#motion-toggle").title = paused ? "Resume animations" : "Pause animations";
  $("#motion-toggle").innerHTML = paused
    ? '<svg width="16" height="16" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6"><path d="m7 4 9 6-9 6Z"/></svg>'
    : '<svg width="16" height="16" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M7 4v12M13 4v12"/></svg>';
}
updateMotion();
$("#motion-toggle").addEventListener("click", () => { userMotionPreference = true; paused = !paused; updateMotion(); });
reducedMotion.addEventListener("change", (event) => { if (!userMotionPreference) { paused = event.matches; updateMotion(); } });

// Native pointer remains visible. A small follower and brief sparks evoke a racing line.
const canvas = $("#cursor-trail");
const context = canvas.getContext("2d");
const halo = $(".cursor-halo");
const coarse = window.matchMedia("(pointer: coarse)");
let pointer = { x: -100, y: -100 }, follower = { x: -100, y: -100 }, particles = [], previous = null, animationFrame = null;
function sizeCanvas() {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = window.innerWidth * ratio;
  canvas.height = window.innerHeight * ratio;
  context?.setTransform(ratio, 0, 0, ratio, 0, 0);
}
sizeCanvas();
window.addEventListener("resize", sizeCanvas, { passive: true });
function animatePointer() {
  animationFrame = null;
  if (!context) return;
  context.clearRect(0, 0, window.innerWidth, window.innerHeight);
  if (paused || coarse.matches || document.hidden) { particles = []; return; }
  follower.x += (pointer.x - follower.x) * .2;
  follower.y += (pointer.y - follower.y) * .2;
  const offset = halo.classList.contains("over-control") ? 21 : 12.5;
  halo.style.transform = `translate(${follower.x - offset}px,${follower.y - offset}px)`;
  for (const p of particles) {
    p.life -= .055;
    p.x += p.vx;
    p.y += p.vy;
    context.strokeStyle = `rgba(255,82,63,${Math.max(0, p.life * .5)})`;
    context.lineWidth = 1;
    context.beginPath(); context.moveTo(p.x, p.y); context.lineTo(p.x - p.vx * 3, p.y - p.vy * 3); context.stroke();
  }
  particles = particles.filter((p) => p.life > 0);
  if (particles.length || Math.abs(pointer.x - follower.x) + Math.abs(pointer.y - follower.y) > .5) animationFrame = requestAnimationFrame(animatePointer);
}
window.addEventListener("pointermove", (event) => {
  if (paused || coarse.matches || event.pointerType === "touch") return;
  pointer = { x: event.clientX, y: event.clientY };
  if (follower.x === -100) follower = { ...pointer };
  halo.style.opacity = "1";
  halo.classList.toggle("over-control", Boolean(event.target.closest("a,button,input,summary")));
  if (previous) {
    const dx = pointer.x - previous.x, dy = pointer.y - previous.y;
    if (Math.hypot(dx, dy) > 8) {
      for (let i = 0; i < 2; i++) particles.push({ x:pointer.x, y:pointer.y, vx:-dx*.025+(Math.random()-.5)*1.2, vy:-dy*.025+(Math.random()-.5)*1.2, life:.7 });
      particles = particles.slice(-35);
    }
  }
  previous = { ...pointer };
  if (animationFrame === null) animationFrame = requestAnimationFrame(animatePointer);
}, { passive: true });
document.documentElement.addEventListener("pointerleave", () => { halo.style.opacity = "0"; previous = null; });
const stage = $(".track-stage");
stage.addEventListener("pointermove", (event) => {
  if (paused || coarse.matches) return;
  const bounds = stage.getBoundingClientRect();
  stage.style.setProperty("--tilt-x", `${-(event.clientY - bounds.top - bounds.height / 2) / 60}deg`);
  stage.style.setProperty("--tilt-y", `${(event.clientX - bounds.left - bounds.width / 2) / 60}deg`);
});
stage.addEventListener("pointerleave", () => { stage.style.setProperty("--tilt-x", "0deg"); stage.style.setProperty("--tilt-y", "0deg"); });

let scrollTicking = false;
window.addEventListener("scroll", () => {
  if (scrollTicking) return;
  scrollTicking = true;
  requestAnimationFrame(() => {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    document.documentElement.style.setProperty("--scroll", `${max > 0 ? window.scrollY / max * 100 : 0}%`);
    scrollTicking = false;
  });
}, { passive: true });
const sectionObserver = new IntersectionObserver((entries) => {
  for (const entry of entries) if (entry.isIntersecting) {
    document.querySelectorAll(".nav-link").forEach((a) => a.classList.toggle("active", a.getAttribute("href") === `#${entry.target.id}`));
  }
}, { rootMargin: "-20% 0px -60% 0px" });
document.querySelectorAll("#forecast,#history,#methodology").forEach((section) => sectionObserver.observe(section));

async function loadSnapshot() {
  try {
    const response = await fetch("snapshot.json");
    if (!response.ok) throw new Error(`Snapshot unavailable (${response.status})`);
    const data = await response.json();
    state.rows = data.rows.sort((a, b) => a.predicted_position - b.predicted_position);
    text("#event-name", data.event.event);
    text("#event-place", `${data.schedule.circuit} · ${data.schedule.location}`);
    text("#snapshot-date", `Snapshot: ${new Date(data.generatedAt).toLocaleDateString("en-GB", { day:"numeric", month:"short", year:"numeric", timeZone:"UTC" })}`);
    $("#metric-mae").innerHTML = `${data.metrics.mae_gap_sec.toFixed(3)}<span>s</span>`;
    text("#metric-position", data.metrics.mean_absolute_position_error.toFixed(2));
    $("#metric-top5").innerHTML = `${Math.round(data.metrics.top_5_agreement * 100)}<span>%</span>`;
    $("#metric-top10").innerHTML = `${Math.round(data.metrics.top_10_agreement * 100)}<span>%</span>`;
    for (const event of data.history) {
      const item = document.createElement("div"); item.className = "archive-item";
      const copy = document.createElement("div");
      const title = document.createElement("h4"); title.textContent = event.event.event;
      const description = document.createElement("p"); description.textContent = `${event.event.season} · Preserved after-FP3 prediction`;
      const status = document.createElement("span");
      const isCurrent = event.event.event_slug === data.event.event_slug;
      status.textContent = isCurrent ? "DISPLAYED ABOVE / PARTIAL COVERAGE" : "ARCHIVE RECORD / RESULT NOT INCLUDED";
      copy.append(title, description); item.append(copy, status); $("#archive-events").append(item);
    }
    selectDriver(state.rows[0]?.driver_code ?? "HAM");
  } catch (error) {
    console.error("Unable to load preview snapshot:", error);
    $("#ranking-body").replaceChildren();
    const tr = document.createElement("tr"), td = cell("no-results", "The historical snapshot could not be loaded. Serve this preview over HTTP and reload.");
    td.colSpan = 4; tr.append(td); $("#ranking-body").append(tr);
    text("#checkpoint-notice", "Preview data unavailable. No substitute predictions have been generated.");
    $("#driver-detail").hidden = true;
    $("#show-all").hidden = true;
  }
}
loadSnapshot();
