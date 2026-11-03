import { selectWeekend, validSessions, countdownParts, lightCount, projectCircuit, countryFlag } from "./race-engine.mjs";
import { CalendarClient, REFRESH_INTERVAL_MS, validateBundle, stableEventId } from "./calendar-source.mjs";

const find = selector => document.querySelector(selector);
const put = (selector, value) => { const node = find(selector); if (node.textContent !== value) node.textContent = value; };
let calendar = null;
let pinnedEvent = null;
let renderedEvent = null;
let renderedSelection = null;
let demo = null;
let calendarClient = null;
let lastRefreshAttempt = 0;
const demoSeconds = [8*86400, 6*86400, 20*3600, 4*3600, 30*60, 3*60, 8, 0];

function localDate(value, withDay = true) {
  return new Intl.DateTimeFormat(undefined, {
    ...(withDay ? { weekday:"short", day:"numeric", month:"short" } : {}),
    hour:"2-digit", minute:"2-digit", timeZoneName:"short",
  }).format(new Date(value));
}

function renderCircuit(event) {
  if (renderedEvent === event.id) return;
  renderedEvent = event.id;
  const circuit = calendar.circuits?.[event.circuitId];
  const outline = projectCircuit(circuit?.points);
  put("#upcoming-event-name", event.name);
  put("#upcoming-circuit-name", circuit?.name ?? event.location);
  put("#circuit-round", `${event.season} / ROUND ${String(event.round).padStart(2, "0")}`);
  put("#circuit-country-word", event.countryCode ?? "AP");
  put("#circuit-flag", countryFlag(event.countryCode));
  find("#circuit-flag").setAttribute("aria-label", event.country);
  put("#circuit-map-note", outline ? "CIRCUIT OUTLINE / ILLUSTRATIVE LAP" : "CIRCUIT OUTLINE UNAVAILABLE");
  find("#circuit-unavailable").hidden = Boolean(outline);
  find(".track-svg").style.visibility = outline ? "visible" : "hidden";
  find(".track-stage").setAttribute("aria-label", `${event.name}: ${circuit?.name ?? event.location}. Circuit geometry with an illustrative animation, not live telemetry.`);
  if (outline) {
    find("#racingPath").setAttribute("d", outline.path);
    document.querySelectorAll(".race-dot").forEach(dot => {
      dot.style.offsetPath = `path('${outline.path}')`;
    });
  }
  const timeline = find("#weekend-sessions");
  timeline.replaceChildren();
  for (const session of validSessions(event)) {
    const item = document.createElement("div");
    item.className = "weekend-session";
    item.dataset.code = session.code;
    const dot = document.createElement("i");
    dot.setAttribute("aria-hidden", "true");
    const label = document.createElement("strong");
    label.textContent = session.code === "R" ? "RACE" : session.code;
    const time = document.createElement("time");
    time.dateTime = session.startUtc;
    time.textContent = new Intl.DateTimeFormat(undefined, { weekday:"short", hour:"2-digit", minute:"2-digit" }).format(new Date(session.startUtc));
    item.title = `${session.name}: ${localDate(session.startUtc)}`;
    item.append(dot, label, time);
    timeline.append(item);
  }
  find("#weekend-sessions").style.setProperty("--session-count", validSessions(event).length);
}

function renderSelection(selection, now) {
  if (!["upcoming", "scheduled_start"].includes(selection.status)) {
    const awaiting = selection.status === "awaiting_schedule";
    if (selection.event) renderCircuit(selection.event);
    put("#next-session-name", awaiting ? "Awaiting confirmed session times" : selection.status === "complete" ? "Next calendar not available yet" : "Session schedule unavailable");
    put("#session-local-time", "Verified session times will appear here when available.");
    put("#weekend-clock-state", "CALENDAR UNAVAILABLE");
    find("#weekend-clock-state").classList.remove("is-demo");
    find("#weekend-clock").setAttribute("aria-label", "Countdown unavailable until session times are confirmed");
    find("#starting-lights").classList.remove("lights-out");
    document.querySelectorAll("[data-countdown-part]").forEach(node => { node.textContent = "—"; });
    document.querySelectorAll(".start-light").forEach(node => node.classList.remove("lit"));
    find("#preview-lights").disabled = true;
    find("#weekend-sessions").replaceChildren();
    if (!selection.event) {
      find("#racingPath").setAttribute("d", "");
      find(".track-svg").style.visibility = "hidden";
      put("#upcoming-event-name", "The next chapter awaits.");
      put("#upcoming-circuit-name", "Calendar unavailable");
      put("#circuit-map-note", "NO FUTURE RACE IN THIS VERIFIED CALENDAR");
      put("#circuit-flag", "");
      renderedEvent=null;
    }
    find("#follow-next-event").hidden = !pinnedEvent;
    return;
  }
  renderCircuit(selection.event);
  const isDemo = Boolean(demo);
  const isStarting = selection.status === "scheduled_start";
  put("#weekend-clock-state", isDemo ? "LIGHTS DEMO · COMPRESSED TIME" : pinnedEvent ? "EXPLORING THE CALENDAR" : "UP NEXT");
  find("#weekend-clock-state").classList.toggle("is-demo", isDemo);
  put("#next-session-name", selection.session.name);
  put("#session-local-time", `${localDate(selection.session.startUtc)} · your time`);
  find("#session-local-time").dateTime = selection.session.startUtc;
  const parts = countdownParts(selection.remainingMs);
  document.querySelectorAll("[data-countdown-part]").forEach(node => {
    const value = String(parts[node.dataset.countdownPart]).padStart(2, "0");
    if (node.textContent !== value) node.textContent = value;
  });
  find("#weekend-clock").setAttribute("aria-label", `${parts.days} days, ${parts.hours} hours, ${parts.minutes} minutes, ${parts.seconds} seconds until ${selection.session.name}`);
  const lights = isStarting ? 0 : lightCount(selection.remainingMs);
  document.querySelectorAll(".start-light").forEach((node, index) => node.classList.toggle("lit", index < lights));
  find("#starting-lights").setAttribute("aria-label", `${lights} of 5 red lights lit. Proximity to the scheduled start.`);
  find("#starting-lights").classList.toggle("lights-out", isStarting);
  put("#lights-caption", isStarting ? "SCHEDULED START / LIGHTS OUT" : "CLOSER TO THE START. ONE LIGHT AT A TIME.");
  const selectionKey = `${selection.event.id}-${selection.session.code}-${selection.session.startUtc}-${selection.status}-${isDemo}`;
  if (selectionKey !== renderedSelection) {
    renderedSelection = selectionKey;
    put("#weekend-announcement", `${isDemo ? "Demonstration. " : ""}${isStarting ? "Scheduled start time reached for" : "Counting down to"} ${selection.session.name}, ${selection.event.name}.`);
  }
  document.querySelectorAll(".weekend-session").forEach(item => {
    const session = selection.sessions.find(s => s.code === item.dataset.code);
    item.classList.toggle("is-next", item.dataset.code === selection.session.code);
    item.classList.toggle("start-passed", session.startMs < now && item.dataset.code !== selection.session.code);
    if (item.dataset.code === selection.session.code) item.setAttribute("aria-current", "step");
    else item.removeAttribute("aria-current");
  });
  find("#follow-next-event").hidden = !pinnedEvent;
  find("#preview-lights").disabled = false;
  put("#preview-lights", isDemo ? "Return to real time ↩" : "Preview the lights ↗");
}

function currentSelection(now) {
  const events = pinnedEvent ? calendar.events.filter(event => event.id === pinnedEvent) : calendar.events;
  return selectWeekend(events, now);
}

function tick() {
  if (!calendar) return;
  const now = Date.now();
  let selection = currentSelection(now);
  if (demo) {
    const phase = Math.floor((now - demo.startedAt) / 1000);
    if (phase >= demoSeconds.length) demo = null;
    else {
      const remainingMs = demoSeconds[phase] * 1000;
      selection = { ...demo.selection, remainingMs, status: remainingMs === 0 ? "scheduled_start" : "upcoming" };
    }
  }
  renderSelection(selection, now);
}

find("#preview-lights").addEventListener("click", () => {
  if (demo) demo = null;
  else {
    const selection = currentSelection(Date.now());
    if (selection.status !== "upcoming") return;
    demo = { startedAt: Date.now(), selection };
  }
  tick();
});
find("#follow-next-event").addEventListener("click", () => {
  pinnedEvent = null; demo = null;
  find("#event-explorer").value = "auto";
  tick();
});
find("#event-explorer").addEventListener("change", event => {
  pinnedEvent = event.target.value === "auto" ? null : event.target.value;
  demo = null; tick();
});

function applyCalendar(loaded) {
  loaded={...loaded,events:loaded.events.map(event=>({...event,id:stableEventId(event)}))};
  calendar = validateBundle(loaded);
  // Same event identity can have new times, venue or geometry after a source update.
  renderedEvent = null;
  renderedSelection = null;
  demo = null;
  const now = Date.now();
  if (pinnedEvent && !loaded.events.some(event=>event.id === pinnedEvent)) pinnedEvent=null;
  const explorer = find("#event-explorer");
  explorer.replaceChildren(new Option("Follow the next Grand Prix", "auto"));
  for (const event of loaded.events) {
    const selection=selectWeekend([event],now);
    if (!["upcoming","scheduled_start","awaiting_schedule"].includes(selection.status)) continue;
    explorer.append(new Option(`${countryFlag(event.countryCode)} ${event.name}`,event.id));
  }
  explorer.value=pinnedEvent ?? "auto";
  tick();
}

function showSourceStatus(status) {
  const checked=status.checkedAt ? new Date(status.checkedAt) : null;
  const date=checked && Number.isFinite(checked.getTime()) ? new Intl.DateTimeFormat(undefined,{day:"numeric",month:"short",hour:"2-digit",minute:"2-digit"}).format(checked) : "unknown";
  const state=status.state;
  const label=state === "fresh" ? `Schedule checked ${date}`
    : state === "geometry_cached" ? `Schedule checked ${date} · cached circuit`
    : state === "next_season_cached" ? `Schedule checked ${date} · next season pending`
    : state === "refreshing" ? "Checking schedule…"
    : state === "offline" ? `Last verified ${date} · refresh unavailable`
    : `Saved schedule · ${date}`;
  put("#calendar-checked",label);
  find("#calendar-checked").dataset.status=state;
  find("#calendar-checked").title=state === "offline" ? "The public source could not be refreshed. This countdown uses the last verified schedule; it may have changed. Retrying automatically."
    : `Public FastF1 calendar; checked every five minutes and when this tab returns. ${status.cacheAvailable===false ? "Browser storage unavailable; the current page still uses the verified update." : ""}`;
  put("#calendar-refresh-notice",state === "offline" ? "Schedule refresh unavailable. Showing the last verified times; checking again automatically." : state === "geometry_cached" ? "Schedule updated. Circuit geometry could not be refreshed; using the saved outline." : state === "next_season_cached" ? "Current-season schedule updated. The next season could not be verified; any future-season dates remain the last verified version." : "");
  find("#calendar-refresh-notice").hidden=!["offline","geometry_cached","next_season_cached"].includes(state);
}

function refreshCalendar() {
  if (!calendarClient || document.hidden) return;
  lastRefreshAttempt=Date.now();
  return calendarClient.refresh();
}

async function loadCalendar() {
  let initial;
  try {
    const response = await fetch("data/weekends.json", { cache:"no-cache" });
    if (!response.ok) throw new Error(`Calendar HTTP ${response.status}`);
    const loaded = await response.json();
    initial=validateBundle(loaded);
  } catch (error) {
    console.warn("Bundled preview calendar unavailable:",error);
  }
  let storage;
  try { storage=window.localStorage; } catch { storage=undefined; }
  calendarClient=new CalendarClient({initial,storage,onUpdate:applyCalendar,onStatus:showSourceStatus});
  calendarClient.restore();
  if (!calendar) renderSelection({status:"unavailable"},Date.now());
  refreshCalendar();
  setInterval(()=>{ if (!document.hidden) tick(); },250);
  setInterval(refreshCalendar,REFRESH_INTERVAL_MS);
  const onReturn=()=>{
    if (document.hidden) return;
    tick();
    if (Date.now()-lastRefreshAttempt>=15_000) refreshCalendar();
  };
  document.addEventListener("visibilitychange",onReturn);
  window.addEventListener("focus",onReturn);
  window.addEventListener("online",refreshCalendar);
  find("#refresh-calendar").addEventListener("click",()=>{
    refreshCalendar();
  });
}
loadCalendar();
