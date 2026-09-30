// nssim dashboard: turns the run's events (see nssim/bus.py) into tables, plots and a message log.
// Everything is keyed on the run's clock (the MCB clock in the sim), shown as seconds since the run
// started. Panels without data (no ground truth on the real robot, say) just stay empty.
"use strict";

const $ = (id) => document.getElementById(id);
const DEG = 180 / Math.PI;
const css = getComputedStyle(document.documentElement);
const C = ["--c1", "--c2", "--c3", "--c4"].map((name) => css.getPropertyValue(name).trim());

const plots = {
  yaw: new TimePlot($("p-yaw"), { unit: "°", wrap: 360, digits: 1, series: [
    { name: "turret", color: C[0] }, { name: "CV aim", color: C[1] }, { name: "robot", color: C[2] }] }),
  spin: new TimePlot($("p-spin"), { unit: "rad/s", zero: true, series: [
    { name: "true", color: C[2] }, { name: "filter", color: C[0] }] }),
  center: new TimePlot($("p-center"), { unit: "cm", zero: true, digits: 1, series: [
    { name: "horizontal", color: C[0] }, { name: "height", color: C[1] }] }),
  radius: new TimePlot($("p-radius"), { unit: "cm", zero: true, digits: 1, series: [
    { name: "radius", color: C[0] }] }),
  orient: new TimePlot($("p-orient"), { unit: "°", zero: true, digits: 1, series: [
    { name: "orientation (mod 180°)", color: C[3] }] }),
  plates: new TimePlot($("p-plates"), { zero: true, digits: 0, series: [
    { name: "visible", color: C[2] }, { name: "detected", color: C[0] }] }),
  latency: new TimePlot($("p-latency"), { unit: "ms", zero: true, digits: 1, series: [
    { name: "arrival", color: C[2] }, { name: "CV processing", color: C[0] }, { name: "aim at MCB", color: C[1] }] }),
};

const LOG_KEEP = 600;
const LOG_SHOWN = 250;
const LOG_OFF_BY_DEFAULT = new Set(["ODOMETRY"]); // 250 Hz; switch it on to see it

let state;
let logFilters = new Map(); // message type -> shown (kept across runs)

function reset() {
  state = {
    t0: null,
    tNow: 0,
    run: null,
    ended: false,
    traffic: { mcb_to_cv: new Map(), cv_to_mcb: new Map() },
    log: [],
    frames: [], // {t, wall} of recent frames, for the frame rate and real-time factor
    lastFrame: null,
    lastMetrics: null,
    lastTruth: null,
    lastTrack: null,
    lastAim: null,
    lastOdomPlot: -Infinity,
    tracked: null,
  };
  for (const plot of Object.values(plots)) plot.clear();
  $("run-name").textContent = "";
}

let paused = false;
let windowS = 30;

// --- events -------------------------------------------------------------------------------

function ingest(event) {
  const { topic, data } = event;
  if (topic === "end") {
    state.ended = true;
    setConnection("ended", "run ended");
    return;
  }
  if (topic === "run") {
    state.run = data;
    state.t0 = event.t;
    $("run-name").textContent = data.name;
    return;
  }
  if (state.t0 === null) state.t0 = event.t;
  const t = event.t - state.t0;
  if (t > state.tNow) state.tNow = t;

  if (topic === "uart") onUart(t, data);
  else if (topic === "frame") onFrame(t, data, event.wall);
  else if (topic === "truth") onTruth(t, data);
  else if (topic === "metrics") onMetrics(t, data);
  else if (topic === "cv.track") state.lastTrack = data;
}

function onUart(t, msg) {
  const table = state.traffic[msg.dir];
  if (!table) return;
  let row = table.get(msg.type);
  if (!row) table.set(msg.type, (row = { count: 0, times: [], fields: null }));
  row.count++;
  row.times.push(t);
  if (row.times.length > 4000) row.times.splice(0, 2000);
  row.fields = msg.fields;

  if (msg.type === "ODOMETRY" && t - state.lastOdomPlot >= 0.02) {
    plots.yaw.push(0, t, wrapDeg(msg.fields.yaw * DEG));
    state.lastOdomPlot = t;
  } else if (msg.type === "TURRET_AIM_DATA") {
    const hasTarget = !(msg.fields.yaw === 0 && msg.fields.pitch === 0); // (0, 0) means no target
    plots.yaw.push(1, t, hasTarget ? wrapDeg(msg.fields.yaw * DEG) : null);
    state.lastAim = hasTarget ? msg.fields : null;
  }

  if (!logFilters.has(msg.type)) {
    logFilters.set(msg.type, !LOG_OFF_BY_DEFAULT.has(msg.type));
    renderLogFilters();
  }
  state.log.push({ t, msg });
  if (state.log.length > LOG_KEEP * 2) state.log.splice(0, state.log.length - LOG_KEEP);
}

function onFrame(t, frame, wall) {
  state.lastFrame = frame;
  state.frames.push({ t, wall });
  if (state.frames.length > 2000) state.frames.splice(0, 1000);
  plots.latency.push(0, t, frame.arrival_ms);
  plots.latency.push(1, t, frame.processing_ms);
  plots.latency.push(2, t, frame.aim_ms);
}

function onTruth(t, truth) {
  state.lastTruth = truth;
  const target = truth.targets.find((tg) => tg.name === state.tracked) || truth.targets[0];
  if (target) plots.yaw.push(2, t, wrapDeg(target.bearing * DEG));
}

function onMetrics(t, m) {
  state.lastMetrics = m;
  plots.plates.push(0, t, m.visible);
  plots.plates.push(1, t, m.detected);
  const f = m.filter;
  if (f) state.tracked = f.target;
  plots.spin.push(0, t, f ? f.true_omega : null);
  plots.spin.push(1, t, f ? f.omega : null);
  plots.center.push(0, t, f ? f.center_xy_m * 100 : null);
  plots.center.push(1, t, f ? f.center_z_m * 100 : null);
  plots.radius.push(0, t, f ? f.radius_m * 100 : null);
  plots.orient.push(0, t, f ? f.orientation_rad * DEG : null);
}

// --- rendering ----------------------------------------------------------------------------

function render() {
  renderHeader();
  if (paused) return;
  for (const plot of Object.values(plots)) plot.draw(state.tNow, windowS);
  renderTraffic("mcb_to_cv");
  renderTraffic("cv_to_mcb");
  renderTracking();
  renderLog();
}

function renderHeader() {
  $("s-time").textContent = state.t0 === null ? "–" : `${state.tNow.toFixed(1)} s`;
  const frames = state.frames;
  const recent = frames.filter((f) => f.t > state.tNow - 1);
  $("s-fps").textContent = recent.length ? `${recent.length} fps` : "–";
  // Real-time factor: sim seconds per wall second over the last couple of wall seconds.
  const lastWall = frames.length ? frames[frames.length - 1].wall : 0;
  const lately = frames.filter((f) => f.wall > lastWall - 2);
  if (lately.length > 2) {
    const a = lately[0], b = lately[lately.length - 1];
    $("s-rtf").textContent = b.wall > a.wall ? `×${((b.t - a.t) / (b.wall - a.wall)).toFixed(2)}` : "–";
  } else {
    $("s-rtf").textContent = "–";
  }
  const f = state.lastFrame;
  $("s-cv").textContent = f ? `${f.processing_ms.toFixed(1)} ms` : "–";
  $("s-aim").textContent = f && f.aim_ms != null ? `${f.aim_ms.toFixed(1)} ms` : "–";
}

function renderTraffic(dir) {
  const table = state.traffic[dir];
  const body = $(`t-${dir}`);
  if (!table.size) {
    body.innerHTML = `<tr><td colspan="4" class="empty">nothing yet</td></tr>`;
    return;
  }
  const rows = [...table.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([type, row]) => {
    const rate = row.times.filter((t) => t > state.tNow - 1).length;
    return `<tr><td>${type}</td><td class="num">${rate} Hz</td><td class="num">${row.count}</td>` +
      `<td class="fields">${escapeHtml(formatFields(row.fields))}</td></tr>`;
  });
  body.innerHTML = rows.join("");
}

function renderTracking() {
  const items = [];
  const m = state.lastMetrics, f = m && m.filter, track = state.lastTrack, aim = state.lastAim;
  if (state.run) items.push(["enemies", state.run.targets.map((tg) => `${tg.name} (${tg.number})`).join(", ")]);
  if (f) {
    items.push(["tracking", f.target]);
    items.push(["spin", `${f.omega.toFixed(2)} rad/s  (true ${f.true_omega.toFixed(2)})`]);
    items.push(["radius", `${f.radius.toFixed(3)} m  (true ${f.true_radius.toFixed(3)})`]);
    items.push(["center off by", `${(f.center_xy_m * 100).toFixed(1)} cm`]);
    items.push(["orientation off by", `${(f.orientation_rad * DEG).toFixed(1)}°`]);
  } else if (track) {
    items.push(["tracking", track.pf_alive ? "yes" : "no"]);
  } else {
    items.push(["tracking", "no"]);
  }
  if (m) items.push(["plates", `${m.detected} detected of ${m.visible} visible`]);
  if (aim) items.push(["aim", `yaw ${(aim.yaw * DEG).toFixed(1)}°  pitch ${(aim.pitch * DEG).toFixed(1)}°  ${aim.distance.toFixed(2)} m  id ${aim.target_id}`]);
  if (track && track.pipeline_delay_s != null) items.push(["pipeline delay", `${(track.pipeline_delay_s * 1000).toFixed(1)} ms`]);
  $("tracking").innerHTML = items.map(([k, v]) => `<dt>${k}</dt><dd>${escapeHtml(v)}</dd>`).join("");
}

function renderLogFilters() {
  const box = $("log-filters");
  box.innerHTML = "";
  for (const [type, shown] of [...logFilters.entries()].sort(([a], [b]) => a.localeCompare(b))) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = shown;
    input.addEventListener("change", () => logFilters.set(type, input.checked));
    label.append(input, type);
    box.append(label);
  }
}

function renderLog() {
  const box = $("log");
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 20;
  const lines = [];
  for (let i = state.log.length - 1; i >= 0 && lines.length < LOG_SHOWN; i--) {
    const { t, msg } = state.log[i];
    if (!logFilters.get(msg.type)) continue;
    const toCv = msg.dir === "mcb_to_cv";
    const arrow = toCv ? "MCB → CV" : "CV → MCB";
    lines.push(`<span class="${toCv ? "to-cv" : "to-mcb"}">${t.toFixed(3).padStart(9)}  ${arrow}  ` +
      `${msg.type.padEnd(16)}</span>${escapeHtml(formatFields(msg.fields))}`);
  }
  box.innerHTML = lines.reverse().join("\n") || `<span class="empty">no messages</span>`;
  if (atBottom) box.scrollTop = box.scrollHeight;
}

function formatFields(fields) {
  if (!fields) return "";
  return Object.entries(fields).map(([k, v]) => `${k} ${typeof v === "number" ? formatNumber(v) : v}`).join("  ");
}

function formatNumber(v) {
  return Number.isInteger(v) ? String(v) : v.toFixed(3);
}

function wrapDeg(v) {
  return ((((v + 180) % 360) + 360) % 360) - 180;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);
}

function setConnection(kind, text) {
  const pill = $("conn");
  pill.className = `pill ${kind}`;
  pill.textContent = text;
}

// --- connection ---------------------------------------------------------------------------

function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => setConnection("live", "live");
  ws.onmessage = (message) => {
    const packet = JSON.parse(message.data);
    if (packet.kind === "snapshot") reset();
    for (const event of packet.events) ingest(event);
  };
  ws.onclose = () => {
    if (!state.ended) setConnection("down", "waiting for a run…");
    setTimeout(connect, 1000); // the next run (on the same port) picks up where this one left off
  };
}

$("pause").addEventListener("click", (e) => {
  paused = !paused;
  e.target.setAttribute("aria-pressed", String(paused));
  e.target.textContent = paused ? "Resume" : "Pause";
});
$("window").addEventListener("change", (e) => (windowS = Number(e.target.value)));

reset();
connect();
setInterval(render, 150);
