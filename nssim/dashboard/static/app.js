// nssim dashboard: tabs of graphs over any signal of the run, a shared time range (live, paused,
// zoomed), and on the Overview tab the serial traffic, tracking values and message log.
// Times are seconds since the run started. Panels without data (no ground truth on the real robot,
// say) just stay empty.
"use strict";

const $ = (id) => document.getElementById(id);
const DEG = 180 / Math.PI;
const LOG_KEEP = 600;
const LOG_SHOWN = 250;
const LOG_OFF_BY_DEFAULT = new Set(["ODOMETRY"]); // 250 Hz; tick it to see it

const hub = new SignalHub();
const layout = new Layout();
let state;
let logFilters = new Map(); // message type -> shown (kept across runs)
let graphViews = []; // the active tab's graphs
let ws = null;
let subscribed = new Set();
let subscribedKey = "";
let catalogVersion = -1;

// --- the time range every graph shows ----------------------------------------------------

const view = {
  live: true,
  windowS: 30,
  t0: 0,
  t1: 30,
  range() {
    if (!this.live) return [this.t0, this.t1];
    const start = Math.max(0, state.tNow - this.windowS); // fill from the run's start until it's a window long
    return [start, start + this.windowS];
  },
  set(t0, t1) {
    if (!(t1 > t0)) return;
    this.live = false;
    this.t0 = t0;
    this.t1 = Math.max(t1, t0 + 0.001);
    updateTimeControls();
    requestDraw();
  },
  pause() {
    const [a, b] = this.range();
    this.set(a, b);
  },
  goLive() {
    this.live = true;
    updateTimeControls();
    requestDraw();
  },
  handlers: {
    zoom: (a, b) => view.set(a, b),
    pan: (dt) => { const [a, b] = view.range(); view.set(a + dt, b + dt); },
    scale: (tc, f) => { const [a, b] = view.range(); view.set(tc - (tc - a) * f, tc + (b - tc) * f); },
    live: () => view.goLive(),
  },
};

const app = { hub, layout, view, requestDraw, graphsChanged, removeGraph };

// --- panel state ---------------------------------------------------------------------------

function resetState() {
  state = {
    tNow: 0,
    run: null,
    ended: false,
    traffic: { mcb_to_cv: new Map(), cv_to_mcb: new Map() },
    log: [],
    frames: [], // {t, wall} of recent frames, for the frame rate and real-time factor
    lastFrame: null,
    lastMetrics: null,
    lastTrack: null,
    lastAim: null,
  };
  $("run-name").textContent = "";
}

function ingest(event) {
  const { topic, data } = event;
  if (topic === "end") {
    state.ended = true;
    setConnection("ended", "run ended");
    return;
  }
  if (topic === "run") {
    state.run = data;
    $("run-name").textContent = data.replay ? `${data.name} (replay)` : data.name;
    return;
  }
  const t = event.t - hub.offset;
  if (t > state.tNow) state.tNow = t;
  if (topic === "uart") onUart(t, data);
  else if (topic === "frame") {
    state.lastFrame = data;
    state.frames.push({ t, wall: event.wall });
    if (state.frames.length > 2000) state.frames.splice(0, 1000);
  } else if (topic === "metrics") state.lastMetrics = data;
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
  if (msg.type === "TURRET_AIM_DATA") {
    state.lastAim = msg.fields.yaw === 0 && msg.fields.pitch === 0 ? null : msg.fields; // (0, 0): no target
  }
  if (!logFilters.has(msg.type)) {
    logFilters.set(msg.type, !LOG_OFF_BY_DEFAULT.has(msg.type));
    renderLogFilters();
  }
  state.log.push({ t, msg });
  if (state.log.length > LOG_KEEP * 2) state.log.splice(0, state.log.length - LOG_KEEP);
}

// --- tabs and graphs -------------------------------------------------------------------------

function renderTabs() {
  const nav = $("tabs");
  nav.innerHTML = "";
  for (const tab of layout.tabs) {
    const active = tab.id === layout.data.active;
    const button = document.createElement("button");
    button.className = `tab${active ? " active" : ""}`;
    button.textContent = tab.name;
    button.title = active ? "Double-click to rename" : "";
    button.onclick = () => {
      if (!active) {
        layout.setActive(tab.id);
        renderTabs();
        buildTab();
      }
    };
    button.ondblclick = () => renameTab(tab);
    nav.append(button);
    if (active && !tab.builtin) {
      const remove = document.createElement("button");
      remove.className = "tab-tool";
      remove.title = `Delete tab "${tab.name}"`;
      remove.textContent = "✕";
      remove.onclick = () => {
        if (!confirm(`Delete the tab "${tab.name}" and its graphs?`)) return;
        layout.removeTab(tab.id);
        renderTabs();
        buildTab();
        graphsChanged();
      };
      nav.append(remove);
    }
  }
  const add = document.createElement("button");
  add.className = "tab add-tab";
  add.textContent = "+ New tab";
  add.onclick = () => {
    layout.addTab();
    renderTabs();
    buildTab();
  };
  nav.append(add);
}

function renameTab(tab) {
  const name = prompt("Tab name", tab.name);
  if (name && name.trim()) {
    tab.name = name.trim();
    layout.save();
    renderTabs();
  }
}

function buildTab() {
  const tab = layout.active;
  document.body.classList.toggle("custom-tab", !tab.builtin);
  const slot = $("graphs");
  slot.innerHTML = "";
  graphViews = tab.graphs.map((graph) => {
    const gv = new GraphView(graph, tab, app);
    slot.append(gv.el);
    return gv;
  });
  requestDraw();
}

function removeGraph(gv) {
  layout.removeGraph(gv.tab, gv.graph);
  buildTab();
  graphsChanged();
}

$("add-graph").onclick = () => {
  const graph = layout.addGraph(layout.active);
  buildTab();
  const gv = graphViews.find((v) => v.graph === graph);
  gv.toggleEditor();
  gv.el.scrollIntoView({ block: "nearest", behavior: "smooth" });
  gv.el.querySelector(".sig-add input").focus();
};

function graphsChanged() {
  updateSubscription();
  requestDraw();
}

// Stream every signal any tab graphs (so switching tabs is instant), and fetch what came before.
function updateSubscription() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const names = [...new Set(layout.allGraphs().flatMap((g) => g.signals.flatMap((s) => hub.expand(s.name))))].sort();
  const key = names.join("|");
  if (key === subscribedKey) return;
  const added = names.filter((n) => !subscribed.has(n));
  subscribedKey = key;
  subscribed = new Set(names);
  ws.send(JSON.stringify({ kind: "subscribe", signals: names }));
  if (added.length && state.tNow > 0) {
    hub.backfill(added, Math.max(0, state.tNow - hub.keepS), state.tNow + 1).then(requestDraw).catch(() => {});
  }
}

// --- drawing -----------------------------------------------------------------------------------

let drawQueued = false;
function requestDraw() {
  if (drawQueued) return;
  drawQueued = true;
  requestAnimationFrame(() => {
    drawQueued = false;
    drawGraphs();
  });
}

function drawGraphs() {
  const [t0, t1] = view.range();
  // Live with a window the buffers hold: draw them. Otherwise ask the server for the range (a long
  // live window is refetched every two seconds).
  const fromBuffers = view.live && view.windowS <= hub.keepS;
  const key = view.live ? `live:${view.windowS}:${Math.floor(state.tNow / 2)}` : `${t0.toFixed(4)}:${t1.toFixed(4)}`;
  for (const gv of graphViews) {
    if (!fromBuffers) gv.ensureFetched(t0, t1, key);
    gv.draw(t0, t1, fromBuffers);
  }
}

function render() {
  renderHeader();
  if (hub.version !== catalogVersion) {
    catalogVersion = hub.version;
    $("signal-list").innerHTML = hub.names.map((n) => `<option value="${n}"></option>`).join("");
  }
  if (layout.active && layout.active.builtin) {
    renderTraffic("mcb_to_cv");
    renderTraffic("cv_to_mcb");
    renderTracking();
    renderLog();
  }
  drawGraphs();
}

function renderHeader() {
  $("s-time").textContent = `${state.tNow.toFixed(1)} s`;
  const frames = state.frames;
  const recent = frames.filter((f) => f.t > state.tNow - 1);
  $("s-fps").textContent = recent.length && !state.run?.replay ? `${recent.length} fps` : "–";
  const lastWall = frames.length ? frames[frames.length - 1].wall : 0;
  const lately = frames.filter((f) => f.wall > lastWall - 2);
  if (lately.length > 2 && !state.run?.replay) {
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
  body.innerHTML = [...table.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([type, row]) => {
    const rate = row.times.filter((t) => t > state.tNow - 1).length;
    return `<tr><td>${type}</td><td class="num">${rate} Hz</td><td class="num">${row.count}</td>` +
      `<td class="fields">${escapeHtml(formatFields(row.fields))}</td></tr>`;
  }).join("");
}

function renderTracking() {
  const items = [];
  const m = state.lastMetrics, f = m && m.filter, track = state.lastTrack, aim = state.lastAim;
  if (state.run && state.run.targets) items.push(["enemies", state.run.targets.map((tg) => `${tg.name} (${tg.number})`).join(", ")]);
  if (f) {
    items.push(["tracking", f.target]);
    items.push(["spin", `${f.omega.toFixed(2)} rad/s  (true ${f.true_omega.toFixed(2)})`]);
    items.push(["radius", `${f.radius.toFixed(3)} m  (true ${f.true_radius.toFixed(3)})`]);
    items.push(["center off by", `${(f.center_xy_m * 100).toFixed(1)} cm`]);
    items.push(["orientation off by", `${(f.orientation_rad * DEG).toFixed(1)}°`]);
  } else {
    items.push(["tracking", track && track.pf_alive ? "yes" : "no"]);
  }
  if (m) items.push(["plates", `${m.detected} detected of ${m.visible} visible`]);
  if (aim) items.push(["aim", `yaw ${(aim.yaw * DEG).toFixed(1)}°  pitch ${(aim.pitch * DEG).toFixed(1)}°  ${aim.distance.toFixed(2)} m  id ${aim.target_id}`]);
  if (track && track.pipeline_delay_s != null) items.push(["lead (delay + fire latency)", `${(track.pipeline_delay_s * 1000).toFixed(1)} ms`]);
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
    lines.push(`<span class="${toCv ? "to-cv" : "to-mcb"}">${t.toFixed(3).padStart(9)}  ${toCv ? "MCB → CV" : "CV → MCB"}  ` +
      `${msg.type.padEnd(16)}</span>${escapeHtml(formatFields(msg.fields))}`);
  }
  box.innerHTML = lines.reverse().join("\n") || `<span class="empty">no messages</span>`;
  if (atBottom) box.scrollTop = box.scrollHeight;
}

function formatFields(fields) {
  if (!fields) return "";
  return Object.entries(fields).map(([k, v]) => `${k} ${typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(3)) : v}`).join("  ");
}

function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);
}

function setConnection(kind, text) {
  const pill = $("conn");
  pill.className = `pill ${kind}`;
  pill.textContent = text;
}

// --- time controls -----------------------------------------------------------------------------

function parseDuration(text) {
  const m = /^\s*(\d+(?:\.\d+)?|\.\d+)\s*(ms|s|sec|m|min|h)?\s*$/i.exec(text);
  if (!m) return null;
  const unit = (m[2] || "s").toLowerCase();
  const seconds = Number(m[1]) * { ms: 0.001, s: 1, sec: 1, m: 60, min: 60, h: 3600 }[unit];
  return seconds > 0 ? seconds : null;
}

function formatDuration(s) {
  if (s >= 3600 && s % 3600 === 0) return `${s / 3600} h`;
  if (s >= 120 && s % 60 === 0) return `${s / 60} min`;
  return `${+s.toFixed(3)} s`;
}

function updateTimeControls() {
  const button = $("live-btn");
  button.textContent = view.live ? "● Live" : "❚❚ Paused (back to live)";
  button.className = view.live ? "live" : "paused";
  $("range-box").hidden = view.live;
  const from = $("range-from"), to = $("range-to");
  if (!view.live && document.activeElement !== from && document.activeElement !== to) {
    from.value = view.t0.toFixed(3);
    to.value = view.t1.toFixed(3);
  }
}

$("live-btn").onclick = () => (view.live ? view.pause() : view.goLive());
$("window-input").onchange = (e) => {
  const seconds = parseDuration(e.target.value);
  if (seconds) {
    view.windowS = seconds;
    if (!view.live) view.set(view.t1 - seconds, view.t1);
    requestDraw();
  }
  e.target.value = formatDuration(view.windowS);
};
const setRangeFromInputs = () => view.set(Number($("range-from").value), Number($("range-to").value));
$("range-from").onchange = setRangeFromInputs;
$("range-to").onchange = setRangeFromInputs;

// --- connection --------------------------------------------------------------------------------

async function onSnapshot(packet) {
  resetState();
  const run = packet.events.find((e) => e.topic === "run");
  hub.reset(packet.signals || [], run ? run.t : packet.events.length ? packet.events[0].t : 0);
  for (const event of packet.events) ingest(event);
  try {
    const catalog = await (await fetch("/api/catalog")).json();
    hub.addNames(catalog.signals || []);
    if (catalog.span) state.tNow = Math.max(state.tNow, catalog.span[1] - hub.offset);
  } catch (e) {
    console.warn("catalog", e);
  }
  subscribed = new Set();
  subscribedKey = "";
  updateSubscription();
  if (state.run && state.run.replay) {
    setConnection("replay", "replay");
    view.set(0, Math.max(state.tNow, 1)); // a recorded run: show all of it, paused
  }
  requestDraw();
}

function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => setConnection("live", "live");
  ws.onmessage = (message) => {
    const packet = JSON.parse(message.data);
    if (packet.kind === "snapshot") {
      onSnapshot(packet);
      return;
    }
    for (const event of packet.events) ingest(event);
    if (packet.series) hub.append(packet.series);
    if (packet.new_signals && packet.new_signals.length && hub.addNames(packet.new_signals)) updateSubscription();
  };
  ws.onclose = () => {
    if (!state.ended) setConnection("down", "waiting for a run…");
    setTimeout(connect, 1000); // the next run (on the same port) picks up where this one left off
  };
}

async function start() {
  resetState();
  await layout.load();
  renderTabs();
  buildTab();
  updateTimeControls();
  connect();
  setInterval(render, 150);
}

start();
