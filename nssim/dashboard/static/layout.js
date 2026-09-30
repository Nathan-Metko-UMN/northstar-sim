// Tabs and graphs: the layout (saved on the sim machine, so every browser sees the same one), the
// graph panels, and the editor for a graph's signals, units and y axis.
"use strict";

const PALETTE = ["--c1", "--c2", "--c3", "--c4", "--c5", "--c6", "--c7", "--c8"];

const DEFAULT_LAYOUT = {
  version: 1,
  active: "overview",
  tabs: [
    {
      id: "overview",
      name: "Overview",
      builtin: true,
      graphs: [
        { id: "yaw", title: "Turret yaw", y: { auto: true }, signals: [
          { name: "uart.mcb_to_cv.ODOMETRY.yaw", label: "turret (odometry)", transform: "deg" },
          { name: "uart.cv_to_mcb.TURRET_AIM_DATA.yaw", label: "CV aim", transform: "deg" },
          { name: "truth.targets.*.bearing", label: "robot", transform: "deg" }] },
        { id: "spin", title: "Spin rate", unit: "rad/s", y: { auto: true, zero: true }, signals: [
          { name: "metrics.filter.true_omega", label: "true" },
          { name: "metrics.filter.omega", label: "filter" }] },
        { id: "center", title: "Filter center error", y: { auto: true, zero: true }, signals: [
          { name: "metrics.filter.center_xy_m", label: "horizontal", transform: "cm" },
          { name: "metrics.filter.center_z_m", label: "height", transform: "cm" }] },
        { id: "radius", title: "Filter radius error", y: { auto: true, zero: true }, signals: [
          { name: "metrics.filter.radius_m", label: "radius", transform: "cm" }] },
        { id: "orient", title: "Filter orientation error (mod 180°)", y: { auto: true, zero: true }, signals: [
          { name: "metrics.filter.orientation_rad", label: "orientation", transform: "deg" }] },
        { id: "plates", title: "Plates per frame", y: { auto: true, zero: true }, digits: 0, signals: [
          { name: "metrics.visible", label: "visible" },
          { name: "metrics.detected", label: "detected" }] },
        { id: "latency", title: "Latency from capture", unit: "ms", y: { auto: true, zero: true }, signals: [
          { name: "frame.arrival_ms", label: "frame arrives" },
          { name: "frame.processing_ms", label: "CV processing" },
          { name: "frame.aim_ms", label: "aim at MCB" }] },
      ],
    },
  ],
};

class Layout {
  constructor() {
    this.data = structuredClone(DEFAULT_LAYOUT);
    this._saveTimer = null;
  }

  async load() {
    try {
      const saved = await (await fetch("/api/layout")).json();
      if (saved && Array.isArray(saved.tabs) && saved.tabs.length) this.data = saved;
    } catch (e) {
      console.warn("layout not loaded, using the default", e);
    }
    if (!this.tab(this.data.active)) this.data.active = this.data.tabs[0].id;
  }

  save() {
    clearTimeout(this._saveTimer);
    this._saveTimer = setTimeout(() => {
      fetch("/api/layout", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(this.data) })
        .catch((e) => console.warn("layout not saved", e));
    }, 400);
  }

  get tabs() { return this.data.tabs; }
  tab(id) { return this.data.tabs.find((t) => t.id === id); }
  get active() { return this.tab(this.data.active); }

  setActive(id) {
    this.data.active = id;
    this.save();
  }

  addTab() {
    const tab = { id: newId("tab"), name: `Tab ${this.data.tabs.length + 1}`, graphs: [] };
    this.data.tabs.push(tab);
    this.data.active = tab.id;
    this.save();
    return tab;
  }

  removeTab(id) {
    this.data.tabs = this.data.tabs.filter((t) => t.id !== id);
    if (this.data.active === id) this.data.active = this.data.tabs[0].id;
    this.save();
  }

  addGraph(tab) {
    const graph = { id: newId("graph"), title: "New graph", y: { auto: true }, signals: [] };
    tab.graphs.push(graph);
    this.save();
    return graph;
  }

  removeGraph(tab, graph) {
    tab.graphs = tab.graphs.filter((g) => g !== graph);
    this.save();
  }

  allGraphs() {
    return this.data.tabs.flatMap((t) => t.graphs);
  }
}

// One graph on screen: header, plot, and (when open) its editor.
class GraphView {
  constructor(graph, tab, app) {
    this.graph = graph;
    this.tab = tab;
    this.app = app;
    this.cache = null; // fetched data for the current range: { key, data: Map }
    this.fetching = null;
    this.el = document.createElement("section");
    this.el.className = "panel graph";
    this.el.innerHTML = `
      <div class="graph-head">
        <h2></h2>
        <div class="tools">
          <button class="icon" data-act="edit" title="Edit signals, units and y axis">Edit</button>
          <button class="icon" data-act="wide" title="Full width">Wide</button>
          <button class="icon" data-act="remove" title="Remove this graph">✕</button>
        </div>
      </div>
      <canvas class="plot"></canvas>
      <div class="editor" hidden></div>`;
    this.el.querySelector('[data-act="edit"]').onclick = () => this.toggleEditor();
    this.el.querySelector('[data-act="wide"]').onclick = () => {
      this.graph.wide = !this.graph.wide;
      this.applyLook();
      app.layout.save();
      app.requestDraw();
    };
    this.el.querySelector('[data-act="remove"]').onclick = () => {
      if (confirm(`Remove "${this.graph.title}"?`)) app.removeGraph(this);
    };
    this.plot = new TimePlot(this.el.querySelector("canvas"), app.view.handlers);
    this.applyLook();
  }

  applyLook() {
    this.el.querySelector("h2").textContent = this.graph.title;
    this.el.classList.toggle("wide", !!this.graph.wide);
  }

  // Concrete signal names (wildcards expanded) with how to show each.
  lines() {
    const hub = this.app.hub;
    const out = [];
    for (const sig of this.graph.signals) {
      const names = hub.expand(sig.name);
      const tf = TRANSFORMS[sig.transform] || TRANSFORMS.none;
      for (const name of names) {
        const base = sig.label || shortName(name);
        const label = names.length > 1 || sig.name.includes("*") ? `${base} ${wildcardPart(sig.name, name)}`.trim() : base;
        out.push({ name, label, tf });
      }
    }
    return out;
  }

  draw(t0, t1, fromBuffers) {
    const hub = this.app.hub;
    const css = getComputedStyle(document.documentElement);
    const series = this.lines().map((line, i) => {
      const data = (!fromBuffers && this.cache && this.cache.data.get(line.name)) || hub.buffers.get(line.name);
      return {
        label: line.label,
        color: css.getPropertyValue(PALETTE[i % PALETTE.length]).trim(),
        unit: line.tf.unit || this.graph.unit || "",
        scale: line.tf.scale,
        wrap: line.tf.wrap,
        t: data ? data.t : [],
        v: data ? data.v : [],
      };
    });
    this.plot.draw(series, t0, t1, this.graph.y || { auto: true }, this.graph.digits ?? 2);
  }

  // Data for a range the live buffers don't cover (paused, zoomed, or a long window), fetched once
  // per range and redrawn when it arrives.
  ensureFetched(t0, t1, key) {
    const names = this.lines().map((l) => l.name);
    const width = this.el.querySelector("canvas").clientWidth || 600;
    const fullKey = `${key}|${names.join(",")}|${width}`;
    if ((this.cache && this.cache.key === fullKey) || this.fetching === fullKey || !names.length) return;
    this.fetching = fullKey;
    this.app.hub.fetch(names, t0, t1, Math.round(width * 2))
      .then((data) => {
        if (this.fetching === fullKey) this.cache = { key: fullKey, data };
      })
      .catch(() => {}) // the server is gone (the run ended): what the live buffers hold still draws
      .finally(() => {
        if (this.fetching === fullKey) this.fetching = null;
        this.app.requestDraw();
      });
  }

  toggleEditor() {
    const editor = this.el.querySelector(".editor");
    if (!editor.hidden) {
      editor.hidden = true;
      return;
    }
    this.buildEditor(editor);
    editor.hidden = false;
  }

  buildEditor(editor) {
    const g = this.graph;
    g.y = g.y || { auto: true };
    editor.innerHTML = `
      <label class="row">Title <input data-k="title"></label>
      <div class="sig-list"></div>
      <div class="row sig-add">
        <input list="signal-list" placeholder="add a signal: type to search (* matches any part)" spellcheck="false">
        <button>Add</button>
      </div>
      <div class="row">
        <label><input type="checkbox" data-k="auto"> auto y</label>
        <label><input type="checkbox" data-k="zero"> include 0</label>
        <label>min <input type="number" step="any" data-k="min"></label>
        <label>max <input type="number" step="any" data-k="max"></label>
      </div>
      <div class="row">
        <label>Unit <input data-k="unit" placeholder="for signals shown as is"></label>
        <label>Decimals <input type="number" min="0" max="6" data-k="digits"></label>
        <button class="done">Done</button>
      </div>`;
    const changed = () => {
      this.applyLook();
      this.cache = null;
      this.app.layout.save();
      this.app.graphsChanged();
    };
    const title = editor.querySelector('[data-k="title"]');
    title.value = g.title;
    title.oninput = () => { g.title = title.value; changed(); };

    const list = editor.querySelector(".sig-list");
    const renderSignals = () => {
      list.innerHTML = "";
      g.signals.forEach((sig, i) => {
        const row = document.createElement("div");
        row.className = "sig-row";
        const options = Object.entries(TRANSFORMS)
          .map(([k, t]) => `<option value="${k}"${(sig.transform || "none") === k ? " selected" : ""}>${t.label}</option>`).join("");
        row.innerHTML = `<code></code><input class="label" placeholder="label"><select>${options}</select><button title="Remove">✕</button>`;
        row.querySelector("code").textContent = sig.name;
        const label = row.querySelector("input");
        label.value = sig.label || "";
        label.oninput = () => { sig.label = label.value; changed(); };
        row.querySelector("select").onchange = (e) => { sig.transform = e.target.value; changed(); };
        row.querySelector("button").onclick = () => { g.signals.splice(i, 1); renderSignals(); changed(); };
        list.append(row);
      });
      if (!g.signals.length) list.innerHTML = `<div class="empty">No signals yet.</div>`;
    };
    renderSignals();

    const input = editor.querySelector(".sig-add input");
    const add = () => {
      const name = input.value.trim();
      if (!name) return;
      g.signals.push({ name, label: "", transform: guessTransform(name) });
      input.value = "";
      renderSignals();
      changed();
      input.focus();
    };
    editor.querySelector(".sig-add button").onclick = add;
    input.onkeydown = (e) => { if (e.key === "Enter") add(); };

    const auto = editor.querySelector('[data-k="auto"]');
    const zero = editor.querySelector('[data-k="zero"]');
    const min = editor.querySelector('[data-k="min"]');
    const max = editor.querySelector('[data-k="max"]');
    const syncY = () => { min.disabled = max.disabled = auto.checked; zero.disabled = !auto.checked; };
    auto.checked = g.y.auto !== false;
    zero.checked = !!g.y.zero;
    min.value = g.y.min ?? "";
    max.value = g.y.max ?? "";
    syncY();
    auto.onchange = () => {
      g.y.auto = auto.checked;
      if (!auto.checked && (min.value === "" || max.value === "")) {
        const [lo, hi] = this.visibleRange();
        min.value = g.y.min = Number(lo.toPrecision(3));
        max.value = g.y.max = Number(hi.toPrecision(3));
      }
      syncY();
      changed();
    };
    zero.onchange = () => { g.y.zero = zero.checked; changed(); };
    min.oninput = () => { g.y.min = min.value === "" ? null : Number(min.value); changed(); };
    max.oninput = () => { g.y.max = max.value === "" ? null : Number(max.value); changed(); };

    const unit = editor.querySelector('[data-k="unit"]');
    unit.value = g.unit || "";
    unit.oninput = () => { g.unit = unit.value; changed(); };
    const digits = editor.querySelector('[data-k="digits"]');
    digits.value = g.digits ?? 2;
    digits.oninput = () => { g.digits = digits.value === "" ? 2 : Number(digits.value); changed(); };
    editor.querySelector(".done").onclick = () => { editor.hidden = true; };
  }

  // Current auto-scaled range, to start a fixed y axis from what is on screen.
  visibleRange() {
    const series = this.plot.last ? this.plot.last[0] : [];
    let lo = Infinity, hi = -Infinity;
    for (const s of series) {
      for (let i = 0; i < s.v.length; i++) {
        const v = s.v[i] * (s.scale || 1);
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    return Number.isFinite(lo) ? [lo, hi] : [0, 1];
  }
}

function newId(prefix) {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;
}

function shortName(name) {
  return name.split(".").slice(-2).join(".");
}

// The parts of `name` that the pattern's wildcards matched ("truth.targets.*.bearing" ->
// "infantry").
function wildcardPart(pattern, name) {
  const p = pattern.split("."), n = name.split(".");
  return p.map((part, i) => (part.includes("*") ? n[i] : null)).filter(Boolean).join(".");
}

// A sensible first unit conversion from the signal's name.
function guessTransform(name) {
  const leaf = name.split(".").pop();
  if (/(^|_)(yaw|pitch|roll|orientation|bearing|spin)(_rad)?$/.test(leaf) || leaf.endsWith("_rad")) return "deg";
  return "none";
}
