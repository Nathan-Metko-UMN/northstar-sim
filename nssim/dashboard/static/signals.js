// The page's side of the signals (see nssim/dashboard/store.py): the catalog of names, live
// buffers of the ones being graphed (streamed by the server), and fetches of any range from the
// server's full record of the run. Times here are seconds since the run started.
"use strict";

const TRANSFORMS = {
  none: { label: "as is", scale: 1, wrap: 0, unit: "" },
  deg: { label: "rad → °", scale: 180 / Math.PI, wrap: 360, unit: "°" },
  cm: { label: "m → cm", scale: 100, wrap: 0, unit: "cm" },
  mm: { label: "m → mm", scale: 1000, wrap: 0, unit: "mm" },
  ms: { label: "s → ms", scale: 1000, wrap: 0, unit: "ms" },
  pct: { label: "× 100 (%)", scale: 100, wrap: 0, unit: "%" },
};

class SignalHub {
  constructor() {
    this.keepS = 300; // live buffers hold this much; longer windows and zooms are fetched
    this.reset([], 0);
  }

  reset(names, offset) {
    this.names = [];
    this.nameSet = new Set();
    this.buffers = new Map(); // name -> { t: [], v: [] }
    this.offset = offset; // run clock time of the run's start
    this.version = 0; // bumps when the catalog grows
    this._patterns = new Map();
    this.addNames(names);
  }

  addNames(list) {
    let added = false;
    for (const name of list) {
      if (!this.nameSet.has(name)) {
        this.nameSet.add(name);
        this.names.push(name);
        added = true;
      }
    }
    if (added) {
      this.names.sort();
      this.version++;
      this._patterns.clear();
    }
    return added;
  }

  // "truth.targets.*.bearing" -> every matching name; a plain name stays as it is even before it
  // exists (it may appear later in the run).
  expand(pattern) {
    if (!pattern.includes("*")) return [pattern];
    let hit = this._patterns.get(pattern);
    if (!hit) {
      const re = new RegExp("^" + pattern.split("*").map(escapeRegex).join("[^.]*") + "$");
      hit = this.names.filter((name) => re.test(name));
      this._patterns.set(pattern, hit);
    }
    return hit;
  }

  append(series) {
    for (const [name, points] of Object.entries(series)) {
      let buffer = this.buffers.get(name);
      if (!buffer) this.buffers.set(name, (buffer = { t: [], v: [] }));
      for (const [t, v] of points) {
        const rt = t - this.offset;
        if (buffer.t.length && rt < buffer.t[buffer.t.length - 1]) continue;
        buffer.t.push(rt);
        buffer.v.push(v === null ? NaN : v);
      }
      if (buffer.t.length > 4096 && buffer.t[buffer.t.length - 1] - buffer.t[0] > this.keepS * 1.2) {
        const cut = lowerBound(buffer.t, buffer.t[buffer.t.length - 1] - this.keepS);
        buffer.t.splice(0, cut);
        buffer.v.splice(0, cut);
      }
    }
  }

  // Any signals over [t0, t1], thinned by the server to about `points` each.
  async fetch(names, t0, t1, points) {
    const response = await fetch("/api/series", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ names, t0: t0 + this.offset, t1: t1 + this.offset, points }),
    });
    if (!response.ok) throw new Error(`series: ${response.status}`);
    const body = await response.json();
    const out = new Map();
    for (const [name, { t, v }] of Object.entries(body)) {
      out.set(name, { t: t.map((x) => x - this.offset), v: v.map((x) => (x === null ? NaN : x)) });
    }
    return out;
  }

  // Fill the live buffers of newly graphed signals with what came before they were subscribed.
  async backfill(names, t0, t1) {
    if (!names.length) return;
    const data = await this.fetch(names, t0, t1, 4000);
    for (const [name, older] of data) {
      const buffer = this.buffers.get(name);
      if (!buffer || !buffer.t.length) {
        this.buffers.set(name, older);
        continue;
      }
      const cut = lowerBound(older.t, buffer.t[0]);
      buffer.t = older.t.slice(0, cut).concat(buffer.t);
      buffer.v = older.v.slice(0, cut).concat(buffer.v);
    }
  }
}

function escapeRegex(s) {
  return s.replace(/[.+?^${}()|[\]\\]/g, "\\$&");
}
