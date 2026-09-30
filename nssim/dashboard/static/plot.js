// A time-series plot on a canvas. It draws what it is given (times in seconds since the run
// started) over a time range, and turns gestures into requests to whoever owns the time range:
// drag to zoom to a range, Shift+drag (or middle-drag) to pan, Ctrl+scroll or pinch to zoom around
// the pointer, double-click to go back to live. Hovering shows the values under the pointer.
// No libraries, so the page works without internet.
"use strict";

class TimePlot {
  // handlers: { zoom(t0, t1), pan(dt), scale(tCenter, factor), live() }
  constructor(canvas, handlers) {
    this.canvas = canvas;
    this.handlers = handlers;
    this.area = null; // plot area and time range of the last draw
    this.hoverX = null;
    this.drag = null;
    this.last = null; // arguments of the last draw, to redraw for hover and selection
    this._bind();
  }

  timeAt(x) {
    const a = this.area;
    return a.t0 + ((x - a.left) / a.w) * (a.t1 - a.t0);
  }

  _bind() {
    const c = this.canvas;
    c.addEventListener("pointerdown", (e) => {
      if (!this.area || (e.button !== 0 && e.button !== 1)) return;
      const pan = e.shiftKey || e.button === 1;
      this.drag = { mode: pan ? "pan" : "select", x0: e.offsetX, x1: e.offsetX, lastX: e.offsetX };
      c.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    c.addEventListener("pointermove", (e) => {
      if (!this.area) return;
      if (this.drag) {
        if (this.drag.mode === "pan") {
          const dt = -((e.offsetX - this.drag.lastX) / this.area.w) * (this.area.t1 - this.area.t0);
          this.drag.lastX = e.offsetX;
          if (dt) this.handlers.pan(dt);
        } else {
          this.drag.x1 = e.offsetX;
          this.redraw();
        }
      } else {
        this.hoverX = e.offsetX;
        this.redraw();
      }
    });
    const finish = (e, cancelled) => {
      const drag = this.drag;
      this.drag = null;
      if (drag && drag.mode === "select" && !cancelled && Math.abs(drag.x1 - drag.x0) > 4) {
        const a = this.timeAt(Math.min(drag.x0, drag.x1));
        const b = this.timeAt(Math.max(drag.x0, drag.x1));
        this.handlers.zoom(a, b);
      }
      this.redraw();
    };
    c.addEventListener("pointerup", (e) => finish(e, false));
    c.addEventListener("pointercancel", (e) => finish(e, true)); // e.g. the page scrolled instead
    c.addEventListener("pointerleave", () => {
      if (!this.drag) {
        this.hoverX = null;
        this.redraw();
      }
    });
    c.addEventListener("wheel", (e) => {
      if (!this.area || !(e.ctrlKey || e.metaKey)) return; // plain scrolling scrolls the page
      e.preventDefault();
      this.handlers.scale(this.timeAt(e.offsetX), Math.exp(e.deltaY * 0.003));
    }, { passive: false });
    c.addEventListener("dblclick", () => this.handlers.live());
  }

  redraw() {
    if (this.last) this.draw(...this.last);
  }

  // series: [{ label, color, unit, scale, wrap, t: [], v: [] }] (raw values; scale and wrap are
  // applied while drawing). y: { auto, zero, min, max }.
  draw(series, t0, t1, y, digits = 2) {
    this.last = [series, t0, t1, y, digits];
    const canvas = this.canvas;
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth;
    const cssH = canvas.clientHeight;
    if (!cssW || !cssH) return;
    if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    const css = getComputedStyle(document.documentElement);
    const gridColor = css.getPropertyValue("--grid");
    const mutedColor = css.getPropertyValue("--muted");
    const textColor = css.getPropertyValue("--text");
    const accent = css.getPropertyValue("--accent");
    const font = "11px " + css.getPropertyValue("--mono");
    ctx.font = font;

    const value = (s, i) => {
      let v = s.v[i] * (s.scale || 1);
      if (s.wrap) v = ((((v + s.wrap / 2) % s.wrap) + s.wrap) % s.wrap) - s.wrap / 2;
      return v;
    };

    // Legend first (it wraps on narrow screens; the plot starts below it). Values are the ones
    // under the pointer, or the newest in view.
    const hoverT = this.area && this.hoverX !== null && this.hoverX >= this.area.left && this.hoverX <= this.area.left + this.area.w
      ? this.timeAt(this.hoverX) : null;
    const legend = [];
    let lx = 48, row = 0;
    for (const s of series) {
      const i = hoverT === null ? lastIndexAtOrBefore(s, t1) : nearestIndex(s, hoverT);
      const v = i < 0 ? NaN : value(s, i);
      const shown = Number.isFinite(v) ? v.toFixed(digits) : "–";
      const label = `${s.label} ${shown}${s.unit ? " " + s.unit : ""}`;
      const width = ctx.measureText(label).width + 14;
      if (lx > 48 && lx + width > cssW - 8) {
        lx = 48;
        row++;
      }
      legend.push({ color: s.color, label, x: lx, y: 9 + row * 14 });
      lx += width + 16;
    }

    const left = 48, right = 8, top = 22 + row * 14, bottom = 18;
    const w = Math.max(10, cssW - left - right);
    const h = Math.max(10, cssH - top - bottom);
    this.area = { left, top, w, h, t0, t1 };
    const span = t1 - t0 || 1;
    const x = (t) => left + ((t - t0) / span) * w;

    // y range: fixed, or what is visible without its extreme 1% on each side (so one spike, like
    // the CV warming up, doesn't flatten the rest; points beyond are clipped).
    let lo, hi;
    if (!y.auto && Number.isFinite(y.min) && Number.isFinite(y.max) && y.max > y.min) {
      lo = y.min;
      hi = y.max;
    } else {
      const sample = [];
      for (const s of series) {
        const start = lowerBound(s.t, t0);
        const end = lowerBound(s.t, t1 + 1e-9);
        const step = Math.max(1, Math.floor((end - start) / 1000));
        for (let i = start; i < end; i += step) {
          const v = value(s, i);
          if (Number.isFinite(v)) sample.push(v);
        }
      }
      lo = Infinity;
      hi = -Infinity;
      if (sample.length > 50) {
        sample.sort((a, b) => a - b);
        lo = sample[Math.floor(sample.length * 0.01)];
        hi = sample[Math.ceil(sample.length * 0.99) - 1];
      } else {
        for (const v of sample) {
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
      }
      if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
      if (y.zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
      if (hi - lo < 1e-9) { lo -= 1; hi += 1; }
      const pad = (hi - lo) * 0.08;
      lo -= pad;
      hi += pad;
    }
    const yOf = (v) => top + (1 - (v - lo) / (hi - lo)) * h;

    // grid and labels
    ctx.lineWidth = 1;
    ctx.strokeStyle = gridColor;
    ctx.fillStyle = mutedColor;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const yStep = niceStep((hi - lo) / 4);
    for (let v = Math.ceil(lo / yStep) * yStep; v <= hi; v += yStep) {
      const py = Math.round(yOf(v)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(left, py);
      ctx.lineTo(left + w, py);
      ctx.stroke();
      ctx.fillText(formatTick(v, yStep), left - 6, py);
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const tStep = niceStep(span / 5);
    for (let t = Math.ceil(t0 / tStep) * tStep; t <= t1; t += tStep) {
      const px = Math.round(x(t)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(px, top);
      ctx.lineTo(px, top + h);
      ctx.stroke();
      ctx.fillText(formatTick(t, tStep) + " s", px, top + h + 4);
    }

    // series
    ctx.save();
    ctx.beginPath();
    ctx.rect(left, top, w, h);
    ctx.clip();
    ctx.lineWidth = 1.5;
    ctx.lineJoin = "round";
    let anyData = false;
    for (const s of series) {
      const start = Math.max(0, lowerBound(s.t, t0) - 1);
      const end = Math.min(s.t.length, lowerBound(s.t, t1 + 1e-9) + 1);
      const stride = Math.max(1, Math.floor((end - start) / (w * 2))); // ~2 points per pixel at most
      ctx.strokeStyle = s.color;
      ctx.beginPath();
      let pen = false, last = NaN;
      for (let i = start; i < end; i += stride) {
        const v = value(s, i);
        if (!Number.isFinite(v)) { pen = false; continue; }
        if (pen && s.wrap && Math.abs(v - last) > s.wrap / 2) pen = false;
        const px = x(s.t[i]), py = yOf(v);
        if (pen) ctx.lineTo(px, py);
        else ctx.moveTo(px, py);
        pen = true;
        last = v;
        anyData = true;
      }
      ctx.stroke();
    }
    ctx.restore();

    if (!anyData) {
      ctx.fillStyle = mutedColor;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(series.length ? "no data in this range" : "no signals: edit this graph to add some", left + w / 2, top + h / 2);
    }

    // pointer line, selection
    if (hoverT !== null && !this.drag) {
      const px = Math.round(this.hoverX) + 0.5;
      ctx.strokeStyle = textColor;
      ctx.globalAlpha = 0.35;
      ctx.beginPath();
      ctx.moveTo(px, top);
      ctx.lineTo(px, top + h);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = textColor;
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText(`${hoverT.toFixed(3)} s`, Math.min(px + 4, left + w - 70), top + 2);
    }
    if (this.drag && this.drag.mode === "select") {
      const a = Math.max(left, Math.min(this.drag.x0, this.drag.x1));
      const b = Math.min(left + w, Math.max(this.drag.x0, this.drag.x1));
      ctx.fillStyle = accent;
      ctx.globalAlpha = 0.18;
      ctx.fillRect(a, top, b - a, h);
      ctx.globalAlpha = 1;
    }

    // legend
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    for (const item of legend) {
      ctx.fillStyle = item.color;
      ctx.fillRect(item.x, item.y - 2, 10, 3);
      ctx.fillStyle = mutedColor;
      ctx.fillText(item.label, item.x + 14, item.y);
    }
  }
}

function lowerBound(sorted, value) {
  let lo = 0, hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sorted[mid] < value) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function lastIndexAtOrBefore(s, t) {
  for (let i = Math.min(lowerBound(s.t, t + 1e-9), s.t.length) - 1, n = 0; i >= 0 && n < 200; i--, n++) {
    if (Number.isFinite(s.v[i])) return i;
  }
  return -1;
}

function nearestIndex(s, t) {
  const i = lowerBound(s.t, t);
  if (i <= 0) return s.t.length ? 0 : -1;
  if (i >= s.t.length) return s.t.length - 1;
  return t - s.t[i - 1] < s.t[i] - t ? i - 1 : i;
}

function niceStep(raw) {
  const exp = Math.pow(10, Math.floor(Math.log10(raw)));
  const f = raw / exp;
  return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * exp;
}

function formatTick(value, step) {
  const digits = Math.max(0, -Math.floor(Math.log10(step)));
  return value.toFixed(Math.min(digits, 3));
}
