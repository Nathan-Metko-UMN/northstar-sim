// A small time-series plot on a canvas: a few series against the run's clock, over a sliding
// window, with an auto-scaled y axis and a legend holding the latest values. No libraries, so the
// page works without internet.
"use strict";

class TimePlot {
  // options: { unit, series: [{ name, color }], zero: bool (keep 0 in view), wrap: period (angles:
  // lines break where the value wraps), digits }
  constructor(canvas, options) {
    this.canvas = canvas;
    this.unit = options.unit || "";
    this.zero = !!options.zero;
    this.wrap = options.wrap || 0;
    this.digits = options.digits ?? 2;
    this.series = options.series.map((s) => ({ ...s, t: [], v: [] }));
    this.keepS = 600; // older points are dropped
  }

  push(index, t, value) {
    const s = this.series[index];
    const n = s.t.length;
    if (n && t < s.t[n - 1]) return; // out of order (e.g. a late event): skip rather than tangle
    s.t.push(t);
    s.v.push(value === null || value === undefined || !Number.isFinite(value) ? NaN : value);
    if (n > 8192 && t - s.t[0] > this.keepS) {
      const cut = lowerBound(s.t, t - this.keepS);
      s.t.splice(0, cut);
      s.v.splice(0, cut);
    }
  }

  clear() {
    for (const s of this.series) {
      s.t.length = 0;
      s.v.length = 0;
    }
  }

  draw(tEnd, windowS) {
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
    const font = "11px " + css.getPropertyValue("--mono");

    // Legend first: it wraps onto more lines on a narrow screen, and the plot starts below it.
    ctx.font = font;
    const legend = [];
    let lx = 48, row = 0;
    for (const s of this.series) {
      const latest = lastFinite(s.v);
      const value = latest === null ? "–" : latest.toFixed(this.digits);
      const label = `${s.name} ${value}${this.unit ? " " + this.unit : ""}`;
      const width = ctx.measureText(label).width + 14;
      if (lx > 48 && lx + width > cssW - 8) {
        lx = 48;
        row++;
      }
      legend.push({ color: s.color, label, x: lx, y: 9 + row * 14 });
      lx += width + 16;
    }

    const left = 48, right = 8, top = 22 + row * 14, bottom = 18;
    const w = cssW - left - right;
    const h = cssH - top - bottom;
    // Until the run is a window long, fill from its start instead of showing empty time before it.
    const t0 = Math.max(0, tEnd - windowS);
    tEnd = t0 + windowS;

    // y range: what is visible, ignoring the extreme 1% on each side so one spike (the CV warming
    // up, say) doesn't flatten everything else. Points beyond the range are clipped.
    const sample = [];
    for (const s of this.series) {
      const start = lowerBound(s.t, t0);
      const step = Math.max(1, Math.floor((s.t.length - start) / 1000));
      for (let i = start; i < s.t.length; i += step) {
        if (Number.isFinite(s.v[i])) sample.push(s.v[i]);
      }
    }
    let lo = Infinity, hi = -Infinity;
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
    if (this.zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
    if (hi - lo < 1e-9) { lo -= 1; hi += 1; }
    const pad = (hi - lo) * 0.08;
    lo -= pad;
    hi += pad;
    const x = (t) => left + ((t - t0) / windowS) * w;
    const y = (v) => top + (1 - (v - lo) / (hi - lo)) * h;

    // grid and labels
    ctx.font = font;
    ctx.lineWidth = 1;
    ctx.strokeStyle = gridColor;
    ctx.fillStyle = mutedColor;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const yStep = niceStep((hi - lo) / 4);
    for (let v = Math.ceil(lo / yStep) * yStep; v <= hi; v += yStep) {
      const py = Math.round(y(v)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(left, py);
      ctx.lineTo(left + w, py);
      ctx.stroke();
      ctx.fillText(formatTick(v, yStep), left - 6, py);
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const tStep = niceStep(windowS / 5);
    for (let t = Math.ceil(t0 / tStep) * tStep; t <= tEnd; t += tStep) {
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
    for (const s of this.series) {
      const start = Math.max(0, lowerBound(s.t, t0) - 1);
      const n = s.t.length - start;
      const stride = Math.max(1, Math.floor(n / (w * 2))); // no more than ~2 points per pixel
      ctx.strokeStyle = s.color;
      ctx.beginPath();
      let pen = false, last = NaN;
      for (let i = start; i < s.t.length; i += stride) {
        const v = s.v[i];
        if (Number.isNaN(v)) { pen = false; continue; }
        if (pen && this.wrap && Math.abs(v - last) > this.wrap / 2) pen = false;
        const px = x(s.t[i]), py = y(v);
        if (pen) ctx.lineTo(px, py);
        else ctx.moveTo(px, py);
        pen = true;
        last = v;
      }
      ctx.stroke();
    }
    ctx.restore();

    // legend: name and latest value
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

function niceStep(raw) {
  const exp = Math.pow(10, Math.floor(Math.log10(raw)));
  const f = raw / exp;
  return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * exp;
}

function formatTick(value, step) {
  const digits = Math.max(0, -Math.floor(Math.log10(step)));
  return value.toFixed(Math.min(digits, 3));
}

function lastFinite(values) {
  for (let i = values.length - 1; i >= 0 && i >= values.length - 50; i--) {
    if (Number.isFinite(values[i])) return values[i];
  }
  return null;
}
