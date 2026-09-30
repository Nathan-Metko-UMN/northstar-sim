"""Every number the run produces, as time series the dashboard can plot, zoom and search.

Each event is flattened into named signals: ``uart.mcb_to_cv.ODOMETRY.yaw``,
``cv.track.state.omega``, ``metrics.filter.center_xy_m``, ``truth.targets.infantry.center.x``,
``frame.processing_ms``... Signals from one kind of event share a time column. The page asks for any
signals over any time range and gets them thinned to about what it can draw, keeping each bucket's
minimum and maximum so spikes survive.
"""

from __future__ import annotations

import math
from array import array
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field

import numpy as np

KEEP_S = 1200.0  # per kind of event; older rows are dropped
SKIP_KEYS = {"wall", "_rx_wall"}  # bookkeeping, not data
SAMPLE_LISTS = {"corner_px", "range_m", "lateral_m", "vertical_m"}  # per-detection samples: summarized
AXES = "xyz"


def flatten(event: dict) -> tuple[str, dict[str, float]]:
    """(group, {signal name: value}) of one event. A group is one kind of event (one time column)."""
    topic, data = event["topic"], event["data"]
    if topic == "uart":
        group = f"uart.{data.get('dir')}.{data.get('type')}"
        values: dict[str, float] = {}
        _walk(data.get("fields") or {}, group, values)
        _walk(data.get("seq"), f"{group}.seq", values)
        return group, values
    values = {}
    _walk(data, topic, values)
    return topic, values


def _walk(value, path: str, out: dict[str, float]) -> None:
    if isinstance(value, bool):
        out[path] = 1.0 if value else 0.0
    elif isinstance(value, (int, float)):
        out[path] = float(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key not in SKIP_KEYS:
                _walk(item, f"{path}.{key}", out)
    elif isinstance(value, (list, tuple)):
        if not value:
            return
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
        name = path.rsplit(".", 1)[-1]
        if numeric and name in SAMPLE_LISTS:
            finite = [v for v in value if v is not None and math.isfinite(v)]
            out[f"{path}.count"] = float(len(finite))
            if finite:
                out[f"{path}.mean"] = float(sum(finite) / len(finite))
        elif numeric and len(value) <= 3:  # a vector
            for axis, v in zip(AXES, value):
                out[f"{path}.{axis}"] = float(v)
        else:
            for i, item in enumerate(value):
                key = item.get("name") if isinstance(item, dict) and isinstance(item.get("name"), str) else str(i)
                _walk(item, f"{path}.{key}", out)
    # strings and nulls carry no plottable number


@dataclass
class _Group:
    t: array = field(default_factory=lambda: array("d"))
    columns: dict[str, array] = field(default_factory=dict)

    def add(self, t: float, values: dict[str, float]) -> list[str]:
        """Append one row; returns signal names seen for the first time."""
        new = []
        rows = len(self.t)
        for name, value in values.items():
            if name not in self.columns:
                self.columns[name] = array("d", [math.nan]) * rows
                new.append(name)
        self.t.append(t)
        for name, column in self.columns.items():
            column.append(values.get(name, math.nan))
        return new

    def trim(self, before: float) -> None:
        cut = bisect_left(self.t, before)
        if cut > len(self.t) // 10:  # in chunks, not every row
            del self.t[:cut]
            for column in self.columns.values():
                del column[:cut]


class SignalStore:
    def __init__(self, keep_s: float = KEEP_S):
        self.keep_s = keep_s
        self._groups: dict[str, _Group] = {}
        self._group_of: dict[str, str] = {}  # signal -> group

    def add(self, event: dict) -> tuple[dict[str, float], list[str]]:
        """Store an event; returns its signals and the names that are new."""
        group_name, values = flatten(event)
        if not values:
            return values, []
        group = self._groups.setdefault(group_name, _Group())
        t = float(event["t"])
        if group.t and t < group.t[-1]:  # out of order (rare): keep the time column sorted
            t = group.t[-1]
        new = group.add(t, values)
        for name in new:
            self._group_of[name] = group_name
        group.trim(t - self.keep_s)
        return values, new

    def names(self) -> list[str]:
        return sorted(self._group_of)

    def span(self) -> tuple[float, float] | None:
        """(first, last) time of anything stored."""
        times = [(g.t[0], g.t[-1]) for g in self._groups.values() if g.t]
        if not times:
            return None
        return min(t[0] for t in times), max(t[1] for t in times)

    def series(self, name: str, t0: float, t1: float, points: int = 1000) -> tuple[list[float], list[float]]:
        """One signal between t0 and t1, thinned to about ``points`` points (bucket minima and
        maxima, so spikes stay visible)."""
        group_name = self._group_of.get(name)
        if group_name is None:
            return [], []
        group = self._groups[group_name]
        i0 = max(bisect_left(group.t, t0) - 1, 0)  # one point either side so lines reach the edges
        i1 = min(bisect_right(group.t, t1) + 1, len(group.t))
        if i1 <= i0:
            return [], []
        # Slices are copies: a numpy view of the growing arrays would stop them from growing.
        t = np.frombuffer(group.t[i0:i1], dtype=np.float64)
        v = np.frombuffer(group.columns[name][i0:i1], dtype=np.float64)
        if len(t) <= points:
            return t.tolist(), _nulls(v)
        buckets = max(1, points // 2)
        edges = np.searchsorted(t, np.linspace(t[0], t[-1], buckets + 1)[:-1])
        starts = np.unique(edges)
        with np.errstate(invalid="ignore"):
            low = np.fmin.reduceat(v, starts)
            high = np.fmax.reduceat(v, starts)
        ends = np.append(starts[1:], len(t)) - 1
        # min then max at each bucket's start and end time: a line through both keeps the envelope
        times = np.column_stack([t[starts], t[ends]]).ravel()
        values = np.column_stack([low, high]).ravel()
        return times.tolist(), _nulls(values)


def _nulls(values: np.ndarray) -> list:
    """NaN as None (JSON null): a gap in the line."""
    return [None if x != x else x for x in values.tolist()]
