"""Replay a recorded run in the dashboard: its log files become the events a live run publishes."""

from __future__ import annotations

import json
from pathlib import Path

from nssim.runner import frame_events


def _load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_events(run_dir: Path, yaw_offset) -> list[dict]:
    """The bus events of a recorded run, in time order, starting with its ``run`` event."""
    run_dir = Path(run_dir)
    config_file = run_dir / "config.json"
    config = json.loads(config_file.read_text()) if config_file.is_file() else {}
    frames = _load(run_dir / "frames.jsonl")
    telemetry = _load(run_dir / "telemetry.jsonl")
    fps = config.get("fps") or 166.0

    per_seq = {kind: {m["seq"]: m for m in telemetry if m.get("type") == kind and "seq" in m} for kind in ("det", "track")}
    events = []
    for record in frames:
        seq = record["seq"]
        for topic, t, data in frame_events(record, yaw_offset, per_seq["det"].get(seq), per_seq["track"].get(seq)):
            events.append({"topic": topic, "t": t, "data": data})
    # The CV's telemetry, at about when it came in: its frame's arrival plus the processing time.
    done = {r["seq"]: (r["arrival_us"] + r["processing_us"]) / 1e6 for r in frames}
    for msg in telemetry:
        t = done.get(msg.get("seq"))
        if t is not None:
            events.append({"topic": f"cv.{msg.get('type', '?')}", "t": t, "data": msg})
    for record in _load(run_dir / "uart.jsonl"):  # runs from before uart.jsonl existed have none
        data = {key: record[key] for key in ("dir", "type", "seq", "fields", "bytes") if key in record}
        events.append({"topic": "uart", "t": record["mcb_us"] / 1e6, "data": data})
    events.sort(key=lambda e: e["t"])

    start = frames[0]["capture_us"] / 1e6 - 1.0 / fps if frames else (events[0]["t"] if events else 0.0)
    targets = []
    for target in config.get("targets", []):
        spec = target.get("spec", {})
        targets.append({
            "name": spec.get("name"),
            "number": spec.get("number"),
            "panel": spec.get("panel"),
            "radius": (spec.get("radius_high", 0.0) + spec.get("radius_low", 0.0)) / 2,
            "plate_height": spec.get("z_offset"),
        })
    run = {"name": config.get("name", run_dir.name), "fps": fps, "replay": str(run_dir), "targets": targets}
    return [{"topic": "run", "t": start, "data": run}] + events
