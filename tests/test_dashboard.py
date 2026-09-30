import asyncio
import json
import time

import aiohttp
import pytest

from nssim.bus import Bus
from nssim.dashboard import DashboardServer


@pytest.fixture
def server():
    s = DashboardServer(port=0, host="127.0.0.1")
    s.start()
    yield s
    s.stop()


def _receive(port: int, want: int) -> list[dict]:
    """Connect like the page does and collect the first ``want`` packets."""

    async def main():
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(f"http://127.0.0.1:{port}/ws") as ws:
                packets = []
                while len(packets) < want:
                    msg = await asyncio.wait_for(ws.receive(), timeout=5)
                    packets.append(json.loads(msg.data))
                return packets

    return asyncio.run(main())


def test_page_is_served(server):
    async def main():
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{server.port}/") as page:
                assert page.status == 200
                assert "nssim dashboard" in await page.text()
            async with session.get(f"http://127.0.0.1:{server.port}/static/app.js") as script:
                assert script.status == 200

    asyncio.run(main())


def test_late_browser_gets_history_then_live_batches(server):
    bus = Bus()
    bus.subscribe(server)
    bus.publish("run", 1.0, {"name": "test", "targets": []})
    bus.publish("metrics", 1.1, {"seq": 0, "visible": 2, "detected": 2, "filter": None})
    bus.publish("uart", 1.1, {"dir": "mcb_to_cv", "type": "ODOMETRY", "seq": 1, "fields": {"yaw": float("nan")}})
    time.sleep(0.2)  # let the server take them in before the browser connects

    def publish_later():
        time.sleep(0.3)
        bus.publish("frame", 1.2, {"seq": 1, "processing_ms": 3.0})

    import threading

    threading.Thread(target=publish_later).start()
    snapshot, live = _receive(server.port, 2)
    assert snapshot["kind"] == "snapshot"
    topics = [e["topic"] for e in snapshot["events"]]
    assert topics[0] == "run" and "metrics" in topics and "uart" in topics
    odometry = next(e for e in snapshot["events"] if e["topic"] == "uart")
    assert odometry["data"]["fields"]["yaw"] is None  # NaN can't go into JSON
    assert live["kind"] == "events" and live["events"][0]["topic"] == "frame"


def test_browsers_hear_that_the_run_ended():
    server = DashboardServer(port=0, host="127.0.0.1")
    server.start()
    port = server.port

    async def main():
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(f"http://127.0.0.1:{port}/ws") as ws:
                await ws.receive()  # snapshot
                await asyncio.get_running_loop().run_in_executor(None, server.stop)
                msg = await asyncio.wait_for(ws.receive(), timeout=5)
                return json.loads(msg.data)

    packet = asyncio.run(main())
    assert packet["events"][-1]["topic"] == "end"


def _post(port: int, path: str, body) -> dict:
    async def main():
        async with aiohttp.ClientSession() as session:
            async with session.post(f"http://127.0.0.1:{port}{path}", json=body) as response:
                return await response.json()

    return asyncio.run(main())


def test_any_signal_over_any_range(server):
    bus = Bus()
    bus.subscribe(server)
    bus.publish("run", 10.0, {"name": "t"})
    for i in range(100):
        bus.publish("frame", 10.0 + i * 0.01, {"seq": i, "processing_ms": float(i)})
    time.sleep(0.2)
    body = _post(server.port, "/api/series", {"names": ["frame.processing_ms"], "t0": 10.5, "t1": 10.6})
    series = body["frame.processing_ms"]
    assert series["v"][0] <= 50.0 and series["v"][-1] >= 60.0  # the range, plus a point either side
    assert all(10.49 <= t <= 10.61 for t in series["t"])


def test_browsers_get_the_values_they_subscribed_to():
    server = DashboardServer(port=0, host="127.0.0.1")
    server.start()
    bus = Bus()
    bus.subscribe(server)

    async def main():
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(f"http://127.0.0.1:{server.port}/ws") as ws:
                await ws.receive()  # snapshot
                await ws.send_json({"kind": "subscribe", "signals": ["metrics.detected"]})
                await asyncio.sleep(0.1)
                bus.publish("metrics", 1.0, {"seq": 0, "detected": 2, "visible": 3})
                return json.loads((await asyncio.wait_for(ws.receive(), timeout=5)).data)

    batch = asyncio.run(main())
    server.stop()
    assert batch["series"] == {"metrics.detected": [[1.0, 2.0]]}
    assert "metrics.visible" in batch["new_signals"]


def test_layout_is_kept_for_every_browser(tmp_path):
    server = DashboardServer(port=0, host="127.0.0.1", layout_file=tmp_path / "layout.json")
    server.start()

    async def main():
        async with aiohttp.ClientSession() as session:
            base = f"http://127.0.0.1:{server.port}/api/layout"
            async with session.get(base) as first:
                assert await first.json() is None  # nothing saved yet: the page uses its default
            async with session.put(base, json={"tabs": [{"id": "a", "name": "Mine", "graphs": []}]}):
                pass
            async with session.get(base) as again:
                return await again.json()

    layout = asyncio.run(main())
    server.stop()
    assert layout["tabs"][0]["name"] == "Mine"


def test_a_recorded_run_replays_as_the_same_events(tmp_path):
    from nssim.dashboard.replay import run_events

    gt = {"t": 0.0, "turret": {"yaw": 0.0, "pitch": 0.0, "yaw_rate": 0.0}, "base_world": [0.0, 0.0, 0.0],
          "camera_world": {"p": [0.0, 0.0, 0.4], "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}, "targets": []}
    frame = {"seq": 0, "capture_us": 2_000_000, "arrival_us": 2_005_000, "processing_us": 3000, "aim_us": 2_009_000,
             "image_bytes": 1000, "wall_ms": {}, "gt": gt}
    (tmp_path / "frames.jsonl").write_text(json.dumps(frame) + "\n")
    (tmp_path / "telemetry.jsonl").write_text(json.dumps({"type": "track", "seq": 0, "pf_alive": False}) + "\n")
    (tmp_path / "uart.jsonl").write_text(json.dumps(
        {"mcb_us": 1_999_000, "dir": "mcb_to_cv", "type": "ODOMETRY", "seq": 3, "fields": {"yaw": 1.0}, "bytes": 37}) + "\n")
    (tmp_path / "config.json").write_text(json.dumps({"name": "demo", "fps": 100.0, "targets": []}))

    events = run_events(tmp_path, (0.0, 0.0, 0.0))
    assert events[0]["topic"] == "run" and events[0]["data"]["replay"]
    assert [e["topic"] for e in events[1:]] == ["uart", "frame", "truth", "metrics", "cv.track"]
    frame_event = next(e for e in events if e["topic"] == "frame")
    assert frame_event["data"]["aim_ms"] == 9.0
