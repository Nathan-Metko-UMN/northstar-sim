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
