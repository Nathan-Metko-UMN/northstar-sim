"""Dashboard server: streams a run's events (see bus.py) to browsers and answers questions about them.

A small aiohttp server in its own thread with its own event loop, so the paced loop only ever does a
non-blocking queue put. Every event also goes into a SignalStore (store.py), which keeps every number
of the run as a time series.

Browsers load the page from ``http://<this machine>:<port>`` and open a WebSocket. They get a
snapshot first (the newest event of each kind, the recent serial traffic and the names of all
signals), then batches about 20 times a second: the new events, the new values of the signals that
browser plots (it subscribes to them), and names seen for the first time. For anything else (an
older time range, a zoomed view) the page asks ``POST /api/series``. A browser that falls behind
loses its oldest batches instead of holding anything up.
"""

from __future__ import annotations

import asyncio
import json
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web

from nssim.paths import REPO_ROOT

from .store import SignalStore

STATIC_DIR = Path(__file__).parent / "static"
LAYOUT_FILE = REPO_ROOT / ".nssim" / "dashboard_layout.json"  # the tabs and graphs, shared by all browsers
UART_TAIL = 400  # messages a new browser gets for its log
BATCH_S = 0.05
CLIENT_BACKLOG = 100  # batches queued per browser before the oldest are dropped
MAX_POINTS = 20_000  # per signal per /api/series request


def _plain(obj):
    """Events as plain JSON types. JSON has no NaN or infinity (the browser's parser rejects the
    whole message), so those become null; a diverged filter can send them."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {key: _plain(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, np.generic):
        return _plain(obj.item())
    return obj


def _dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), allow_nan=False)


@dataclass
class _Browser:
    ws: web.WebSocketResponse
    backlog: asyncio.Queue
    signals: set[str] = field(default_factory=set)


class DashboardServer:
    """A bus sink (call it with events) that serves them to browsers."""

    def __init__(self, port: int = 8050, host: str = "0.0.0.0", layout_file: Path = LAYOUT_FILE):
        self.host = host
        self.port = port
        self.layout_file = Path(layout_file)
        self.store = SignalStore()
        self._incoming: queue.SimpleQueue = queue.SimpleQueue()
        self._uart: deque = deque(maxlen=UART_TAIL)
        self._latest: dict[str, dict] = {}  # newest event of every other topic (run, cv.track, ...)
        self._browsers: list[_Browser] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None

    # --- the sim side ----------------------------------------------------------------------

    def __call__(self, event: dict) -> None:
        event["wall"] = time.time()  # lets the page work out the real-time factor
        self._incoming.put(event)

    def load(self, events) -> None:
        """Take in a whole recorded run before serving it (replay)."""
        for event in events:
            self._take(_plain(event))

    def start(self) -> None:
        self._thread = threading.Thread(target=self._main, name="dashboard", daemon=True)
        self._thread.start()
        self._ready.wait(10)
        if self._error is not None:
            raise self._error

    def stop(self) -> None:
        """Tell the browsers the run ended, give them a moment to get it, and shut down."""
        if self._loop is None:
            return
        self({"topic": "end", "t": 0.0, "data": {}})
        time.sleep(3 * BATCH_S)
        try:
            asyncio.run_coroutine_threadsafe(self._close_sockets(), self._loop).result(timeout=2)
        except Exception:  # noqa: BLE001 - shutting down regardless
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    # --- the server thread -----------------------------------------------------------------

    def _main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        app = web.Application(client_max_size=4 * 1024**2)
        app.router.add_get("/", self._index)
        app.router.add_get("/ws", self._websocket)
        app.router.add_get("/api/catalog", self._catalog)
        app.router.add_post("/api/series", self._series)
        app.router.add_get("/api/layout", self._get_layout)
        app.router.add_put("/api/layout", self._put_layout)
        app.router.add_static("/static", STATIC_DIR)
        runner = web.AppRunner(app, access_log=None, shutdown_timeout=0.5)
        try:
            loop.run_until_complete(runner.setup())
            loop.run_until_complete(web.TCPSite(runner, self.host, self.port).start())
            self.port = runner.addresses[0][1]  # the real one when asked for port 0
        except BaseException as e:  # e.g. the port is taken: report it from start()
            self._error = e
            self._ready.set()
            return
        self._loop = loop
        self._ready.set()
        pump = loop.create_task(self._pump())
        try:
            loop.run_forever()
        finally:
            pump.cancel()
            loop.run_until_complete(runner.cleanup())
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for task in pending:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

    async def _pump(self) -> None:
        while True:
            await asyncio.sleep(BATCH_S)
            batch, series, new_names = [], [], []
            while True:
                try:
                    event = _plain(self._incoming.get_nowait())
                except queue.Empty:
                    break
                values, new = self._take(event)
                batch.append(event)
                new_names += new
                if values:
                    series.append((event["t"], values))
            if not batch or not self._browsers:
                continue
            events_json = _dumps(batch)
            for browser in list(self._browsers):
                mine: dict[str, list] = {}
                for t, values in series:
                    for name in browser.signals.intersection(values):
                        mine.setdefault(name, []).append([t, values[name]])
                message = (f'{{"kind":"events","events":{events_json},"series":{_dumps(_plain(mine))},'
                           f'"new_signals":{_dumps(new_names)}}}')
                if browser.backlog.full():
                    browser.backlog.get_nowait()  # drop the oldest batch for a browser that can't keep up
                browser.backlog.put_nowait(message)

    def _take(self, event: dict) -> tuple[dict, list[str]]:
        """Remember an event (store, serial log tail, newest per topic)."""
        topic = event["topic"]
        if topic == "run":  # a new run starts from a clean slate
            self.store = SignalStore()
            self._uart.clear()
            self._latest.clear()
        if topic == "uart":
            self._uart.append(event)
        else:
            self._latest[topic] = event
        if topic in ("run", "end"):
            return {}, []
        return self.store.add(event)

    def _snapshot(self) -> dict:
        events = [self._latest["run"]] if "run" in self._latest else []
        events += [e for topic, e in self._latest.items() if topic not in ("run", "end")]
        events += list(self._uart)
        if "end" in self._latest:
            events.append(self._latest["end"])
        return {"kind": "snapshot", "events": events, "signals": self.store.names()}

    # --- HTTP ------------------------------------------------------------------------------

    async def _index(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(STATIC_DIR / "index.html")

    async def _catalog(self, request: web.Request) -> web.Response:
        return web.json_response({"signals": self.store.names(), "span": self.store.span()})

    async def _series(self, request: web.Request) -> web.Response:
        """``{"names": [...], "t0": s, "t1": s, "points": n}`` (times on the run's clock) ->
        ``{name: {"t": [...], "v": [...]}}``."""
        body = await request.json()
        t0, t1 = float(body["t0"]), float(body["t1"])
        points = int(min(max(int(body.get("points", 1000)), 10), MAX_POINTS))
        result = {}
        for name in body.get("names", [])[:50]:
            t, v = self.store.series(name, t0, t1, points)
            result[name] = {"t": t, "v": v}
        return web.Response(text=_dumps(result), content_type="application/json")

    async def _get_layout(self, request: web.Request) -> web.Response:
        if not self.layout_file.is_file():
            return web.json_response(None)
        return web.Response(text=self.layout_file.read_text(), content_type="application/json")

    async def _put_layout(self, request: web.Request) -> web.Response:
        layout = await request.json()
        self.layout_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.layout_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(layout, indent=1))
        tmp.replace(self.layout_file)
        return web.json_response({"saved": True})

    # --- WebSocket -------------------------------------------------------------------------

    async def _close_sockets(self) -> None:
        for browser in list(self._browsers):
            await browser.ws.close(message=b"run ended")

    async def _websocket(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=10.0)
        await ws.prepare(request)
        await ws.send_str(_dumps(self._snapshot()))
        browser = _Browser(ws, asyncio.Queue(maxsize=CLIENT_BACKLOG))
        self._browsers.append(browser)
        sender = asyncio.create_task(self._send(ws, browser.backlog))
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    request_ = json.loads(msg.data)
                    if request_.get("kind") == "subscribe":  # the signals this browser plots
                        browser.signals = set(request_.get("signals", []))
                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            self._browsers.remove(browser)
            sender.cancel()
        return ws

    @staticmethod
    async def _send(ws: web.WebSocketResponse, backlog: asyncio.Queue) -> None:
        try:
            while True:
                await ws.send_str(await backlog.get())
        except (ConnectionResetError, RuntimeError):
            pass
