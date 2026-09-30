"""Live dashboard: streams the run's events (see bus.py) to browsers.

A small aiohttp server in its own thread with its own event loop, so the paced loop only ever does a
non-blocking queue put. Browsers load the page from ``http://<this machine>:<port>`` and open a
WebSocket: they get the run info and recent history first, then batches of new events about 20
times a second. A browser that falls behind loses its oldest batches instead of holding anything up.
"""

from __future__ import annotations

import asyncio
import json
import math
import queue
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web

STATIC_DIR = Path(__file__).parent / "static"
HISTORY_S = 120.0  # how far back a browser that connects mid-run can see the plots
HISTORY_TOPICS = ("metrics", "frame", "truth")
HISTORY_MAX_EVENTS = 3000  # per topic in the snapshot (evenly thinned beyond that)
UART_TAIL = 400  # messages a new browser gets for its log
UART_PLOTTED = {"ODOMETRY", "TURRET_AIM_DATA"}  # also kept further back (thinned) for the plots
UART_PLOT_HZ = 50.0
BATCH_S = 0.05
CLIENT_BACKLOG = 100  # batches queued per browser before the oldest are dropped


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


class DashboardServer:
    """A bus sink (call it with events) that serves them to browsers."""

    def __init__(self, port: int = 8050, host: str = "0.0.0.0"):
        self.host = host
        self.port = port
        self._incoming: queue.SimpleQueue = queue.SimpleQueue()
        self._history: dict[str, deque] = {topic: deque() for topic in HISTORY_TOPICS}
        self._uart: deque = deque(maxlen=UART_TAIL)
        self._uart_history: deque = deque()  # plotted message types, thinned to UART_PLOT_HZ
        self._uart_kept_t: dict[str, float] = {}
        self._latest: dict[str, dict] = {}  # newest event of every other topic (run, cv.track, ...)
        self._clients: set[asyncio.Queue] = set()
        self._sockets: set[web.WebSocketResponse] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None

    # --- the sim side ----------------------------------------------------------------------

    def __call__(self, event: dict) -> None:
        event["wall"] = time.time()  # lets the page work out the real-time factor
        self._incoming.put(event)

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
        app = web.Application()
        app.router.add_get("/", self._index)
        app.router.add_get("/ws", self._websocket)
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

    async def _close_sockets(self) -> None:
        for ws in list(self._sockets):
            await ws.close(message=b"run ended")

    async def _index(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(STATIC_DIR / "index.html")

    async def _pump(self) -> None:
        while True:
            await asyncio.sleep(BATCH_S)
            batch = []
            while True:
                try:
                    event = _plain(self._incoming.get_nowait())
                except queue.Empty:
                    break
                self._remember(event)
                batch.append(event)
            if batch and self._clients:
                message = _dumps({"kind": "events", "events": batch})
                for backlog in list(self._clients):
                    if backlog.full():
                        backlog.get_nowait()  # drop the oldest batch for a browser that can't keep up
                    backlog.put_nowait(message)

    def _remember(self, event: dict) -> None:
        topic = event["topic"]
        if topic == "run":  # a new run starts from a clean slate
            for history in self._history.values():
                history.clear()
            self._uart.clear()
            self._uart_history.clear()
            self._uart_kept_t.clear()
            self._latest.clear()
        if topic in self._history:
            history = self._history[topic]
            history.append(event)
            while history and history[0]["t"] < event["t"] - HISTORY_S:
                history.popleft()
        elif topic == "uart":
            self._uart.append(event)
            kind = event["data"].get("type")
            if kind in UART_PLOTTED and event["t"] - self._uart_kept_t.get(kind, -1e9) >= 1 / UART_PLOT_HZ:
                self._uart_kept_t[kind] = event["t"]
                self._uart_history.append(event)
                while self._uart_history and self._uart_history[0]["t"] < event["t"] - HISTORY_S:
                    self._uart_history.popleft()
        else:
            self._latest[topic] = event

    def _snapshot(self) -> list[dict]:
        events = [self._latest["run"]] if "run" in self._latest else []
        events += [e for topic, e in self._latest.items() if topic != "run"]
        for history in self._history.values():
            step = max(1, len(history) // HISTORY_MAX_EVENTS)
            events += list(history)[::step]
        # Older plotted messages first (only those from before the log's tail, so none repeat), then
        # the tail.
        tail_start = self._uart[0]["t"] if self._uart else float("inf")
        events += [e for e in self._uart_history if e["t"] < tail_start]
        events += list(self._uart)
        return events

    async def _websocket(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=10.0)
        await ws.prepare(request)
        await ws.send_str(_dumps({"kind": "snapshot", "events": self._snapshot()}))
        backlog: asyncio.Queue = asyncio.Queue(maxsize=CLIENT_BACKLOG)
        self._clients.add(backlog)
        self._sockets.add(ws)
        sender = asyncio.create_task(self._send(ws, backlog))
        try:
            async for msg in ws:  # nothing is expected from the page yet
                if msg.type == WSMsgType.ERROR:
                    break
        finally:
            self._clients.discard(backlog)
            self._sockets.discard(ws)
            sender.cancel()
        return ws

    @staticmethod
    async def _send(ws: web.WebSocketResponse, backlog: asyncio.Queue) -> None:
        try:
            while True:
                await ws.send_str(await backlog.get())
        except (ConnectionResetError, RuntimeError):
            pass
