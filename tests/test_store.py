import math

from nssim.dashboard.store import SignalStore, flatten


def test_uart_fields_are_named_by_direction_and_type():
    group, values = flatten({"topic": "uart", "t": 1.0, "data": {
        "dir": "mcb_to_cv", "type": "ODOMETRY", "seq": 7, "fields": {"yaw": 3.1, "pitch": 0.05}}})
    assert group == "uart.mcb_to_cv.ODOMETRY"
    assert values == {"uart.mcb_to_cv.ODOMETRY.yaw": 3.1, "uart.mcb_to_cv.ODOMETRY.pitch": 0.05,
                      "uart.mcb_to_cv.ODOMETRY.seq": 7.0}


def test_nested_telemetry_vectors_named_lists_and_booleans():
    _, values = flatten({"topic": "truth", "t": 0.0, "data": {
        "targets": [{"name": "infantry", "center": [1.0, 2.0, 0.2], "omega": 6.0}],
        "ok": True, "label": "text is skipped"}})
    assert values["truth.targets.infantry.center.x"] == 1.0
    assert values["truth.targets.infantry.center.z"] == 0.2
    assert values["truth.targets.infantry.omega"] == 6.0
    assert values["truth.ok"] == 1.0
    assert not any("label" in name for name in values)


def test_per_detection_samples_are_summarized():
    _, values = flatten({"topic": "metrics", "t": 0.0, "data": {"range_m": [0.01, 0.03], "corner_px": []}})
    assert values["metrics.range_m.count"] == 2.0
    assert math.isclose(values["metrics.range_m.mean"], 0.02)


def test_series_over_a_range_keeps_spikes_when_thinned():
    store = SignalStore()
    for i in range(10_000):
        store.add({"topic": "frame", "t": i * 0.001, "data": {"processing_ms": 100.0 if i == 5000 else 3.0}})
    t, v = store.series("frame.processing_ms", 0.0, 10.0, points=200)
    assert len(t) <= 200 and max(v) == 100.0 and min(v) == 3.0
    t, v = store.series("frame.processing_ms", 4.999, 5.001)  # zoomed in: raw points
    assert 100.0 in v and len(v) <= 5


def test_signals_that_appear_later_are_empty_before():
    store = SignalStore()
    store.add({"topic": "metrics", "t": 0.0, "data": {"visible": 2}})
    _, new = store.add({"topic": "metrics", "t": 1.0, "data": {"visible": 2, "filter": {"omega": 5.0}}})
    assert new == ["metrics.filter.omega"]
    t, v = store.series("metrics.filter.omega", 0.0, 1.0)
    assert t == [0.0, 1.0] and v == [None, 5.0]
    assert "metrics.filter.omega" in store.names()
