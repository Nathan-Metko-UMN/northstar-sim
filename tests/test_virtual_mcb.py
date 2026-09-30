import math

import pytest

from nssim.mcb import McbConfig, TurretState, VirtualMcb, wrap_2pi
from nssim.protocol import (
    Empty,
    FrameDecoder,
    Health,
    MsgType,
    Odometry,
    RobotIdMsg,
    TurretAimData,
    decode,
    encode,
)


def run(mcb: VirtualMcb, start_us: int, end_us: int, step_us: int = 1000, turret=None):
    """Poll the MCB like its main loop at ``step_us`` and decode everything it sends."""
    turret = turret or TurretState()
    decoder = FrameDecoder()
    msgs = []
    for t in range(start_us, end_us, step_us):
        for queued_us, data in mcb.poll_tx(t, turret):
            msgs += [(queued_us, decode(f)) for f in decoder.feed(data)]
    return msgs


def test_silent_during_start_delay():
    mcb = VirtualMcb(McbConfig())
    assert run(mcb, 0, 1_000_000) == []


def test_periodic_rates():
    mcb = VirtualMcb(McbConfig())
    msgs = run(mcb, 1_000_000, 2_100_000)
    odom = [m for _, m in msgs if isinstance(m, Odometry)]
    health = [m for _, m in msgs if isinstance(m, Health)]
    ids = [m for _, m in msgs if isinstance(m, RobotIdMsg)]
    assert len(odom) == pytest.approx(1_100_000 / 4_000, abs=1)
    stamps = [o.timestamp_us for o in odom]
    assert all(b - a == 4_000 for a, b in zip(stamps, stamps[1:]))
    assert len(health) == pytest.approx(1_100_000 / 30_000, abs=1)
    # The 1030 ms robot-ID timer, armed at boot: due at 1.03 s and 2.06 s.
    assert [r.robot_id for r in ids] == [3, 3]


def test_odometry_yaw_is_wrapped_like_mahony():
    mcb = VirtualMcb(McbConfig(boot_offset_us=0))
    turret = TurretState(yaw=-0.25, pitch=0.1, yaw_rate=1.5)
    [(_, odom)] = [m for m in run(mcb, 1_000_000, 1_001_000, turret=turret) if isinstance(m[1], Odometry)]
    assert odom.yaw == pytest.approx(2 * math.pi - 0.25, abs=1e-6)
    assert odom.pitch == pytest.approx(0.1, abs=1e-6)
    assert odom.yaw_vel == pytest.approx(1.5)


def test_wrap_2pi():
    assert wrap_2pi(0.0) == 0.0
    assert wrap_2pi(2 * math.pi) == pytest.approx(0.0)
    assert wrap_2pi(-0.1) == pytest.approx(2 * math.pi - 0.1)
    assert wrap_2pi(7.0) == pytest.approx(7.0 - 2 * math.pi)


def test_robot_id_request_only_answers_when_timer_is_due():
    mcb = VirtualMcb(McbConfig())
    request = encode(Empty(MsgType.ROBOT_ID))
    run(mcb, 1_000_000, 1_031_000)  # the periodic send at 1.03 s consumes the timer
    assert mcb.feed_rx(request, now_us=1_500_000) == []
    [(_, data)] = mcb.feed_rx(request, now_us=2_060_000)
    [frame] = FrameDecoder().feed(data)
    assert decode(frame) == RobotIdMsg(robot_id=3)


def test_alive_keeps_cv_online_for_one_second():
    mcb = VirtualMcb(McbConfig())
    assert not mcb.cv_online(0)
    mcb.feed_rx(encode(Empty(MsgType.ALIVE)), now_us=5_000_000)
    assert mcb.cv_online(5_999_999)
    assert not mcb.cv_online(6_000_000)


def test_aim_decode_and_tolerance():
    mcb = VirtualMcb(McbConfig())
    mcb.feed_rx(encode(TurretAimData(yaw=0.3, pitch=0.05, distance=5.0, target_id=3)), 5_000_000)
    state = mcb.aim_state
    assert state.updated
    assert state.max_error_yaw == pytest.approx(1.5 * math.atan(0.075 / 5.0))
    assert mcb.within_aiming_tolerance(0.3 + 0.01, 0.05)
    assert not mcb.within_aiming_tolerance(0.3 + 0.05, 0.05)


def test_aim_tolerance_handles_wraparound():
    mcb = VirtualMcb(McbConfig())
    mcb.feed_rx(encode(TurretAimData(yaw=0.005, pitch=0.0 + 1e-3, distance=3.0, target_id=1)), 0)
    assert mcb.within_aiming_tolerance(2 * math.pi - 0.005, 1e-3)


def test_zero_aim_means_no_target_and_unknown_id_keeps_tolerance():
    mcb = VirtualMcb(McbConfig())
    mcb.feed_rx(encode(TurretAimData(yaw=0.2, pitch=0.1, distance=4.0, target_id=7)), 0)
    tol = mcb.aim_state.max_error_yaw
    mcb.feed_rx(encode(TurretAimData(yaw=0.2, pitch=0.1, distance=2.0, target_id=4)), 1000)
    assert mcb.aim_state.max_error_yaw == tol  # ID 4 is not in the firmware's plate table
    mcb.feed_rx(encode(TurretAimData(0.0, 0.0, 0.0, 0)), 2000)
    assert not mcb.aim_state.updated
    assert not mcb.within_aiming_tolerance(0.0, 0.0)
