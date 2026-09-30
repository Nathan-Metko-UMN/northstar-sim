import threading

import cv2
import numpy as np
import pytest

from nssim.camera import (
    RGGB_TO_BGR,
    CameraModel,
    FrameClient,
    FrameHeader,
    FrameServer,
    LinkModel,
    demosaic_to_bgr,
    mosaic_rggb,
)


def _patch(rgb):
    img = np.zeros((8, 8, 3), np.uint8)
    img[:] = rgb
    return img


@pytest.mark.parametrize("rgb", [(255, 0, 0), (0, 255, 0), (0, 0, 255), (40, 120, 200)])
def test_rggb_roundtrip_preserves_color(rgb):
    bgr = demosaic_to_bgr(mosaic_rggb(_patch(rgb)))
    # Interior pixels are exact for a flat patch.
    assert tuple(bgr[4, 4]) == (rgb[2], rgb[1], rgb[0])


def test_opencv_bayer_name_gotcha():
    # The "obvious" code swaps red and blue for an RGGB sensor; the enemy-color filter
    # would then track the wrong team. RGGB_TO_BGR must stay COLOR_BayerBG2BGR.
    raw = mosaic_rggb(_patch((255, 0, 0)))
    assert tuple(cv2.cvtColor(raw, cv2.COLOR_BayerRG2BGR)[4, 4]) == (255, 0, 0)  # red became blue
    assert RGGB_TO_BGR == cv2.COLOR_BayerBG2BGR


def test_nominal_camera_model():
    cam = CameraModel()
    assert cam.fx == pytest.approx(1739.13, abs=0.01)
    assert cam.cx == 719.5 and cam.cy == 539.5
    assert cam.hfov_deg == pytest.approx(45.0, abs=0.1)
    assert cam.vfov_deg == pytest.approx(34.5, abs=0.1)


def test_link_transfer_time():
    frame_bytes = 1440 * 1080
    assert LinkModel(gbps=2.5).transfer_us(frame_bytes) == pytest.approx(5239, abs=5)
    assert LinkModel(gbps=1.0).transfer_us(frame_bytes) == pytest.approx(13097, abs=5)


@pytest.mark.parametrize("compress", [False, True])
def test_stream_roundtrip(compress):
    server = FrameServer(host="127.0.0.1", port=0)
    image = np.arange(4 * 6, dtype=np.uint8).reshape(4, 6)
    oracle = {"plates": [{"corners": [[1, 2], [3, 4], [5, 6], [7, 8]], "number": "3"}]}

    def serve():
        server.accept({"mode": "paced", "width": 6, "height": 4}, timeout=5)
        header = FrameHeader(
            seq=7,
            capture_mcb_us=5_006_000,
            exposure_us=2000,
            arrival_delay_us=5239,
            width=6,
            height=4,
            odom_watermark_mcb_us=5_008_000,
        )
        server.send_frame(header, image.tobytes(), oracle, compress=compress)
        acks.append(server.recv_ack(timeout=5))

    acks = []
    thread = threading.Thread(target=serve)
    thread.start()
    client = FrameClient("127.0.0.1", server.port)
    assert client.hello["mode"] == "paced" and client.hello["protocol"] == 1
    header, data, got_oracle = client.read_frame()
    assert header.seq == 7 and header.capture_mcb_us == 5_006_000
    assert header.flags == (1 | 2 if compress else 1)
    assert header.odom_watermark_mcb_us == 5_008_000 and header.arrival_delay_us == 5239
    assert np.array_equal(np.frombuffer(data, np.uint8).reshape(4, 6), image)
    assert got_oracle == oracle
    client.ack(7, 1234)
    thread.join(timeout=5)
    client.close()
    server.close()
    assert acks == [(7, 1234)]
