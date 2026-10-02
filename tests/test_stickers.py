import numpy as np
import pytest

from nssim import paths
from nssim.sim.scene import STICKER_GLYPHS, _scaled_symbol

pytestmark = pytest.mark.sim

# The glyphs' widths on DJI's reference stickers, as fractions of the sticker's width. Not used to
# place them (their heights are): they check that TR's glyphs have DJI's shapes.
DJI_WIDTHS = {"infantry": 0.480, "sentry": 0.715, "hero": 0.193}


@pytest.fixture(scope="module")
def assets():
    from nssim.sim.assets import TrAssets
    from nssim.sim.robot_constants import load_plate_dims

    try:
        return TrAssets(plate_dims=load_plate_dims(paths.cv_dir()))
    except FileNotFoundError as e:
        pytest.skip(str(e))


@pytest.mark.parametrize("kind", sorted(STICKER_GLYPHS))
def test_glyph_sits_on_the_panel_as_on_djis_sticker(assets, kind):
    scale = np.asarray(assets.plate_geometry(kind).scale)
    face_min, face_max = (b * scale[:2] for b in assets.panel_face(kind))
    size = face_max - face_min
    v = _scaled_symbol(assets, kind, scale)[0].astype(np.float64)
    lo, hi = v[:, :2].min(axis=0), v[:, :2].max(axis=0)
    glyph = STICKER_GLYPHS[kind]
    assert (hi[1] - lo[1]) / size[1] == pytest.approx(glyph.height, abs=1e-4)
    assert ((lo + hi) / 2 - (face_min + face_max) / 2) / size == pytest.approx(glyph.offset, abs=1e-4)
    assert (hi[0] - lo[0]) / size[0] == pytest.approx(DJI_WIDTHS[kind], rel=0.03)
