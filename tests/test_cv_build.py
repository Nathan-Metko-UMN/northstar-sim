from nssim import cv_build


def _image(monkeypatch, fingerprint):
    monkeypatch.setattr(cv_build, "image_fingerprint", lambda image: fingerprint)


def test_build_dir_is_kept_for_the_image_that_configured_it(tmp_path, monkeypatch):
    build_dir = tmp_path / "sim-jetpack6"
    _image(monkeypatch, "cuda129")
    cv_build._match_build_dir(build_dir, "northstar-cv:sim-jetpack6")
    (build_dir / "CMakeCache.txt").write_text("CUDA 12.9")
    cv_build._match_build_dir(build_dir, "northstar-cv:sim-jetpack6")
    assert (build_dir / "CMakeCache.txt").read_text() == "CUDA 12.9"


def test_build_dir_starts_afresh_in_another_image(tmp_path, monkeypatch):
    build_dir = tmp_path / "sim-jetpack6"
    _image(monkeypatch, "cuda126")
    cv_build._match_build_dir(build_dir, "northstar-cv:sim-jetpack6")
    (build_dir / "CMakeCache.txt").write_text("CUDA 12.6")
    _image(monkeypatch, "cuda129")
    cv_build._match_build_dir(build_dir, "northstar-cv:sim-jetpack6")
    assert not (build_dir / "CMakeCache.txt").exists()
    assert (build_dir / ".nssim-image").read_text().strip() == "cuda129"


def test_build_dir_from_before_the_stamps_starts_afresh(tmp_path, monkeypatch):
    build_dir = tmp_path / "sim-jetpack6"
    build_dir.mkdir()
    (build_dir / "CMakeCache.txt").write_text("unknown image")
    _image(monkeypatch, "cuda129")
    cv_build._match_build_dir(build_dir, "northstar-cv:sim-jetpack6")
    assert not (build_dir / "CMakeCache.txt").exists()
