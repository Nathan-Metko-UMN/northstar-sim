from nssim import paths


def _repo(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(paths, "EXTERNAL", tmp_path / "external")
    (tmp_path / "external").mkdir()
    return tmp_path / "external" / "TR-Simulation-ARCTIC-2026"


def test_hint_for_a_submodule_that_isnt_there(tmp_path, monkeypatch):
    tr = _repo(tmp_path, monkeypatch)
    hint = paths.missing_hint(tr, "TR simulator assets", "TR_SIM_DIR")
    assert "git submodule update --init --recursive" in hint


def test_hint_for_a_submodule_cloned_but_never_checked_out(tmp_path, monkeypatch):
    tr = _repo(tmp_path, monkeypatch)
    tr.mkdir()
    (tr / ".git").write_text("gitdir: ../../.git/modules/external/TR-Simulation-ARCTIC-2026\n")
    hint = paths.missing_hint(tr, "TR simulator assets", "TR_SIM_DIR")
    assert "git submodule update --init --force external/TR-Simulation-ARCTIC-2026" in hint


def test_hint_outside_external_points_at_the_override(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch)
    hint = paths.missing_hint(tmp_path / "elsewhere", "Northstar-CV", "NORTHSTAR_CV_DIR")
    assert "NORTHSTAR_CV_DIR" in hint and "submodule" not in hint
