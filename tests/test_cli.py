import argparse

from nssim import cli
from nssim.scenarios import drive


def _args(**kw):
    return argparse.Namespace(**{"plate_height": None, "radius": None, "radius_low": None, **kw})


def test_radius_sets_both_pairs_of_every_enemy():
    cfg = cli._apply_common(drive(enemies=3), _args(radius=0.32))
    assert all(t.spec.radius_high == t.spec.radius_low == 0.32 for t in cfg.targets)


def test_radius_low_changes_only_the_low_pair():
    cfg = cli._apply_common(drive(enemies=1), _args(radius=0.3, radius_low=0.22))
    spec = cfg.targets[0].spec
    assert (spec.radius_high, spec.radius_low) == (0.3, 0.22)


def test_without_options_the_scenario_keeps_its_own_sizes():
    cfg = cli._apply_common(drive(enemies=2), _args())
    assert [t.spec.radius_high for t in cfg.targets] == [0.25, 0.3]
