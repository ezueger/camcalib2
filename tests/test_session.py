import numpy as np

from camcalib2.session.coverage import CoverageMap
from camcalib2.session.keyframes import KeyframePolicy, KeyframeSelector


def test_coverage_fraction():
    cm = CoverageMap((1600, 1200), grid=(4, 3))
    assert cm.fraction == 0.0
    pts = np.array([[10, 10], [1590, 1190]])
    assert cm.new_cells(pts) == 2
    assert cm.add_points(pts) == 2
    assert cm.fraction == 2 / 12
    assert cm.add_points(pts) == 0  # already covered


def test_coverage_tilt_bins():
    cm = CoverageMap((100, 100))
    cm.add_tilt(None, 0.0)
    assert cm.tilt_bins[8] == 1
    cm.add_tilt(0.0, 0.5)
    assert cm.tilt_bins[:8].sum() == 1


def test_keyframe_selector_gates():
    pol = KeyframePolicy(min_points=4, min_sharpness=0.0,
                         min_motion_px=10.0, min_new_cells=1, min_interval=0.0)
    sel = KeyframeSelector(pol)
    gray = np.random.default_rng(0).integers(0, 255, (200, 200), np.uint8)
    ids = [1, 2, 3, 4]
    pts = np.array([[20.0, 20], [60, 20], [20, 60], [60, 60]])
    ok, reason = sel.consider(gray, ids, pts, 0.0, new_cells=3)
    assert ok and reason == "new_coverage"
    # same frame again: no new cells, no motion
    ok, reason = sel.consider(gray, ids, pts, 1.0, new_cells=0)
    assert not ok and reason == "static"
    # moved
    ok, reason = sel.consider(gray, ids, pts + 50, 2.0, new_cells=0)
    assert ok and reason == "motion"
    # too few points
    ok, reason = sel.consider(gray, ids[:2], pts[:2], 3.0, new_cells=5)
    assert not ok and reason == "too_few_points"
