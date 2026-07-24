import numpy as np

from camcalib2.session.coverage import CoverageMap, Roi
from camcalib2.session.keyframes import KeyframePolicy, KeyframeSelector


def test_coverage_elliptical_excludes_corners():
    """Fisheye: coverage is scored over the inscribed circle, so the black
    square-ROI corners are not counted (full coverage stays reachable)."""
    size = (400, 400)
    roi = Roi.full(size)
    rect = CoverageMap(size, grid=(4, 4), roi=roi)
    ell = CoverageMap(size, grid=(4, 4), roi=roi, elliptical=True)
    assert rect.roi_cell_mask().all()  # rectangle: every cell eligible
    m = ell.roi_cell_mask()
    assert not m[0, 0] and not m[0, -1] and not m[-1, 0] and not m[-1, -1]
    assert m[1, 1] and m[2, 2]  # centre kept
    assert m.sum() < rect.roi_cell_mask().sum()
    # a point in an excluded corner cell never adds coverage
    assert ell.new_cells(np.array([[10.0, 10.0]])) == 0
    assert ell.add_points(np.array([[10.0, 10.0]])) == 0


def test_coverage_fraction():
    cm = CoverageMap((1600, 1200), grid=(4, 3))
    assert cm.fraction == 0.0
    pts = np.array([[10, 10], [1590, 1190]])
    assert cm.new_cells(pts) == 2
    assert cm.add_points(pts) == 2
    assert cm.fraction == 2 / 12
    assert cm.add_points(pts) == 0  # already covered


def test_coverage_densification_runs():
    """A new run re-opens coverage novelty (so re-scanning adds keyframes
    again) while keeping the cumulative per-cell density."""
    cm = CoverageMap((400, 400), grid=(4, 4))
    pt = np.array([[50.0, 50.0]])  # one cell
    assert cm.new_cells(pt) == 1
    assert cm.add_points(pt) == 1
    assert cm.new_cells(pt) == 0          # already covered THIS run
    assert cm.fraction == 1 / 16
    # veil: covered-this-run cell is clear, an untouched cell is red
    veil = cm.veil_alpha()
    assert veil[0, 0] == 0.0 and veil[3, 3] > 0.0

    cm.start_run()
    assert cm.new_cells(pt) == 1          # fresh run -> countable again
    assert cm.add_points(pt) == 1
    assert cm.counts[0, 0] == 2           # cumulative density kept
    assert cm.run_counts[0, 0] == 1       # but per-run reset
    # global novelty ignores runs: an already-covered cell never counts,
    # a never-touched one still does (used to bypass the keyframe cap)
    assert cm.new_cells_global(pt) == 0
    assert cm.new_cells_global(np.array([[350.0, 350.0]])) == 1


def test_coverage_tilt_bins():
    cm = CoverageMap((100, 100))
    cm.add_tilt(None, 0.0)
    assert cm.tilt_bins[8] == 1
    cm.add_tilt(0.0, 0.5)
    assert cm.tilt_bins[:8].sum() == 1


def test_fisheye_uses_lower_coverage_target():
    """Fisheye converges on a lower, reachable coverage target (image
    circle minus the hard rim ring); pinhole keeps the full 0.85."""
    from camcalib2.calibration import CameraModel
    from camcalib2.patterns.board import Checkerboard
    from camcalib2.session import CalibrationSession, SessionConfig

    board = Checkerboard((10, 10), 40.0)
    size = (2448, 2048)
    fish = CalibrationSession(board, size, SessionConfig(model=CameraModel.FISHEYE))
    pin = CalibrationSession(board, size, SessionConfig(model=CameraModel.PINHOLE))
    assert fish._target_coverage() == fish.cfg.fisheye_target_coverage < 0.85
    assert pin._target_coverage() == 0.85
    assert fish.coverage.elliptical and not pin.coverage.elliptical


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


def test_keyframe_partial_view_staffelung():
    """Adaptive staffelung: below ``full_points`` a view is partial and
    only becomes a keyframe by adding new coverage; at/above it the view
    is (near-)complete and motion alone suffices."""
    pol = KeyframePolicy(min_points=4, min_sharpness=0.0,
                         min_motion_px=10.0, min_new_cells=1, min_interval=0.0)
    gray = np.random.default_rng(0).integers(0, 255, (200, 200), np.uint8)
    ids = [1, 2, 3, 4, 5, 6]
    base = np.array([[20.0, 20], [60, 20], [100, 20],
                     [20, 60], [60, 60], [100, 60]])

    # --- partial view (4 of 6 points, floor=4 <= 4 < full=6)
    sel = KeyframeSelector(pol)
    ok, reason = sel.consider(gray, ids[:4], base[:4], 0.0, new_cells=2,
                              min_points=4, full_points=6)
    assert ok and reason == "new_coverage"  # earns its place by coverage
    # partial view that only moved (no new coverage) is rejected
    ok, reason = sel.consider(gray, ids[:4], base[:4] + 50, 1.0, new_cells=0,
                              min_points=4, full_points=6)
    assert not ok and reason == "partial_static"

    # --- (near-)complete view (6 points >= full) is accepted on motion
    sel = KeyframeSelector(pol)
    ok, _ = sel.consider(gray, ids, base, 0.0, new_cells=2,
                         min_points=4, full_points=6)
    assert ok
    ok, reason = sel.consider(gray, ids, base + 50, 1.0, new_cells=0,
                              min_points=4, full_points=6)
    assert ok and reason == "motion"

    # --- bootstrap phase: floor == full, so a partial board is below the
    #     floor and rejected outright (the first solve needs full boards)
    sel = KeyframeSelector(pol)
    ok, reason = sel.consider(gray, ids[:4], base[:4], 0.0, new_cells=2,
                              min_points=6, full_points=6)
    assert not ok and reason == "too_few_points"


def test_keyframe_require_novelty():
    """Once a model exists (require_novelty), a complete view that only
    moved is rejected; a new coverage cell or a new tilt still gets in."""
    pol = KeyframePolicy(min_points=4, min_sharpness=0.0,
                         min_motion_px=10.0, min_new_cells=1, min_interval=0.0)
    gray = np.random.default_rng(0).integers(0, 255, (200, 200), np.uint8)
    ids = [1, 2, 3, 4, 5, 6]
    base = np.array([[20.0, 20], [60, 20], [100, 20],
                     [20, 60], [60, 60], [100, 60]])

    sel = KeyframeSelector(pol)
    ok, _ = sel.consider(gray, ids, base, 0.0, new_cells=2, require_novelty=True)
    assert ok
    # moved but no new coverage and no new tilt -> rejected (was "motion")
    ok, reason = sel.consider(gray, ids, base + 50, 1.0, new_cells=0,
                              require_novelty=True)
    assert not ok and reason == "static"
    # a new tilt direction still earns the keyframe
    ok, reason = sel.consider(gray, ids, base + 80, 2.0, new_cells=0,
                              require_novelty=True, extra_novelty=True)
    assert ok and reason == "new_tilt"
