"""Still-capture state machine and solver-protection tests."""

import numpy as np
import pytest

from camcalib2.calibration import CalibrationResult, CameraModel
from camcalib2.patterns.board import MarkerBoard
from camcalib2.session import CalibrationSession, SessionConfig
from camcalib2.session.controller import CaptureState


@pytest.fixture()
def session():
    board = MarkerBoard.builtin()
    cfg = SessionConfig(settle_s=0.3, cooldown_s=1.0,
                        still_threshold=1.0, move_threshold=3.0)
    return CalibrationSession(board, (640, 480), cfg)


def still_frame(rng=None, base=None):
    if base is None:
        base = np.full((480, 640), 128, np.uint8)
    noise = np.random.default_rng(0).integers(-2, 3, base.shape, np.int16)
    return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def moving_frame(shift):
    img = np.full((480, 640), 128, np.uint8)
    img[:, shift % 640:(shift % 640) + 60] = 220
    return img


def test_tick_still_then_evaluate(session):
    base = np.full((480, 640), 128, np.uint8)
    t = 0.0
    evaluated = False
    for i in range(30):
        tick = session.tick(still_frame(base=base), t)
        t += 0.1
        if tick.evaluate:
            evaluated = True
            break
    assert evaluated, "a still camera must trigger an evaluation"
    assert tick.capture_state is CaptureState.EVALUATING


def test_tick_moving_never_evaluates(session):
    t = 0.0
    for i in range(40):
        tick = session.tick(moving_frame(i * 25), t)
        t += 0.05
        assert not tick.evaluate, "no CV while the camera moves"
        if i > 3:
            assert tick.capture_state in (CaptureState.HOLD, CaptureState.MOVE)


def test_no_recapture_without_motion(session):
    """After a capture the camera MUST move before the next evaluation -
    a resting camera must not flood the system with identical points."""
    base = np.full((480, 640), 128, np.uint8)
    t = 0.0
    # drive to first evaluation
    for _ in range(30):
        tick = session.tick(still_frame(base=base), t)
        t += 0.1
        if tick.evaluate:
            break
    assert tick.evaluate
    # simulate: evaluation captured a keyframe
    session._cooldown_until = t + 1.0
    session._capture_state = CaptureState.CAPTURED
    session._moved_since_capture = False
    # camera keeps resting: cooldown, then MOVE state - never evaluate
    for _ in range(60):
        tick = session.tick(still_frame(base=base), t)
        t += 0.1
        assert not tick.evaluate
    assert tick.capture_state is CaptureState.MOVE
    # now the camera moves ...
    for i in range(6):
        tick = session.tick(moving_frame(i * 40), t)
        t += 0.1
    # ... and settles again -> evaluation re-arms
    evaluated = False
    for _ in range(30):
        tick = session.tick(still_frame(base=base), t)
        t += 0.1
        if tick.evaluate:
            evaluated = True
            break
    assert evaluated


def test_max_keyframes_stops_capture(session):
    session.cfg.max_keyframes = 2
    from camcalib2.calibration import ViewObservation
    pts = np.array([[10.0, 10], [50, 10], [10, 50], [50, 50], [90, 90], [90, 10]], np.float32)
    obj = np.zeros((6, 3), np.float32)
    obj[:, :2] = pts
    session.views.extend([ViewObservation(pts, obj), ViewObservation(pts, obj)])
    tick = session.tick(still_frame(), 0.0)
    assert tick.capture_state is CaptureState.FULL
    assert not tick.evaluate


def test_view_sane_gate(session):
    K = np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1]])
    result = CalibrationResult(
        model=CameraModel.PINHOLE, image_size=(640, 480), camera_matrix=K,
        dist_coeffs=np.zeros(5), rms=0.2, per_view_rms=[], n_views=5, n_points=0)
    import cv2
    obj = np.zeros((25, 3), np.float32)
    obj[:, :2] = np.mgrid[0:5, 0:5].T.reshape(-1, 2) * 40.0
    rvec = np.array([[0.1], [0.05], [0.0]])
    tvec = np.array([[-80.0], [-80.0], [600.0]])
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, np.zeros(5))
    pts = proj.reshape(-1, 2).astype(np.float32)
    ids = list(range(25))
    assert session._view_sane(ids, pts, obj, result)  # consistent view

    # toxic view: half the points shifted by one grid cell
    bad = pts.copy()
    bad[::2] += 60.0
    assert not session._view_sane(ids, bad, obj, result)
