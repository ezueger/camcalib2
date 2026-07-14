"""Round-trip calibration tests on synthetic observations."""

import cv2
import numpy as np
import pytest

from camcalib2.calibration import CameraModel, ViewObservation, calibrate
from camcalib2.patterns.board import MarkerBoard


@pytest.fixture(scope="module")
def board():
    return MarkerBoard.builtin()


def synth_views_pinhole(board, K, dist, image_size, n_views=12, seed=7):
    rng = np.random.default_rng(seed)
    ids = sorted(board.markers)
    obj_all = board.object_points(ids)
    center = obj_all.mean(axis=0)
    views = []
    w, h = image_size
    for i in range(n_views):
        ang = rng.uniform(-0.5, 0.5, size=3)
        rvec = ang.reshape(3, 1)
        tvec = np.array([[rng.uniform(-150, 150)],
                         [rng.uniform(-150, 150)],
                         [rng.uniform(1500, 2600)]])
        obj = obj_all - center
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        pts = proj.reshape(-1, 2)
        inside = ((pts[:, 0] > 10) & (pts[:, 0] < w - 10)
                  & (pts[:, 1] > 10) & (pts[:, 1] < h - 10))
        if inside.sum() < 30:
            continue
        noise = rng.normal(0, 0.1, size=(int(inside.sum()), 2))
        views.append(ViewObservation(pts[inside] + noise, obj[inside]))
    return views


def test_pinhole_roundtrip(board):
    w, h = 1600, 1200
    K = np.array([[1400.0, 0, 810.0], [0, 1398.0, 590.0], [0, 0, 1]])
    dist = np.array([-0.08, 0.11, 0.0004, -0.0003, -0.05])
    views = synth_views_pinhole(board, K, dist, (w, h))
    assert len(views) >= 8
    r = calibrate(views, (w, h), CameraModel.PINHOLE)
    assert r.rms < 0.3
    assert abs(r.fx - 1400) < 2.0
    assert abs(r.fy - 1398) < 2.0
    assert abs(r.cx - 810) < 3.0
    assert abs(r.cy - 590) < 3.0
    assert np.allclose(r.dist_coeffs[:2], dist[:2], atol=0.01)


def synth_views_fisheye(K, D, image_size, n_views=14, seed=3):
    rng = np.random.default_rng(seed)
    # planar 10x10 grid, 40mm squares
    g = np.zeros((100, 3), np.float64)
    g[:, :2] = np.mgrid[0:10, 0:10].T.reshape(-1, 2) * 40.0
    g[:, :2] -= g[:, :2].mean(0)
    w, h = image_size
    views = []
    for i in range(n_views):
        rvec = rng.uniform(-0.6, 0.6, size=(3, 1))
        tvec = np.array([[rng.uniform(-120, 120)],
                         [rng.uniform(-120, 120)],
                         [rng.uniform(350, 800)]])
        proj, _ = cv2.fisheye.projectPoints(g.reshape(1, -1, 3), rvec, tvec, K, D)
        pts = proj.reshape(-1, 2)
        inside = ((pts[:, 0] > 5) & (pts[:, 0] < w - 5)
                  & (pts[:, 1] > 5) & (pts[:, 1] < h - 5))
        if inside.sum() < 40:
            continue
        noise = rng.normal(0, 0.1, size=(int(inside.sum()), 2))
        views.append(ViewObservation(pts[inside] + noise, g[inside]))
    return views


def test_fisheye_roundtrip():
    w, h = 1920, 1600
    K = np.array([[420.0, 0, 950.0], [0, 421.0, 790.0], [0, 0, 1]])
    D = np.array([0.006, 0.001, -0.0006, -0.0003]).reshape(4, 1)
    views = synth_views_fisheye(K, D, (w, h))
    assert len(views) >= 8
    r = calibrate(views, (w, h), CameraModel.FISHEYE)
    assert r.rms < 0.3
    assert abs(r.fx - 420) < 3.0
    assert abs(r.cx - 950) < 4.0
    assert abs(r.cy - 790) < 4.0
