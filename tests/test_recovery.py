"""Model-guided recovery tests (synthetic)."""

import cv2
import numpy as np
import pytest

from camcalib2.calibration import CalibrationResult, CameraModel, ViewObservation
from camcalib2.detection.dot_marker import DotMarkerDetector
from camcalib2.detection.recovery import (RecoveryConfig, make_projector,
                                          recover_checkerboard_corners,
                                          recover_dot_markers)
from camcalib2.patterns.board import MarkerBoard
from camcalib2.patterns.dotcode import SLOT_PITCH_DEG, encode


def make_pinhole_result(K, dist, size):
    return CalibrationResult(
        model=CameraModel.PINHOLE, image_size=size, camera_matrix=K,
        dist_coeffs=np.asarray(dist, np.float64), rms=0.1, per_view_rms=[],
        n_views=1, n_points=0)


def render_marker(img, cx, cy, r, marker_id, max_ring_dots=8):
    cv2.circle(img, (int(round(cx)), int(round(cy))), r, 0, -1, cv2.LINE_AA)
    bits = encode(marker_id)
    drawn = 0
    for slot, bit in enumerate(bits):
        if not bit or drawn >= max_ring_dots:
            continue
        ang = np.deg2rad(slot * SLOT_PITCH_DEG)
        x = cx + 1.8 * r * np.cos(ang)
        y = cy + 1.8 * r * np.sin(ang)
        cv2.circle(img, (int(round(x)), int(round(y))), max(2, int(r * 0.19) + 1),
                   0, -1, cv2.LINE_AA)
        drawn += 1


def test_recover_dot_markers():
    board = MarkerBoard.builtin()
    ids = sorted(board.markers)
    size = (1600, 1300)
    K = np.array([[900.0, 0, 800.0], [0, 900.0, 650.0], [0, 0, 1]])
    dist = np.zeros(5)
    rvec = np.zeros((3, 1))
    tvec = np.array([[-560.0], [-430.0], [900.0]])

    obj = board.object_points(ids)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)

    img = np.full((size[1], size[0]), 210, np.uint8)
    degraded = set(ids[::9])  # every 9th marker: only 3 ring dots -> undecodable
    for mid, (x, y) in zip(ids, proj):
        if not (40 < x < size[0] - 40 and 40 < y < size[1] - 40):
            continue
        render_marker(img, x, y, 11, mid,
                      max_ring_dots=3 if mid in degraded else 8)

    det = DotMarkerDetector(board)
    markers = det.detect(img)
    found = {m.marker_id for m in markers}
    missing_degraded = degraded & set(ids) - found
    assert len(missing_degraded) >= 10  # degraded markers were not decodable

    view = ViewObservation(
        np.array([m.center for m in markers], np.float32),
        board.object_points([m.marker_id for m in markers]),
        marker_ids=[m.marker_id for m in markers])
    result = make_pinhole_result(K, dist, size)
    projector = make_projector(result, view)
    view2, n = recover_dot_markers(img, board, view, projector)
    assert n >= 0.8 * len(missing_degraded)

    # recovered centers must be accurate
    truth = {mid: p for mid, p in zip(ids, proj)}
    for mid, pt in zip(view2.marker_ids[len(view):], view2.image_points[len(view):]):
        err = np.hypot(pt[0] - truth[mid][0], pt[1] - truth[mid][1])
        assert err < 0.7


def test_recover_checkerboard_corners():
    size = (1400, 1100)
    K = np.array([[800.0, 0, 700.0], [0, 800.0, 550.0], [0, 0, 1]])
    dist = np.zeros(5)
    rvec = np.array([[0.08], [-0.06], [0.03]])
    tvec = np.array([[-430.0], [-350.0], [1100.0]])
    s = 70.0
    cols = rows = 12  # inner corner grid 11x11

    # render the checkerboard
    img = np.full((size[1], size[0]), 235, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2:
                continue
            sq = np.array([[c * s, r * s, 0], [(c + 1) * s, r * s, 0],
                           [(c + 1) * s, (r + 1) * s, 0], [c * s, (r + 1) * s, 0]],
                          np.float64) - (cols * s / 2, rows * s / 2, 0)
            p, _ = cv2.projectPoints(sq, rvec, tvec, K, dist)
            cv2.fillConvexPoly(img, np.round(p.reshape(-1, 2)).astype(np.int32), 15,
                               cv2.LINE_AA)

    # inner corners ground truth (11x11), object grid (col*s, row*s)
    gt_obj, gt_px = [], []
    for r in range(1, rows):
        for c in range(1, cols):
            o = np.array([[c * s, r * s, 0.0]]) - (cols * s / 2, rows * s / 2, 0)
            p, _ = cv2.projectPoints(o, rvec, tvec, K, dist)
            gt_obj.append(((c - 1) * s, (r - 1) * s, 0.0))  # detector-style local grid
            gt_px.append(p.reshape(2))
    gt_obj = np.array(gt_obj, np.float32)
    gt_px = np.array(gt_px, np.float32)

    # simulate a partial detection: drop the two outermost rings of corners
    inner = ((gt_obj[:, 0] >= 2 * s) & (gt_obj[:, 0] <= 8 * s)
             & (gt_obj[:, 1] >= 2 * s) & (gt_obj[:, 1] <= 8 * s))
    view = ViewObservation(gt_px[inner], gt_obj[inner])
    dropped = int((~inner).sum())

    result = make_pinhole_result(K, dist, size)
    projector = make_projector(result, view)
    view2, n = recover_checkerboard_corners(img, view, projector, s)
    assert n >= 0.9 * dropped

    # recovered corners must be sub-pixel accurate against ground truth
    gt_lookup = {(round(o[0], 1), round(o[1], 1)): p for o, p in zip(gt_obj, gt_px)}
    added_obj = view2.object_points[len(view):]
    added_px = view2.image_points[len(view):]
    errs = []
    for o, p in zip(added_obj, added_px):
        key = (round(float(o[0]), 1), round(float(o[1]), 1))
        if key in gt_lookup:
            g = gt_lookup[key]
            errs.append(float(np.hypot(p[0] - g[0], p[1] - g[1])))
    assert errs and np.median(errs) < 0.5
