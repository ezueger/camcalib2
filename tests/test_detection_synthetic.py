"""Detector test on synthetically rendered marker boards."""

import cv2
import numpy as np
import pytest

from camcalib2.patterns.board import MarkerBoard
from camcalib2.patterns.dotcode import N_SLOTS, SLOT_PITCH_DEG, encode
from camcalib2.detection.dot_marker import DotMarkerDetector


def render_marker(img, cx, cy, r_center, marker_id, rotation_deg=0.0):
    ring_r = 1.75 * r_center
    dot_r = max(2, int(round(r_center * 0.22)))
    cv2.circle(img, (int(round(cx)), int(round(cy))), int(round(r_center)), 0, -1,
               cv2.LINE_AA)
    bits = encode(marker_id)
    for slot, bit in enumerate(bits):
        if not bit:
            continue
        ang = np.deg2rad(slot * SLOT_PITCH_DEG + rotation_deg)
        x = cx + ring_r * np.cos(ang)
        y = cy + ring_r * np.sin(ang)
        cv2.circle(img, (int(round(x)), int(round(y))), dot_r, 0, -1, cv2.LINE_AA)


@pytest.fixture(scope="module")
def board():
    return MarkerBoard.builtin()


def make_image(board, ids, rotation_deg=0.0, spacing=90, r_center=11):
    n = int(np.ceil(np.sqrt(len(ids))))
    size = spacing * (n + 1)
    img = np.full((size, size), 220, np.uint8)
    truth = {}
    for k, mid in enumerate(ids):
        cx = spacing * (1 + k % n)
        cy = spacing * (1 + k // n)
        render_marker(img, cx, cy, r_center, mid, rotation_deg)
        truth[mid] = (cx, cy)
    return img, truth


def test_detect_synthetic_board(board):
    ids = sorted(board.markers)[:36]
    img, truth = make_image(board, ids)
    det = DotMarkerDetector(board)
    found = {m.marker_id: m.center for m in det.detect(img)}
    hits = set(found) & set(truth)
    assert len(hits) >= 0.9 * len(ids)
    errs = [np.hypot(found[m][0] - truth[m][0], found[m][1] - truth[m][1])
            for m in hits]
    assert np.median(errs) < 0.5


@pytest.mark.parametrize("rotation", [0, 45, 117, 200, 302])
def test_detect_rotated_markers(board, rotation):
    """The board may be viewed under any in-plane rotation."""
    ids = sorted(board.markers)[40:60]
    img, truth = make_image(board, ids, rotation_deg=rotation)
    det = DotMarkerDetector(board)
    found = {m.marker_id for m in det.detect(img)}
    assert len(found & set(truth)) >= 0.85 * len(ids)
    # no misidentifications
    assert not (found - set(truth))
