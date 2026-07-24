"""Checkerboard corner detection (used e.g. for fisheye lenses).

Uses OpenCV's sector-based detector (findChessboardCornersSB) with the
``CALIB_CB_LARGER`` meta variant: a small seed grid is grown to the
largest visible grid and every corner is labelled with its (row, col)
board index.  This makes *partially visible* boards usable, which is
essential for video-based scanning where the board often extends beyond
the image or the fisheye circle.

For intrinsics-only calibration the absolute board offset of a partial
view is irrelevant (it is absorbed by the per-view extrinsics), so the
returned object points are simply ``(col, row) * square_size``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..patterns.board import Checkerboard


@dataclass
class CheckerboardDetection:
    #: (N,2) sub-pixel corner positions
    image_points: np.ndarray
    #: (N,3) matching board coordinates (mm), origin at the visible grid corner
    object_points: np.ndarray
    #: found grid extent (rows, cols)
    grid_shape: tuple[int, int]


class CheckerboardDetector:
    def __init__(self, board: Checkerboard, seed: tuple[int, int] = (5, 5),
                 min_corners: int = 8, fast: bool = False):
        """``fast=True`` skips the EXHAUSTIVE/ACCURACY passes - roughly
        6x faster, finds nearly the same grid. Use for the live preview;
        keyframes that feed the calibration should use the full mode.

        ``seed`` is the smallest grid ``findChessboardCornersSB`` will look
        for; with ``CALIB_CB_LARGER`` it grows to the largest visible grid,
        but it returns *nothing* below ``seed`` corners. It therefore sets
        the real partial-view floor. It is deliberately kept at (5,5):
        smaller seeds do enable thin edge slivers, but they also make the
        detector hand small/inconsistent partial grids to the fragile
        fisheye KB solver, which then fails to bootstrap a model at all
        (measured: seed (4,4) leaves the session stuck pre-model). Partial
        rim coverage for fisheye is instead handled by the model-guided
        recovery / OCam refinement, not by shrinking the seed here.

        ``min_corners`` is an extra floor on top of the seed. The adaptive
        keyframe staffelung keeps the bootstrap phase on near-complete
        boards (``min_points_bootstrap``); only once a model exists do
        smaller grids become keyframes, guarded by ``_view_sane``."""
        self.board = board
        self.seed = seed
        self.min_corners = min_corners
        self.fast = fast

    def detect(self, gray: np.ndarray) -> CheckerboardDetection | None:
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_LARGER
        if not self.fast:
            flags |= cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        found, corners, meta = cv2.findChessboardCornersSBWithMeta(
            gray, self.seed, flags=flags)
        if not found or corners is None or len(corners) < self.min_corners:
            return None
        rows, cols = meta.shape
        pts = corners.reshape(-1, 2).astype(np.float32)
        # meta is row-major over the found grid; build matching board coords
        s = self.board.square_size
        rr, cc = np.mgrid[0:rows, 0:cols]
        obj = np.stack([cc.ravel() * s, rr.ravel() * s,
                        np.zeros(rows * cols)], axis=1).astype(np.float32)
        if len(obj) != len(pts):
            return None
        return CheckerboardDetection(image_points=pts, object_points=obj,
                                     grid_shape=(rows, cols))
