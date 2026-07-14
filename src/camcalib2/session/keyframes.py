"""Keyframe selection for the live calibration loop.

A video frame becomes a keyframe when it is sharp, has enough
observations, and adds information: new sensor coverage, a new board
tilt, or sufficient motion since the previous keyframe.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class KeyframePolicy:
    min_points: int = 12
    #: minimum Laplacian variance in observation regions (blur gate)
    min_sharpness: float = 25.0
    #: median displacement (px) since last keyframe that counts as "moved"
    min_motion_px: float = 24.0
    #: a frame with at least this many new cells is always interesting
    min_new_cells: int = 2
    #: minimum seconds between keyframes
    min_interval: float = 0.25


class KeyframeSelector:
    def __init__(self, policy: KeyframePolicy | None = None):
        self.policy = policy or KeyframePolicy()
        self._last_points: dict[int, tuple[float, float]] | None = None
        self._last_ts: float = -1e9

    @staticmethod
    def sharpness(gray: np.ndarray, points: np.ndarray, win: int = 24) -> float:
        """Median Laplacian variance in patches around the observations."""
        if len(points) == 0:
            return 0.0
        idx = np.linspace(0, len(points) - 1, min(9, len(points))).astype(int)
        vals = []
        h, w = gray.shape[:2]
        for x, y in np.asarray(points, dtype=int)[idx]:
            x0, y0 = max(0, x - win), max(0, y - win)
            patch = gray[y0:y0 + 2 * win, x0:x0 + 2 * win]
            if patch.size:
                vals.append(cv2.Laplacian(patch, cv2.CV_64F).var())
        return float(np.median(vals)) if vals else 0.0

    def consider(self, gray: np.ndarray, ids: list[int], points: np.ndarray,
                 timestamp: float, new_cells: int) -> tuple[bool, str]:
        """Decide whether this frame should become a keyframe.

        Returns (accepted, reason). ``reason`` describes the rejection or
        the acceptance trigger (useful for UI feedback).
        """
        p = self.policy
        if len(points) < p.min_points:
            return False, "too_few_points"
        if timestamp - self._last_ts < p.min_interval:
            return False, "too_soon"
        sharp = self.sharpness(gray, points)
        if sharp < p.min_sharpness:
            return False, "blurry"

        moved = True
        if self._last_points is not None:
            common = [i for i in ids if i in self._last_points]
            if len(common) >= 4:
                d = [np.hypot(points[ids.index(i)][0] - self._last_points[i][0],
                              points[ids.index(i)][1] - self._last_points[i][1])
                     for i in common]
                moved = float(np.median(d)) >= p.min_motion_px

        if new_cells >= p.min_new_cells or moved:
            self._last_points = {i: tuple(pt) for i, pt in zip(ids, points)}
            self._last_ts = timestamp
            reason = "new_coverage" if new_cells >= p.min_new_cells else "motion"
            return True, reason
        return False, "static"
