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
    #: absolute floor: a frame with fewer observations never becomes a
    #: keyframe. This is the *partial-view* floor used once a preliminary
    #: model exists (see ``min_points_bootstrap``).
    min_points: int = 6
    #: near-complete board required until the first model exists - the
    #: bootstrap solve must be well-conditioned and cannot yet be
    #: sanity-checked (no model to reproject against). It also classifies
    #: partial vs (near-)complete views afterwards: a view below this count
    #: only earns a keyframe by adding new coverage (see ``consider``).
    min_points_bootstrap: int = 12
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
                 timestamp: float, new_cells: int, *,
                 min_points: int | None = None,
                 full_points: int | None = None,
                 require_novelty: bool = False,
                 extra_novelty: bool = False) -> tuple[bool, str]:
        """Decide whether this frame should become a keyframe.

        ``min_points`` is the adaptive floor for *this* frame (fewer
        observations never become a keyframe); ``full_points`` is the count
        at/above which the board counts as (near-)complete. A *partial*
        view (``min_points <= n < full_points``) only earns a keyframe by
        adding new information, never on motion alone - this is what lets
        the board run off the sensor edge to free up the periphery without
        flooding the keyframe budget with redundant partial centre views.

        ``require_novelty`` extends that discipline to *every* view: once a
        preliminary model exists, a keyframe must add new sensor coverage
        or a new board tilt (``extra_novelty``); a redundant view that only
        moved is rejected, so the budget stays free for the corners/edges
        that still need covering. During bootstrap (``require_novelty``
        False) a complete view is still accepted on motion, to gather the
        varied views the first solve needs.

        All keyword flags default to the classic behaviour so direct
        callers are unaffected.

        Returns (accepted, reason). ``reason`` describes the rejection or
        the acceptance trigger (useful for UI feedback).
        """
        p = self.policy
        floor = p.min_points if min_points is None else min_points
        full = floor if full_points is None else full_points
        n = len(points)
        if n < floor:
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

        partial = n < full
        adds_coverage = new_cells >= p.min_new_cells
        novel = adds_coverage or extra_novelty
        # motion alone only earns a keyframe for a complete view while we
        # still lack a model; otherwise real novelty (coverage/tilt) is
        # required.
        accepted = novel or (moved and not partial and not require_novelty)
        if accepted:
            self._last_points = {i: tuple(pt) for i, pt in zip(ids, points)}
            self._last_ts = timestamp
            if adds_coverage:
                reason = "new_coverage"
            elif extra_novelty:
                reason = "new_tilt"
            else:
                reason = "motion"
            return True, reason
        return False, "partial_static" if partial else "static"
