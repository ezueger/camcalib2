"""Live calibration session: detection -> keyframes -> incremental solve.

The session consumes frames (from a camera, video file or image folder),
selects keyframes, tracks sensor coverage and recalibrates in a
background thread while the user moves the camera around the target -
the same feel as setting up FaceTime.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from ..calibration import CalibrationResult, CameraModel, ViewObservation, calibrate
from ..detection.checkerboard import CheckerboardDetector
from ..detection.dot_marker import DotMarkerDetector
from ..patterns.board import Checkerboard, MarkerBoard
from .coverage import CoverageMap
from .keyframes import KeyframePolicy, KeyframeSelector


class SessionState(str, Enum):
    WAITING = "waiting"  # no target visible yet
    SCANNING = "scanning"  # collecting keyframes
    CONVERGED = "converged"  # intrinsics stable, coverage reached
    FINISHED = "finished"


@dataclass
class SessionConfig:
    model: CameraModel = CameraModel.PINHOLE
    target_coverage: float = 0.85
    target_tilt_fraction: float = 0.5
    min_keyframes: int = 12
    max_keyframes: int = 80
    #: recalibrate every N new keyframes
    recalib_every: int = 4
    #: relative parameter change (fx,fy,cx,cy) considered converged
    convergence_eps: float = 1e-3
    keyframe_policy: KeyframePolicy = field(default_factory=KeyframePolicy)


@dataclass
class FrameFeedback:
    """Everything the UI needs to render one processed frame."""

    ids: list[int]
    points: np.ndarray  # (N,2)
    keyframe: bool
    reason: str
    state: SessionState
    coverage: float
    tilt_coverage: float
    n_keyframes: int
    rms: float | None
    result: CalibrationResult | None
    progress: float  # 0..1 combined progress for the UI ring


class CalibrationSession:
    def __init__(self, target: MarkerBoard | Checkerboard,
                 image_size: tuple[int, int],
                 config: SessionConfig | None = None):
        self.cfg = config or SessionConfig()
        self.image_size = image_size
        self.target = target
        if isinstance(target, MarkerBoard):
            self._detector = DotMarkerDetector(target)
            self._detect = self._detect_markers
        else:
            self._detector = CheckerboardDetector(target)
            self._detect = self._detect_checkerboard

        self.coverage = CoverageMap(image_size)
        self.selector = KeyframeSelector(self.cfg.keyframe_policy)
        self.views: list[ViewObservation] = []
        self.state = SessionState.WAITING

        self._result: CalibrationResult | None = None
        self._prev_params: np.ndarray | None = None
        self._converged_solves = 0
        self._lock = threading.Lock()
        self._solver_busy = threading.Event()
        self._pending_views = 0

    # ------------------------------------------------------------------
    def _detect_markers(self, gray):
        markers = self._detector.detect(gray)
        if not markers:
            return [], np.empty((0, 2), np.float32), None
        ids = [m.marker_id for m in markers]
        pts = np.array([m.center for m in markers], dtype=np.float32)
        obj = self.target.object_points(ids)
        return ids, pts, obj

    def _detect_checkerboard(self, gray):
        det = self._detector.detect(gray)
        if det is None:
            return [], np.empty((0, 2), np.float32), None
        ids = list(range(len(det.image_points)))
        return ids, det.image_points, det.object_points

    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray, timestamp: float) -> FrameFeedback:
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ids, pts, obj = self._detect(gray)

        keyframe = False
        reason = "no_target"
        if len(ids) > 0 and self.state in (SessionState.WAITING, SessionState.SCANNING):
            if self.state is SessionState.WAITING:
                self.state = SessionState.SCANNING
            new_cells = self.coverage.new_cells(pts)
            keyframe, reason = self.selector.consider(gray, ids, pts, timestamp, new_cells)
            if keyframe:
                self._accept_keyframe(ids, pts, obj, timestamp)

        with self._lock:
            result = self._result
        rms = result.rms if result else None
        progress = self._progress()
        if (self.state is SessionState.SCANNING and progress >= 1.0
                and self._converged_solves >= 2):
            self.state = SessionState.CONVERGED

        return FrameFeedback(
            ids=ids, points=pts, keyframe=keyframe, reason=reason,
            state=self.state, coverage=self.coverage.fraction,
            tilt_coverage=self.coverage.tilt_fraction,
            n_keyframes=len(self.views), rms=rms, result=result,
            progress=progress)

    # ------------------------------------------------------------------
    def _accept_keyframe(self, ids, pts, obj, timestamp):
        self.coverage.add_points(pts)
        self.coverage.add_tilt(*self._tilt(pts, obj))
        view = ViewObservation(pts, obj, marker_ids=list(ids), timestamp=timestamp)
        with self._lock:
            self.views.append(view)
            self._pending_views += 1
            n, pending = len(self.views), self._pending_views
        enough = n >= max(4, self.cfg.min_keyframes // 2)
        if enough and pending >= self.cfg.recalib_every and not self._solver_busy.is_set():
            self._start_solver()

    def _tilt(self, pts, obj):
        """Board tilt direction/magnitude from the anisotropy of the
        local affine mapping board->image (cheap, no intrinsics needed)."""
        if len(pts) < 6:
            return None, 0.0
        o = np.asarray(obj[:, :2], np.float64)
        p = np.asarray(pts, np.float64)
        o_c = o - o.mean(0)
        p_c = p - p.mean(0)
        A, *_ = np.linalg.lstsq(o_c, p_c, rcond=None)
        u, s, vt = np.linalg.svd(A.T)
        if s[0] < 1e-9:
            return None, 0.0
        mag = 1.0 - s[1] / s[0]  # 0 = frontal, ->1 = strongly tilted
        direction = float(np.arctan2(u[1, 0], u[0, 0]))
        return direction, float(mag)

    # ------------------------------------------------------------------
    def _start_solver(self):
        self._solver_busy.set()
        threading.Thread(target=self._solve, daemon=True).start()

    def _solve(self):
        try:
            with self._lock:
                views = list(self.views)
                self._pending_views = 0
            try:
                result = calibrate(views, self.image_size, self.cfg.model)
            except (cv2.error, RuntimeError, ValueError):
                return
            params = np.array([result.fx, result.fy, result.cx, result.cy])
            with self._lock:
                if self._prev_params is not None:
                    rel = np.abs(params - self._prev_params) / np.maximum(np.abs(self._prev_params), 1.0)
                    if float(rel.max()) < self.cfg.convergence_eps:
                        self._converged_solves += 1
                    else:
                        self._converged_solves = 0
                self._prev_params = params
                self._result = result
        finally:
            self._solver_busy.clear()

    # ------------------------------------------------------------------
    def _progress(self) -> float:
        c = min(1.0, self.coverage.fraction / self.cfg.target_coverage)
        t = min(1.0, self.coverage.tilt_fraction / self.cfg.target_tilt_fraction)
        k = min(1.0, len(self.views) / self.cfg.min_keyframes)
        return min(c, t, k)

    def finish(self, wait: bool = True) -> CalibrationResult:
        """Final full solve over all keyframes."""
        if wait:
            self._solver_busy.wait(timeout=60.0)
        with self._lock:
            views = list(self.views)
        result = calibrate(views, self.image_size, self.cfg.model)
        with self._lock:
            self._result = result
        self.state = SessionState.FINISHED
        return result
