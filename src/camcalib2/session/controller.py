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


class CaptureState(str, Enum):
    """Still-capture flow: move the camera, hold still, capture, repeat."""

    MOVE = "move"  # waiting for the camera to move to a new position
    HOLD = "hold"  # motion low - settling before the evaluation
    EVALUATING = "evaluating"  # precise detection running
    CAPTURED = "captured"  # capture accepted - cooldown/flash phase
    FULL = "full"  # max keyframes reached


@dataclass
class StillTick:
    """Cheap per-frame result of the still-capture state machine."""

    capture_state: CaptureState
    #: run the (expensive) evaluation on this frame now
    evaluate: bool
    motion: float
    #: 0..1 settle progress while holding still
    hold_progress: float = 0.0
    message: str = ""


@dataclass
class SessionConfig:
    model: CameraModel = CameraModel.PINHOLE
    target_coverage: float = 0.85
    target_tilt_fraction: float = 0.5
    min_keyframes: int = 12
    max_keyframes: int = 60
    #: recalibrate every N new keyframes
    recalib_every: int = 4
    #: skip the background recalibration above this many views for
    #: fisheye - the robust solve grows quadratically and would stall
    #: the live loop; the final solve covers it
    fisheye_recalib_view_cap: int = 30
    #: relative parameter change (fx,fy,cx,cy) considered converged
    convergence_eps: float = 1e-3
    keyframe_policy: KeyframePolicy = field(default_factory=KeyframePolicy)
    # --- still-capture flow ---------------------------------------------
    #: mean abs frame difference (0..255, on a downscaled image) below
    #: which the camera counts as still
    still_threshold: float = 1.2
    #: motion score above which the camera counts as moving (hysteresis)
    move_threshold: float = 2.5
    #: seconds the image must be still before the evaluation runs
    settle_s: float = 0.4
    #: pause after a capture before the next one can be armed
    cooldown_s: float = 1.0
    #: median PnP residual (px) above which a keyframe detection is
    #: rejected as toxic (mis-indexed grid, gross mismatch)
    keyframe_gate_px: float = 8.0


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
    capture_state: CaptureState | None = None


class CalibrationSession:
    def __init__(self, target: MarkerBoard | Checkerboard,
                 image_size: tuple[int, int],
                 config: SessionConfig | None = None):
        self.cfg = config or SessionConfig()
        self.image_size = image_size
        self.target = target
        if isinstance(target, MarkerBoard):
            # fast preview detector for the live loop (coverage/UI),
            # precise detector re-runs only on accepted keyframes
            from ..detection.dot_marker import DotMarkerDetectorConfig
            self._detector = DotMarkerDetector(
                target, DotMarkerDetectorConfig(refine_subpixel=False))
            self._detector_precise = DotMarkerDetector(target)
            self._detect = self._detect_markers
            self._detect_precise = lambda gray: self._detect_markers(gray, precise=True)
        else:
            self._detector = CheckerboardDetector(target, fast=True)
            self._detector_precise = CheckerboardDetector(target)
            self._detect = self._detect_checkerboard
            self._detect_precise = lambda gray: self._detect_checkerboard(gray, precise=True)

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

        # still-capture state machine
        self._capture_state = CaptureState.HOLD  # first capture needs no motion
        self._still_since: float | None = None
        self._cooldown_until = -1e9
        self._moved_since_capture = True
        self._prev_small: np.ndarray | None = None
        self._motion_ema: float | None = None
        self._motion_history: list[float] = []

    # ------------------------------------------------------------------
    # still-capture flow: cheap per-frame tick; the expensive evaluation
    # runs only when the camera has settled
    def motion_score(self, gray: np.ndarray) -> float:
        """Smoothed mean absolute frame difference on a ~240px image.

        At that scale hand tremor is sub-pixel and disappears; what
        remains is sensor noise + real camera movement."""
        h, w = gray.shape[:2]
        scale = 240.0 / max(w, 1)
        small = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (3, 3), 0)
        prev, self._prev_small = self._prev_small, small
        if prev is None or prev.shape != small.shape:
            return 255.0
        raw = float(cv2.absdiff(small, prev).mean())
        # EMA smoothing suppresses single-frame spikes (flicker etc.)
        self._motion_ema = raw if self._motion_ema is None else \
            0.5 * self._motion_ema + 0.5 * raw
        score = self._motion_ema
        self._motion_history.append(score)
        if len(self._motion_history) > 120:
            del self._motion_history[:60]
        return score

    def _thresholds(self) -> tuple[float, float]:
        """Adaptive (still, move) thresholds relative to the camera's
        noise floor - hand-held cameras are never perfectly still, and
        the noise floor depends on sensor/exposure/gain. The floor is
        estimated as a low percentile of the recent motion scores (when
        the camera rests, the score converges to pure sensor noise)."""
        cfg = self.cfg
        if len(self._motion_history) < 15:
            return cfg.still_threshold, cfg.move_threshold
        # cap the floor estimate: if the camera keeps moving, the
        # percentile rises with it and everything would look "still"
        floor = min(float(np.percentile(self._motion_history[-120:], 10)), 3.0)
        still = max(cfg.still_threshold, floor * 1.6 + 0.3)
        move = max(cfg.move_threshold, still * 2.0)
        return still, move

    def tick(self, frame: np.ndarray, timestamp: float) -> StillTick:
        """Advance the move -> hold-still -> capture state machine."""
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion = self.motion_score(gray)
        cfg = self.cfg
        still_thr, move_thr = self._thresholds()

        if len(self.views) >= cfg.max_keyframes:
            self._capture_state = CaptureState.FULL
            return StillTick(CaptureState.FULL, False, motion,
                             message="Genug Aufnahmen - Ergebnis berechnen")

        if self._capture_state is CaptureState.EVALUATING:
            return StillTick(CaptureState.EVALUATING, False, motion,
                             message="Auswertung läuft …")

        if timestamp < self._cooldown_until:
            return StillTick(CaptureState.CAPTURED, False, motion,
                             message="Aufgenommen ✓  - weiter zur nächsten Position")

        if self._capture_state is CaptureState.CAPTURED:
            self._capture_state = CaptureState.MOVE

        if self._capture_state is CaptureState.MOVE:
            if not self._moved_since_capture:
                if motion > move_thr:
                    self._moved_since_capture = True
                else:
                    return StillTick(CaptureState.MOVE, False, motion,
                                     message="Kamera zur nächsten Position bewegen")
            self._capture_state = CaptureState.HOLD
            self._still_since = None

        # HOLD: wait for the image to settle - the settle timer restarts
        # whenever the motion rises above the still threshold (sustained
        # moderate movement must never accumulate "still" time)
        if motion > move_thr:
            self._still_since = None
            return StillTick(CaptureState.HOLD, False, motion,
                             message="Bewegung erkannt - gleich still halten")
        if motion > still_thr:
            self._still_since = None
            return StillTick(CaptureState.HOLD, False, motion,
                             message="Still halten …")
        if self._still_since is None:
            self._still_since = timestamp
        held = timestamp - self._still_since
        if held < cfg.settle_s:
            return StillTick(CaptureState.HOLD, False, motion,
                             hold_progress=held / cfg.settle_s,
                             message="Still halten …")
        self._capture_state = CaptureState.EVALUATING
        return StillTick(CaptureState.EVALUATING, True, motion,
                         hold_progress=1.0, message="Auswertung läuft …")

    def evaluate(self, frame: np.ndarray, timestamp: float) -> FrameFeedback:
        """Precise detection + keyframe accept for a settled frame."""
        fb = self.process(frame, timestamp, precise=True)
        if fb.keyframe:
            self._cooldown_until = timestamp + self.cfg.cooldown_s
            self._capture_state = CaptureState.CAPTURED
            self._moved_since_capture = False
            fb.capture_state = CaptureState.CAPTURED
        else:
            # nothing usable - re-arm for another try at this position
            self._capture_state = CaptureState.HOLD
            self._still_since = None
            fb.capture_state = CaptureState.HOLD
        return fb

    # ------------------------------------------------------------------
    def _detect_markers(self, gray, precise=False):
        det = self._detector_precise if precise else self._detector
        markers = det.detect(gray)
        if not markers:
            return [], np.empty((0, 2), np.float32), None
        ids = [m.marker_id for m in markers]
        pts = np.array([m.center for m in markers], dtype=np.float32)
        obj = self.target.object_points(ids)
        return ids, pts, obj

    def _detect_checkerboard(self, gray, precise=False):
        det = (self._detector_precise if precise else self._detector).detect(gray)
        if det is None:
            return [], np.empty((0, 2), np.float32), None
        ids = list(range(len(det.image_points)))
        return ids, det.image_points, det.object_points

    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray, timestamp: float,
                precise: bool = False) -> FrameFeedback:
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ids, pts, obj = (self._detect_precise(gray) if precise
                         else self._detect(gray))

        # model-guided recovery: once a preliminary calibration exists,
        # re-measure observations the detector missed (lens periphery!).
        # Recovered points enrich coverage/UI on every frame; keyframes
        # are re-detected precisely below, where recovery is applied only
        # for pinhole (fisheye rim points would poison OpenCV's fragile
        # KB solver - they are re-recovered into the OCam refinement on
        # export instead).
        with self._lock:
            model_result = self._result
        if model_result is not None and len(ids) >= 8:
            ids, pts, obj = self._recover(gray, ids, pts, obj, model_result)

        keyframe = False
        reason = "no_target"
        if len(ids) > 0 and self.state in (SessionState.WAITING, SessionState.SCANNING):
            if self.state is SessionState.WAITING:
                self.state = SessionState.SCANNING
            new_cells = self.coverage.new_cells(pts)
            keyframe, reason = self.selector.consider(gray, ids, pts, timestamp, new_cells)
            if keyframe:
                # the live loop ran the fast preview detector; re-detect
                # this one frame precisely (sub-pixel) for the calibration
                if precise:
                    p_ids, p_pts, p_obj = ids, pts, obj
                else:
                    p_ids, p_pts, p_obj = self._detect_precise(gray)
                if len(p_ids) >= self.cfg.keyframe_policy.min_points:
                    if model_result is not None and not self._view_sane(
                            p_ids, p_pts, p_obj, model_result):
                        keyframe, reason = False, "detection_rejected"
                    elif len(self.views) >= self.cfg.max_keyframes:
                        keyframe, reason = False, "max_keyframes"
                    else:
                        if (self.cfg.model is CameraModel.PINHOLE
                                and model_result is not None and len(p_ids) >= 8):
                            p_ids, p_pts, p_obj = self._recover(
                                gray, p_ids, p_pts, p_obj, model_result)
                        self._accept_keyframe(p_ids, p_pts, p_obj, timestamp)
                else:
                    keyframe, reason = False, "precise_detect_failed"

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
    def _view_sane(self, ids, pts, obj, model_result) -> bool:
        """Toxic-detection gate: once a model exists, a keyframe must be
        consistent with it (median PnP reprojection error below
        ``keyframe_gate_px``). Catches mis-indexed checkerboard grids
        and gross mismatches before they poison the solve."""
        try:
            from ..detection.recovery import make_projector
            view = ViewObservation(pts, obj, marker_ids=list(ids))
            projector = make_projector(model_result, view)
            if projector is None:
                return True  # cannot judge - let the solver gates decide
            pred = projector(view.object_points.astype(np.float64))
            err = np.linalg.norm(pred - view.image_points, axis=1)
            err = err[np.isfinite(err)]
            if err.size < 6:
                return True
            return float(np.median(err)) < self.cfg.keyframe_gate_px
        except (cv2.error, ValueError):
            return True

    # ------------------------------------------------------------------
    def _recover(self, gray, ids, pts, obj, model_result):
        from ..calibration.solver import ViewObservation
        from ..detection.recovery import (make_projector, recover_dot_markers,
                                          recover_checkerboard_corners)
        try:
            view = ViewObservation(pts, obj, marker_ids=list(ids))
            # tighter angle cap for fisheye: these points feed the
            # incremental KB solver, whose init is fragile at the rim
            theta_cap = 85.0 if self.cfg.model is CameraModel.FISHEYE else 100.0
            projector = make_projector(model_result, view, theta_max_deg=theta_cap)
            if projector is None:
                return ids, pts, obj
            if isinstance(self.target, MarkerBoard):
                view2, n = recover_dot_markers(gray, self.target, view, projector)
            else:
                view2, n = recover_checkerboard_corners(
                    gray, view, projector, self.target.square_size)
            if n > 0:
                return (list(view2.marker_ids), view2.image_points,
                        view2.object_points)
        except (cv2.error, ValueError):
            pass
        return ids, pts, obj

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
        if (self.cfg.model is CameraModel.FISHEYE
                and n > self.cfg.fisheye_recalib_view_cap):
            return  # robust fisheye solve grows quadratically - final solve only
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

    def resume(self) -> None:
        """Continue scanning after a finish() - keyframes, coverage and
        the current calibration are kept, new keyframes improve them."""
        if self.state in (SessionState.FINISHED, SessionState.CONVERGED):
            self.state = SessionState.SCANNING
