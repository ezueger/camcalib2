"""Camera intrinsics calibration (pinhole and fisheye) on top of OpenCV."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np


class CameraModel(str, Enum):
    PINHOLE = "pinhole"
    FISHEYE = "fisheye"


@dataclass
class ViewObservation:
    """Point correspondences of one view (one keyframe)."""

    image_points: np.ndarray  # (N,2) float32
    object_points: np.ndarray  # (N,3) float32, board coordinates in mm
    marker_ids: list[int] | None = None
    timestamp: float | None = None

    def __post_init__(self):
        self.image_points = np.ascontiguousarray(self.image_points, dtype=np.float32)
        self.object_points = np.ascontiguousarray(self.object_points, dtype=np.float32)
        if len(self.image_points) != len(self.object_points):
            raise ValueError("image/object point count mismatch")

    def __len__(self):
        return len(self.image_points)


@dataclass
class CalibrationResult:
    model: CameraModel
    image_size: tuple[int, int]  # (width, height)
    camera_matrix: np.ndarray  # 3x3
    dist_coeffs: np.ndarray  # (5,) pinhole [k1 k2 p1 p2 k3] / (4,) fisheye [k1..k4]
    rms: float
    per_view_rms: list[float]
    n_views: int
    n_points: int
    rvecs: list[np.ndarray] = field(default_factory=list)
    tvecs: list[np.ndarray] = field(default_factory=list)
    per_point_errors: list[np.ndarray] = field(default_factory=list)
    #: image points of the views actually used (aligned with per_point_errors)
    used_image_points: list[np.ndarray] = field(default_factory=list)

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])


MIN_POINTS_PER_VIEW = 6


def calibrate(views: list[ViewObservation], image_size: tuple[int, int],
              model: CameraModel = CameraModel.PINHOLE,
              reject_outliers: bool = True) -> CalibrationResult:
    views = [v for v in views if len(v) >= MIN_POINTS_PER_VIEW]
    if len(views) < 3:
        raise ValueError("need at least 3 usable views")
    if model is CameraModel.PINHOLE:
        return _calibrate_pinhole(views, image_size, reject_outliers)
    return _calibrate_fisheye(views, image_size)


# ----------------------------------------------------------------------
def _calibrate_pinhole(views, image_size, reject_outliers) -> CalibrationResult:
    obj = [v.object_points.reshape(-1, 1, 3) for v in views]
    img = [v.image_points.reshape(-1, 1, 2) for v in views]

    # pass 1: planar approximation (z=0) to obtain a stable initial guess -
    # OpenCV cannot self-initialize from non-planar boards
    obj_planar = []
    for o in obj:
        p = o.copy()
        p[..., 2] = 0.0
        obj_planar.append(p)
    rms0, K, dist, _, _ = cv2.calibrateCamera(
        obj_planar, img, image_size, None, None)

    # pass 2: refine with the measured (non-planar) board geometry
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj, img, image_size, K, dist, flags=cv2.CALIB_USE_INTRINSIC_GUESS)

    per_view, per_point = _pinhole_errors(views, K, dist, rvecs, tvecs)

    if reject_outliers:
        # drop points with reprojection error > max(3*rms, 2px), recalibrate
        thr = max(3.0 * rms, 2.0)
        cleaned = []
        dropped = 0
        for v, err in zip(views, per_point):
            keep = err < thr
            dropped += int((~keep).sum())
            if keep.sum() >= MIN_POINTS_PER_VIEW:
                cleaned.append(ViewObservation(
                    v.image_points[keep], v.object_points[keep],
                    marker_ids=[m for m, k in zip(v.marker_ids, keep) if k] if v.marker_ids else None,
                    timestamp=v.timestamp))
        if dropped and len(cleaned) >= 3:
            views = cleaned
            obj = [v.object_points.reshape(-1, 1, 3) for v in views]
            img = [v.image_points.reshape(-1, 1, 2) for v in views]
            rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                obj, img, image_size, K, dist, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
            per_view, per_point = _pinhole_errors(views, K, dist, rvecs, tvecs)

    return CalibrationResult(
        model=CameraModel.PINHOLE, image_size=tuple(image_size),
        camera_matrix=K, dist_coeffs=dist.ravel()[:5], rms=float(rms),
        per_view_rms=per_view, n_views=len(views),
        n_points=sum(len(v) for v in views),
        rvecs=list(rvecs), tvecs=list(tvecs), per_point_errors=per_point,
        used_image_points=[v.image_points for v in views])


def _pinhole_errors(views, K, dist, rvecs, tvecs):
    per_view, per_point = [], []
    for v, r, t in zip(views, rvecs, tvecs):
        proj, _ = cv2.projectPoints(v.object_points, r, t, K, dist)
        err = np.linalg.norm(proj.reshape(-1, 2) - v.image_points, axis=1)
        per_point.append(err)
        per_view.append(float(np.sqrt(np.mean(err ** 2))))
    return per_view, per_point


# ----------------------------------------------------------------------
def _fisheye_flags() -> int:
    # constants moved to the top-level namespace in OpenCV 5
    rec = getattr(cv2.fisheye, "CALIB_RECOMPUTE_EXTRINSIC",
                  getattr(cv2, "CALIB_RECOMPUTE_EXTRINSIC"))
    skew = getattr(cv2.fisheye, "CALIB_FIX_SKEW", getattr(cv2, "CALIB_FIX_SKEW"))
    guess = getattr(cv2.fisheye, "CALIB_USE_INTRINSIC_GUESS",
                    getattr(cv2, "CALIB_USE_INTRINSIC_GUESS"))
    return rec | skew | guess


_FISHEYE_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-10)


def _fisheye_joint(views, image_size, K0, D0):
    obj = [v.object_points.reshape(1, -1, 3).astype(np.float64) for v in views]
    img = [v.image_points.reshape(1, -1, 2).astype(np.float64) for v in views]
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        obj, img, image_size, K0.copy(), D0.copy(),
        flags=_fisheye_flags(), criteria=_FISHEYE_CRIT)
    per_view, per_point = [], []
    for v, r, t in zip(views, rvecs, tvecs):
        proj, _ = cv2.fisheye.projectPoints(
            v.object_points.reshape(1, -1, 3).astype(np.float64), r, t, K, D)
        err = np.linalg.norm(proj.reshape(-1, 2) - v.image_points, axis=1)
        per_point.append(err)
        per_view.append(float(np.sqrt(np.mean(err ** 2))))
    return rms, K, D, rvecs, tvecs, per_view, per_point


def _calibrate_fisheye(views, image_size) -> CalibrationResult:
    """Robust Kannala-Brandt calibration.

    OpenCV's fisheye solver is fragile: the per-view homography-based
    extrinsics initialization diverges for strongly distorted views and a
    single poisoned view drags the whole joint solve (observed: one view
    at 250 px rms while all others fit at 0.3 px).  Strategy:

    1. initial focal guess from the image size (equidistant, ~180 deg),
    2. quick single-view prefilter drops views whose extrinsics
       initialization fails outright,
    3. joint solve, then iteratively drop per-view rms outliers and
       re-solve until clean.
    """
    w, h = image_size
    D0 = np.zeros((4, 1))

    # 1) initial guess: equidistant lens filling the smaller image dimension
    f0 = 0.32 * min(w, h)
    K0 = np.array([[f0, 0, w / 2.0], [0, f0, h / 2.0], [0, 0, 1]])

    # 2) rank views by single-view fit quality (also drops broken ones)
    ranked = []
    for v in views:
        try:
            res = cv2.fisheye.calibrate(
                [v.object_points.reshape(1, -1, 3).astype(np.float64)],
                [v.image_points.reshape(1, -1, 2).astype(np.float64)],
                image_size, K0.copy(), D0.copy(), flags=_fisheye_flags(),
                criteria=_FISHEYE_CRIT)
            tz = abs(float(res[4][0].ravel()[2]))
            if res[0] < 5.0 and tz < 1e5:
                ranked.append((res[0], v))
        except cv2.error:
            continue
    if len(ranked) < 3:
        raise RuntimeError("fisheye calibration failed: not enough usable views")
    ranked.sort(key=lambda x: x[0])

    # 3) grow the view set incrementally; a view whose (fragile,
    #    homography-based) extrinsics initialization breaks the joint
    #    solve is identified by the failing addition and skipped
    usable = [ranked[0][1], ranked[1][1], ranked[2][1]]
    result = _fisheye_joint(usable, image_size, K0, D0)
    K, D = result[1], result[2]
    for _, v in ranked[3:]:
        try:
            cand = _fisheye_joint(usable + [v], image_size, K, D)
        except cv2.error:
            continue
        usable.append(v)
        result = cand
        K, D = result[1], result[2]
    rms, K, D, rvecs, tvecs, per_view, per_point = result

    # 3) iteratively drop poisoned views (typically stuck extrinsics)
    for _ in range(6):
        med = float(np.median(per_view))
        thr = max(4.0 * med, 2.0)
        keep = [i for i, e in enumerate(per_view) if e <= thr]
        if len(keep) == len(usable) or len(keep) < 3:
            break
        usable = [usable[i] for i in keep]
        rms, K, D, rvecs, tvecs, per_view, per_point = _fisheye_joint(
            usable, image_size, K, D)

    rms = float(np.sqrt(np.mean(np.concatenate(per_point) ** 2)))
    return CalibrationResult(
        model=CameraModel.FISHEYE, image_size=tuple(image_size),
        camera_matrix=np.asarray(K), dist_coeffs=np.asarray(D).ravel()[:4],
        rms=rms, per_view_rms=per_view, n_views=len(usable),
        n_points=sum(len(v) for v in usable),
        rvecs=list(rvecs), tvecs=list(tvecs), per_point_errors=per_point,
        used_image_points=[v.image_points for v in usable])
