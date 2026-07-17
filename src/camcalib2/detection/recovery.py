"""Model-guided marker recovery.

Once a (preliminary) calibration exists, observations that the detector
missed can be recovered - this matters most at the lens periphery, where
distortion, vignetting and blur break the regular detection but where
observations constrain the distortion model the most.

Two mechanisms, both validated by a strict prediction-distance gate (a
final safety net is the solver's reprojection outlier rejection):

* **Dot markers**: the original software re-associated *undecoded*
  detections with predicted positions (refineMarkerDetection). We go one
  step further and *re-measure* at the predicted position: local
  thresholding + sub-pixel centroid of the central dot, with a ring-dot
  plausibility check. This also recovers markers whose ring was too
  degraded to even count as a detection candidate.
* **Checkerboards**: the SB detector returns a partial grid; the grid is
  extrapolated beyond its border, predicted corners are refined with
  ``cornerSubPix`` and gated. This extends fisheye views right to the
  image circle where whole rows/columns are typically lost.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..calibration.solver import CalibrationResult, CameraModel, ViewObservation
from ..patterns.board import MarkerBoard


@dataclass
class RecoveryConfig:
    #: max distance (px) between prediction and re-measurement
    max_residual: float = 6.0
    #: minimum ring dots visible around a recovered center
    min_ring_dots: int = 2
    #: extrapolation margin (grid units) around a detected checkerboard grid
    checker_margin: int = 3
    #: cornerSubPix window half-size (px)
    corner_win: int = 8


# ----------------------------------------------------------------------
def make_projector(result: CalibrationResult, view: ViewObservation,
                   theta_max_deg: float = 100.0):
    """Build a function board-points(N,3) -> pixels(N,2) for the view.

    Estimates the view pose against the current model via PnP on the
    view's existing correspondences. For fisheye, predictions beyond
    ``theta_max_deg`` off-axis are masked (NaN) - use a tighter cap
    (e.g. 85 deg) when the recovered points feed OpenCV's fragile KB
    solver again, the full range when they only feed the OCam refiner.
    """
    obj = view.object_points.astype(np.float64)
    img = view.image_points.astype(np.float64)
    if len(obj) < 6:
        return None
    K = result.camera_matrix
    if result.model is CameraModel.PINHOLE:
        dist = np.asarray(result.dist_coeffs[:5], np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, img.reshape(-1, 1, 2), K, dist,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        t = np.asarray(tvec, np.float64).ravel()

        def project(points):
            pts = np.asarray(points, np.float64)
            proj, _ = cv2.projectPoints(pts, rvec, tvec, K, dist)
            proj = proj.reshape(-1, 2)
            pc = pts @ R.T + t
            proj[pc[:, 2] <= 1e-6] = np.nan  # behind the camera
            return proj
        return project

    D = np.asarray(result.dist_coeffs[:4], np.float64).reshape(4, 1)
    und = cv2.fisheye.undistortPoints(img.reshape(1, -1, 2), K, D)
    ok, rvec, tvec = cv2.solvePnP(obj, und.reshape(-1, 1, 2), np.eye(3), None,
                                  flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rvec = np.asarray(rvec, np.float64)
    tvec = np.asarray(tvec, np.float64)
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.ravel()

    def project(points):
        pts = np.asarray(points, np.float64)
        proj, _ = cv2.fisheye.projectPoints(pts.reshape(1, -1, 3), rvec, tvec, K, D)
        proj = proj.reshape(-1, 2)
        # mask angles beyond the trustworthy KB range (the polynomial
        # wraps around and predictions would land at bogus positions)
        pc = pts @ R.T + t
        theta = np.arctan2(np.hypot(pc[:, 0], pc[:, 1]), pc[:, 2])
        proj[theta > np.deg2rad(theta_max_deg)] = np.nan
        return proj
    return project


# ----------------------------------------------------------------------
def _measure_dot(gray, px, py, win, cfg: RecoveryConfig):
    """Re-measure a dot-marker center near the predicted position.

    Returns the sub-pixel centroid or None. Requires a plausible central
    blob plus at least ``min_ring_dots`` small blobs on the ring annulus
    (the marker signature - guards against dirt and background)."""
    h, w = gray.shape[:2]
    x0, y0 = int(px) - win, int(py) - win
    x1, y1 = int(px) + win + 1, int(py) + win + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    patch = gray[y0:y1, x0:x1]
    if patch.std() < 4.0:  # flat area (vignetted border, no content)
        return None
    binary = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    n, lab, st, ce = cv2.connectedComponentsWithStats(binary)
    if n < 2:
        return None
    pcx, pcy = px - x0, py - y0
    # central blob: closest to the prediction, reasonably sized
    cands = [i for i in range(1, n)
             if 6 <= st[i, cv2.CC_STAT_AREA] <= patch.size / 4]
    if not cands:
        return None
    best = min(cands, key=lambda i: (ce[i][0] - pcx) ** 2 + (ce[i][1] - pcy) ** 2)
    bx, by = ce[best]
    if np.hypot(bx - pcx, by - pcy) > cfg.max_residual:
        return None
    r0 = float(np.sqrt(st[best, cv2.CC_STAT_AREA] / np.pi))

    # ring plausibility: small blobs in the annulus
    ring = 0
    for i in range(1, n):
        if i == best:
            continue
        rr = np.hypot(ce[i][0] - bx, ce[i][1] - by)
        if 1.2 * r0 <= rr <= 2.6 * r0 and st[i, cv2.CC_STAT_AREA] < st[best, cv2.CC_STAT_AREA]:
            ring += 1
    if ring < cfg.min_ring_dots:
        return None

    # sub-pixel intensity-weighted centroid over the dilated component
    mask = (lab == best).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    pf = patch.astype(np.float32)
    outside = pf[mask == 0]
    bg = float(np.percentile(outside, 75)) if outside.size else float(pf.max())
    wgt = np.clip(bg - pf, 0, None) * mask
    total = float(wgt.sum())
    if total < 1e-6:
        return None
    ys, xs = np.mgrid[0:pf.shape[0], 0:pf.shape[1]]
    mx = float((wgt * xs).sum() / total) + x0
    my = float((wgt * ys).sum() / total) + y0
    if np.hypot(mx - px, my - py) > cfg.max_residual:
        return None
    return (mx, my)


def recover_dot_markers(gray, board: MarkerBoard, view: ViewObservation,
                        projector, cfg: RecoveryConfig | None = None
                        ) -> tuple[ViewObservation, int]:
    """Re-measure board markers missing from the view at their predicted
    positions. Returns (possibly extended view, number recovered)."""
    cfg = cfg or RecoveryConfig()
    if view.marker_ids is None or projector is None:
        return view, 0
    have = set(view.marker_ids)
    missing = [m for m in sorted(board.markers) if m not in have]
    if not missing:
        return view, 0
    h, w = gray.shape[:2]

    pred = projector(board.object_points(missing))
    # local search window from the observed nearest-neighbor marker pitch
    pts = view.image_points
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    pitch = float(np.median(np.sqrt(d2.min(axis=1))))
    win = int(np.clip(pitch * 0.45, 10, 120))

    new_ids, new_pts = [], []
    for mid, (px, py) in zip(missing, pred):
        if not (np.isfinite(px) and np.isfinite(py)):
            continue
        if not (-win < px < w + win and -win < py < h + win):
            continue
        m = _measure_dot(gray, float(px), float(py), win, cfg)
        if m is not None:
            new_ids.append(mid)
            new_pts.append(m)
    if not new_ids:
        return view, 0

    ids = list(view.marker_ids) + new_ids
    pts = np.vstack([view.image_points, np.array(new_pts, np.float32)])
    obj = np.vstack([view.object_points, board.object_points(new_ids)])
    return ViewObservation(pts, obj, marker_ids=ids, timestamp=view.timestamp), len(new_ids)


# ----------------------------------------------------------------------
def recover_checkerboard_corners(gray, view: ViewObservation, projector,
                                 square_size: float,
                                 cfg: RecoveryConfig | None = None
                                 ) -> tuple[ViewObservation, int]:
    """Extend a partial checkerboard grid beyond its detected border.

    The view's object points must lie on a (col*s, row*s, 0) grid (as
    produced by CheckerboardDetector). Candidate corners in a margin of
    ``cfg.checker_margin`` grid units around the detected grid are
    projected, refined with cornerSubPix and gated by the prediction
    distance. Recovered corners get extrapolated object coordinates,
    which is consistent for intrinsics (the board offset is absorbed by
    the extrinsics)."""
    cfg = cfg or RecoveryConfig()
    if projector is None or len(view) < 6:
        return view, 0
    h, w = gray.shape[:2]
    s = square_size
    grid = np.round(view.object_points[:, :2] / s).astype(int)
    have = {(int(c), int(r)) for c, r in grid}
    c0, c1 = grid[:, 0].min(), grid[:, 0].max()
    r0, r1 = grid[:, 1].min(), grid[:, 1].max()
    m = cfg.checker_margin
    candidates = [(c, r)
                  for c in range(c0 - m, c1 + m + 1)
                  for r in range(r0 - m, r1 + m + 1)
                  if (c, r) not in have]
    if not candidates:
        return view, 0
    obj = np.array([(c * s, r * s, 0.0) for c, r in candidates], np.float64)
    pred = projector(obj)

    win = cfg.corner_win
    new_obj, new_pts = [], []
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    for (c, r), (px, py) in zip(candidates, pred):
        if not (np.isfinite(px) and np.isfinite(py)):
            continue
        if not (win + 1 <= px < w - win - 1 and win + 1 <= py < h - win - 1):
            continue
        patch = gray[int(py) - win:int(py) + win + 1,
                     int(px) - win:int(px) + win + 1]
        if patch.std() < 8.0:  # outside board / image circle
            continue
        pt = np.array([[px, py]], np.float32)
        refined = cv2.cornerSubPix(gray, pt.reshape(-1, 1, 2), (win, win),
                                   (-1, -1), term).reshape(-1, 2)[0]
        if np.hypot(refined[0] - px, refined[1] - py) > cfg.max_residual:
            continue
        # saddle-point quality: reject weak corners (board edge, texture)
        q = cv2.cornerMinEigenVal(patch, blockSize=5)
        cyi, cxi = patch.shape[0] // 2, patch.shape[1] // 2
        if float(q[cyi - 2:cyi + 3, cxi - 2:cxi + 3].max()) < 1e-3:
            continue
        new_obj.append((c * s, r * s, 0.0))
        new_pts.append(refined)
    if not new_pts:
        return view, 0

    pts = np.vstack([view.image_points, np.array(new_pts, np.float32)])
    obj = np.vstack([view.object_points, np.array(new_obj, np.float32)])
    ids = None
    if view.marker_ids is not None:
        ids = list(view.marker_ids) + [-1] * len(new_pts)
    return ViewObservation(pts, obj, marker_ids=ids, timestamp=view.timestamp), len(new_pts)
