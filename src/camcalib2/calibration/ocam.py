"""Native OCam (Scaramuzza) fisheye model and calibration.

Model (matching the legacy software's ``<id>-ocam.xml``):

* pixel offset  p - (cx, cy) = A @ (u, v)  with  A = [[c, d], [e, 1]]
  (the affine part absorbs sensor shear/ellipticity and, in part,
  lens decentering - which Kannala-Brandt cannot model),
* the sensor point (u, v) with radius rho maps to the viewing ray
  (u, v, f(rho)),  f(rho) = a0 + a2 rho^2 + a3 rho^3 + a4 rho^4
  (a1 = 0 by convention; theta(rho) = atan2(rho, f(rho)) can exceed
  90 deg, i.e. FOV > 180 deg is representable).

Calibration bootstraps from the robust Kannala-Brandt solve (center,
extrinsics, r(theta) curve) and refines all intrinsics + extrinsics
jointly with a sparse Levenberg-Marquardt (scipy).
"""

from __future__ import annotations

import inspect
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np

from .solver import (CalibrationResult, CameraModel, ViewObservation,
                     _calibrate_fisheye, _check_cancel, _report)


@dataclass
class OcamModel:
    image_size: tuple[int, int]
    cx: float
    cy: float
    c: float = 1.0
    d: float = 0.0
    e: float = 0.0
    #: polynomial [a0, a2, a3, a4]
    poly: np.ndarray = field(default_factory=lambda: np.zeros(4))

    # ------------------------------------------------------------------
    def f(self, rho):
        a0, a2, a3, a4 = self.poly
        return a0 + a2 * rho ** 2 + a3 * rho ** 3 + a4 * rho ** 4

    def f_prime(self, rho):
        _, a2, a3, a4 = self.poly
        return 2 * a2 * rho + 3 * a3 * rho ** 2 + 4 * a4 * rho ** 3

    def theta(self, rho):
        """Off-axis angle for sensor radius rho."""
        return np.arctan2(rho, self.f(rho))

    # ------------------------------------------------------------------
    def cam2world(self, pixels: np.ndarray) -> np.ndarray:
        """Pixel (N,2) -> unit rays (N,3), z forward."""
        p = np.asarray(pixels, np.float64) - (self.cx, self.cy)
        det = self.c - self.d * self.e
        u = (p[:, 0] - self.d * p[:, 1]) / det
        v = (-self.e * p[:, 0] + self.c * p[:, 1]) / det
        rho = np.hypot(u, v)
        w = self.f(rho)
        rays = np.stack([u, v, w], axis=1)
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    def world2cam(self, points: np.ndarray) -> np.ndarray:
        """Camera-frame 3D points (N,3) -> pixels (N,2).

        Solves f(rho)/rho = z/m per point (vectorized Newton, seeded from
        a dense theta->rho table).
        """
        pts = np.asarray(points, np.float64)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        m = np.hypot(x, y)
        theta_q = np.arctan2(m, z)  # off-axis angle of the query points

        rho = self._rho_of_theta(theta_q)
        # Newton on g(rho) = f(rho) - t*rho, t = z/m = cot(theta)
        t = np.where(m > 1e-12, z / np.maximum(m, 1e-12), np.inf)
        for _ in range(4):
            g = self.f(rho) - t * rho
            gp = self.f_prime(rho) - t
            step = np.where(np.abs(gp) > 1e-12, g / gp, 0.0)
            rho = np.clip(rho - step, 0.0, self._rho_valid_max())

        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.where(m > 1e-12, rho / np.maximum(m, 1e-12), 0.0)
        u = x * scale
        v = y * scale
        px = self.c * u + self.d * v + self.cx
        py = self.e * u + v + self.cy
        return np.stack([px, py], axis=1)

    # ------------------------------------------------------------------
    def _table(self):
        """Cached (rho, theta) lookup over the monotonic range."""
        cached = getattr(self, "_tbl", None)
        if cached is not None:
            return cached
        w, h = self.image_size
        rs = np.linspace(0, float(np.hypot(w, h)), 2048)
        th = self.theta(rs)
        bad = np.nonzero(np.diff(th) <= 0)[0]
        cut = int(bad[0]) + 1 if bad.size else len(rs)
        object.__setattr__(self, "_tbl", (rs[:cut], th[:cut]))
        return self._tbl

    def _rho_valid_max(self) -> float:
        """Largest rho for which theta(rho) is still monotonic."""
        return float(self._table()[0][-1])

    def _rho_of_theta(self, thetas):
        rs, th = self._table()
        return np.interp(thetas, th, rs)

    def r_of_theta(self, thetas):
        """Image radius (before affine) for off-axis angles."""
        return self._rho_of_theta(thetas)


@dataclass
class OcamCalibrationResult:
    model: OcamModel
    rms: float
    per_view_rms: list[float]
    n_views: int
    n_points: int
    rvecs: list[np.ndarray]
    tvecs: list[np.ndarray]
    per_point_errors: list[np.ndarray] = field(default_factory=list)
    used_image_points: list[np.ndarray] = field(default_factory=list)
    used_reprojections: list[np.ndarray] = field(default_factory=list)
    #: the Kannala-Brandt bootstrap result (for comparison/diagnostics)
    kb: CalibrationResult | None = None


def calibrate_ocam(views: list[ViewObservation], image_size: tuple[int, int],
                   refine_affine: bool = True,
                   refine_views: list[ViewObservation] | None = None,
                   *, progress_cb=None, cancel_event=None,
                   kb: CalibrationResult | None = None
                   ) -> OcamCalibrationResult:
    """Scaramuzza calibration bootstrapped from the robust KB solve.

    ``refine_views`` (aligned 1:1 with ``views``) optionally provides
    enriched observations for the LM refinement - e.g. recovered rim
    points. The fragile OpenCV KB bootstrap runs on the conservative
    ``views``; our own bundle adjustment (warm-started extrinsics, no
    homography init) digests the extra periphery observations safely.

    ``kb`` optionally supplies an already-computed Kannala-Brandt solve to
    skip the (expensive) bootstrap - valid only when it was solved over the
    same ``views`` (so it must not be reused together with ``refine_views``).
    ``progress_cb(local_0_1, msg)`` and ``cancel_event`` drive the UI.
    """
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    if refine_views is not None and len(refine_views) != len(views):
        raise ValueError("refine_views must align with views")

    # bootstrap: reuse the caller's KB solve when given, else compute it
    boot_hi = 0.0 if kb is not None else 0.25
    if kb is None:
        kb = _calibrate_fisheye(
            views, image_size, cancel_event=cancel_event,
            progress_cb=lambda f, m: _report(progress_cb, boot_hi * f, m))
    _check_cancel(cancel_event)
    _report(progress_cb, boot_hi + 0.02, "OCam Initialisierung …")

    # --- initialize the polynomial from the KB r(theta) curve.
    # The fit must cover the radius range of ALL points fed to the LM
    # (recovered rim points may exceed the KB observations; a quartic
    # extrapolates catastrophically beyond its fit range)
    r_max = 0.0
    for v in (refine_views or views):
        r = np.hypot(v.image_points[:, 0] - kb.cx, v.image_points[:, 1] - kb.cy)
        r_max = max(r_max, float(r.max()))
    thetas = np.linspace(0.02, _kb_theta_max(kb, r_max * 1.03), 256)
    k = kb.dist_coeffs
    th_d = thetas * (1 + k[0] * thetas**2 + k[1] * thetas**4
                     + k[2] * thetas**6 + k[3] * thetas**8)
    f_iso = 0.5 * (kb.fx + kb.fy)
    rho = f_iso * th_d
    # f(rho) = rho * cot(theta)
    target = rho / np.tan(thetas)
    Amat = np.stack([np.ones_like(rho), rho**2, rho**3, rho**4], axis=1)
    poly, *_ = np.linalg.lstsq(Amat, target, rcond=None)

    model = OcamModel(image_size=image_size, cx=kb.cx, cy=kb.cy, poly=poly)

    # --- match KB's used views back to the input list (it may have
    #     dropped poisoned views and rejected individual points)
    used_views = []
    for uv in kb.used_image_points:
        idx = None
        for i, v in enumerate(views):
            vi = v.image_points
            if len(vi) == len(uv) and np.allclose(vi, uv):
                idx = i
                break
            # point-level rejection: uv is a subset of the original view
            if len(uv) < len(vi):
                d = np.abs(vi[:, None, :] - uv[None, :, :]).sum(-1).min(axis=0)
                if float(d.max()) < 1e-3:
                    idx = i
                    break
        if idx is None:
            raise ValueError("cannot match KB views back to input views")
        used_views.append(refine_views[idx] if refine_views is not None else views[idx])

    if refine_views is not None:
        # refine_views are enriched versions of the *raw* views - they may
        # reintroduce gross outliers that the KB solve's point-level
        # rejection already removed. Gate every point against the KB
        # model before the LM sees it.
        K = kb.camera_matrix
        D = np.asarray(kb.dist_coeffs[:4], np.float64).reshape(4, 1)
        thr = max(4.0 * kb.rms, 3.0)
        gated = []
        for v, rv, tv in zip(used_views, kb.rvecs, kb.tvecs):
            proj, _ = cv2.fisheye.projectPoints(
                v.object_points.reshape(1, -1, 3).astype(np.float64),
                np.asarray(rv, np.float64), np.asarray(tv, np.float64), K, D)
            err = np.linalg.norm(proj.reshape(-1, 2) - v.image_points, axis=1)
            keep = err < thr
            if keep.sum() < 6:
                gated.append(v)
                continue
            gated.append(ViewObservation(
                v.image_points[keep], v.object_points[keep],
                marker_ids=[m for m, k in zip(v.marker_ids, keep) if k]
                if v.marker_ids else None,
                timestamp=v.timestamp))
        used_views = gated

    # --- parameter vector: [cx, cy, c, d, e, b0, b2, b3, b4, (rvec,tvec)*n]
    # the polynomial is optimized on the normalized radius rho/r0 so all
    # coefficients are O(100) - the raw a4 ~ 1e-10 breaks the numeric
    # Jacobian's step size selection otherwise
    n_views = len(used_views)
    w, h = image_size
    r0 = float(np.hypot(w, h)) / 2.0
    r0_pows = np.array([1.0, r0 ** 2, r0 ** 3, r0 ** 4])
    x0 = np.concatenate([
        [model.cx, model.cy, 1.0, 0.0, 0.0], poly * r0_pows,
        *[np.concatenate([r.ravel(), t.ravel()]) for r, t in zip(kb.rvecs, kb.tvecs)]])

    counts = [len(v) for v in used_views]
    offsets = np.concatenate([[0], np.cumsum(counts)])
    n_res = int(offsets[-1]) * 2

    def unpack(x):
        m = OcamModel(image_size=image_size, cx=x[0], cy=x[1],
                      c=x[2] if refine_affine else 1.0,
                      d=x[3] if refine_affine else 0.0,
                      e=x[4] if refine_affine else 0.0,
                      poly=x[5:9] / r0_pows)
        exts = x[9:].reshape(n_views, 6)
        return m, exts

    def residuals(x):
        m, exts = unpack(x)
        out = np.empty((int(offsets[-1]), 2))
        for i, v in enumerate(used_views):
            R, _ = cv2.Rodrigues(exts[i, :3])
            pc = v.object_points.astype(np.float64) @ R.T + exts[i, 3:]
            proj = m.world2cam(pc)
            out[offsets[i]:offsets[i + 1]] = proj - v.image_points
        return out.ravel()

    # sparsity: each residual depends on intrinsics + its view's extrinsics
    sparsity = lil_matrix((n_res, len(x0)), dtype=int)
    sparsity[:, :9] = 1
    for i in range(n_views):
        sparsity[offsets[i] * 2:offsets[i + 1] * 2, 9 + 6 * i: 15 + 6 * i] = 1

    # progress/cancel wiring for the two LM stages: the finite-difference
    # Jacobian is parallelized across a thread pool via ``workers`` (fresh
    # OcamModel per residual() call -> thread-safe), and a per-iteration
    # ``callback`` reports nfev-based progress and honours the cancel token.
    _MAX_NFEV = 200
    _ls_params = inspect.signature(least_squares).parameters
    _has_cb = "callback" in _ls_params
    _has_workers = "workers" in _ls_params

    def _lm_extra(lo, hi, msg, pool):
        extra = {}
        if _has_workers and pool is not None:
            extra["workers"] = pool.map
        if _has_cb:
            def _cb(intermediate_result):
                if cancel_event is not None and cancel_event.is_set():
                    raise StopIteration  # SciPy stops cleanly (status -2)
                nfev = getattr(intermediate_result, "nfev", None)
                if nfev is not None:
                    _report(progress_cb,
                            lo + (hi - lo) * min(1.0, nfev / _MAX_NFEV), msg)
            extra["callback"] = _cb
        return extra

    n_jac = max(1, min(os.cpu_count() or 4, 16))
    lm1_lo = boot_hi + 0.05
    with ThreadPoolExecutor(max_workers=n_jac) as jac_ex:
        # stage 1: robust loss - initial predictions for extreme rim points
        # can be far off (the bootstrap polynomial is only trustworthy
        # inside its fit range); soft_l1 keeps them from dominating
        sol = least_squares(residuals, x0, jac_sparsity=sparsity, method="trf",
                            x_scale="jac", max_nfev=_MAX_NFEV, ftol=1e-10,
                            xtol=1e-10, loss="soft_l1", f_scale=2.0, verbose=0,
                            **_lm_extra(lm1_lo, 0.62, "OCam Feinabgleich 1 …",
                                        jac_ex))
        _check_cancel(cancel_event)

        # stage 2: drop residual outliers under the converged model, then
        # polish with a plain least-squares fit
        m1, _ = unpack(sol.x)
        res1 = np.linalg.norm(residuals(sol.x).reshape(-1, 2), axis=1)
        rms1 = float(np.sqrt(np.mean(res1 ** 2)))
        keep_thr = max(4.0 * rms1, 3.0)
        cleaned_views = []
        for i, v in enumerate(used_views):
            keep = res1[offsets[i]:offsets[i + 1]] < keep_thr
            if keep.sum() >= 6:
                cleaned_views.append(ViewObservation(
                    v.image_points[keep], v.object_points[keep],
                    marker_ids=[mid for mid, kf in zip(v.marker_ids, keep) if kf]
                    if v.marker_ids else None, timestamp=v.timestamp))
            else:
                cleaned_views.append(v)
        if sum(len(v) for v in cleaned_views) != int(offsets[-1]):
            used_views = cleaned_views
            counts = [len(v) for v in used_views]
            offsets = np.concatenate([[0], np.cumsum(counts)])
            n_res = int(offsets[-1]) * 2
            sparsity = lil_matrix((n_res, len(x0)), dtype=int)
            sparsity[:, :9] = 1
            for i in range(n_views):
                sparsity[offsets[i] * 2:offsets[i + 1] * 2, 9 + 6 * i: 15 + 6 * i] = 1
        _report(progress_cb, 0.66, "OCam Feinabgleich 2 …")
        sol = least_squares(residuals, sol.x, jac_sparsity=sparsity,
                            method="trf", x_scale="jac", max_nfev=_MAX_NFEV,
                            ftol=1e-10, xtol=1e-10, verbose=0,
                            **_lm_extra(0.70, 0.98, "OCam Feinabgleich 2 …",
                                        jac_ex))
    _check_cancel(cancel_event)

    m, exts = unpack(sol.x)
    per_view_rms, per_point, per_reproj, rvecs, tvecs = [], [], [], [], []
    for i, v in enumerate(used_views):
        R, _ = cv2.Rodrigues(exts[i, :3])
        pc = v.object_points.astype(np.float64) @ R.T + exts[i, 3:]
        proj = m.world2cam(pc)
        err = np.linalg.norm(proj - v.image_points, axis=1)
        per_point.append(err)
        per_reproj.append(proj)
        per_view_rms.append(float(np.sqrt(np.mean(err ** 2))))
        rvecs.append(exts[i, :3].reshape(3, 1))
        tvecs.append(exts[i, 3:].reshape(3, 1))
    rms = float(np.sqrt(np.mean(np.concatenate(per_point) ** 2)))

    _report(progress_cb, 1.0, "Fertig")
    return OcamCalibrationResult(
        model=m, rms=rms, per_view_rms=per_view_rms,
        n_views=n_views, n_points=int(offsets[-1]),
        rvecs=rvecs, tvecs=tvecs, per_point_errors=per_point,
        used_image_points=[v.image_points for v in used_views],
        used_reprojections=per_reproj, kb=kb)


def _kb_theta_max(kb: CalibrationResult, r_max: float | None = None) -> float:
    """Off-axis angle of the given image radius under the KB model
    (default: largest radius among the KB observations)."""
    if r_max is None:
        r_max = 0.0
        for pts in kb.used_image_points:
            r = np.hypot(pts[:, 0] - kb.cx, pts[:, 1] - kb.cy)
            r_max = max(r_max, float(r.max()))
    k = kb.dist_coeffs
    f_iso = 0.5 * (kb.fx + kb.fy)
    ths = np.linspace(0.01, np.pi * 0.65, 2048)
    th_d = ths * (1 + k[0] * ths**2 + k[1] * ths**4 + k[2] * ths**6 + k[3] * ths**8)
    rr = f_iso * th_d
    cut = np.argmax(np.diff(rr) <= 0) or len(rr) - 1
    return float(np.interp(r_max, rr[:cut], ths[:cut]))
