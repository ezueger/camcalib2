"""Detector for circular coded dot markers.

Pipeline per frame:

1. adaptive threshold + connected components (optionally on a downscaled
   working image for speed),
2. candidate centers = larger elliptical blobs; ring dots = small blobs
   in an annulus around a center,
3. local affine normalization using the center blob's second moments
   (compensates perspective/fisheye tilt of the marker plane),
4. angular quantization of the ring dots onto the 14-slot grid,
5. rotation-invariant codebook lookup,
6. sub-pixel refinement of the center via intensity-weighted centroid
   on the full-resolution image.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..patterns.board import MarkerBoard
from ..patterns.dotcode import N_SLOTS, N_RING_DOTS, QUIET_SLOT, SYNC_DOT_SLOTS


@dataclass
class DetectedMarker:
    marker_id: int
    center: tuple[float, float]  # sub-pixel, full-resolution coordinates
    ring_dots: int
    slot_rotation: int  # observed rotation vs canonical orientation
    residual: float  # rms angular slot residual (slot units)


@dataclass
class DotMarkerDetectorConfig:
    #: downscale factor applied for blob search (1 = full resolution)
    work_scale: float = 0.5
    #: adaptive threshold window (odd, in working-image pixels)
    thresh_block: int = 51
    thresh_c: float = 12.0
    #: min/max center blob area in *full-resolution* pixels
    min_center_area: float = 40.0
    max_center_area: float = 12000.0
    #: ring dot area relative to center area
    min_dot_area_ratio: float = 0.005
    max_dot_area_ratio: float = 0.35
    #: annulus (relative to center equivalent radius) searched for ring dots
    min_ring_ratio: float = 1.25
    max_ring_ratio: float = 2.6
    #: max rms slot residual accepted for decoding
    max_slot_residual: float = 0.22
    #: blobs bigger than this multiple of the median dot area are split in two
    split_dot_factor: float = 1.7
    #: full-resolution sub-pixel center refinement. Costs ~0.5 ms per
    #: marker - disable for live preview (coverage/UI), enable for
    #: keyframes that feed the calibration.
    refine_subpixel: bool = True


def identify_board(gray, boards: list[MarkerBoard] | None = None,
                   config: "DotMarkerDetectorConfig | None" = None):
    """Detect which known marker board is visible in the image.

    Boards may share the same id layout (different measured prints) or be
    id-subsets of each other, so the detection count alone is ambiguous.
    Candidates with the same (maximum) count are disambiguated by the
    planar-homography residual of their measured 2D geometry - the
    per-print measurement differences (up to ~1.7 mm) are well above the
    detector noise.  Returns ``(board, n_detected)`` or ``(None, 0)``.
    """
    if boards is None:
        boards = [MarkerBoard.builtin(n) for n in MarkerBoard.builtin_names()]
    detections = []
    best_n = 0
    for board in boards:
        markers = DotMarkerDetector(board, config).detect(gray)
        detections.append(markers)
        best_n = max(best_n, len(markers))
    if best_n < 12:
        best = max(zip(boards, detections), key=lambda bd: len(bd[1]), default=(None, []))
        return (best[0], len(best[1])) if best[1] else (None, 0)

    candidates = [(b, d) for b, d in zip(boards, detections) if len(d) == best_n]
    if len(candidates) == 1:
        return candidates[0][0], best_n

    best, best_res = None, np.inf
    for board, markers in candidates:
        obj = board.object_points([m.marker_id for m in markers])[:, :2]
        img = np.array([m.center for m in markers], np.float64)
        H, _ = cv2.findHomography(obj, img, cv2.RANSAC, 5.0)
        if H is None:
            continue
        proj = cv2.perspectiveTransform(obj.reshape(-1, 1, 2), H).reshape(-1, 2)
        res = float(np.median(np.linalg.norm(proj - img, axis=1)))
        if res < best_res:
            best, best_res = board, res
    return (best, best_n) if best is not None else (None, 0)


class DotMarkerDetector:
    def __init__(self, board: MarkerBoard, config: DotMarkerDetectorConfig | None = None):
        self.board = board
        self.cfg = config or DotMarkerDetectorConfig()

    # ------------------------------------------------------------------
    def detect(self, gray: np.ndarray) -> list[DetectedMarker]:
        cfg = self.cfg
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        scale = cfg.work_scale
        if scale != 1.0:
            work = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            work = gray

        block = cfg.thresh_block | 1
        binary = cv2.adaptiveThreshold(
            work, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block, cfg.thresh_c)

        n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if n <= 1:
            return []

        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64) / (scale * scale)
        cents = centroids[1:] / scale  # full-resolution coordinates
        min_center_area_w = cfg.min_center_area
        max_center_area_w = cfg.max_center_area

        center_idx = np.nonzero(
            (areas >= min_center_area_w) & (areas <= max_center_area_w))[0]
        if center_idx.size == 0:
            return []

        # grid index over all blobs for fast annulus queries
        cell = 16.0
        grid: dict[tuple[int, int], list[int]] = {}
        for i in range(len(cents)):
            key = (int(cents[i][0] // cell), int(cents[i][1] // cell))
            grid.setdefault(key, []).append(i)

        markers: list[DetectedMarker] = []
        used_centers: set[int] = set()
        for ci in center_idx:
            if ci in used_centers:
                continue
            marker = self._try_marker(gray, labels, stats, ci, areas, cents, grid, cell, scale)
            if marker is not None:
                markers.append(marker)

        # deduplicate ids (keep lowest residual)
        best: dict[int, DetectedMarker] = {}
        for m in markers:
            prev = best.get(m.marker_id)
            if prev is None or m.residual < prev.residual:
                best[m.marker_id] = m
        return sorted(best.values(), key=lambda m: m.marker_id)

    # ------------------------------------------------------------------
    def _try_marker(self, gray, labels, stats, ci, areas, cents, grid, cell, scale):
        cfg = self.cfg
        c = cents[ci]
        area = areas[ci]
        r0 = np.sqrt(area / np.pi)
        rmax = cfg.max_ring_ratio * r0 * 1.4

        # collect nearby small blobs
        k0x, k1x = int((c[0] - rmax) // cell), int((c[0] + rmax) // cell)
        k0y, k1y = int((c[1] - rmax) // cell), int((c[1] + rmax) // cell)
        nearby = []
        for kx in range(k0x, k1x + 1):
            for ky in range(k0y, k1y + 1):
                nearby.extend(grid.get((kx, ky), ()))

        # affine normalization from center blob second moments
        A_inv, c_ref = self._center_shape(gray, labels, stats, ci, scale)
        if A_inv is None:
            return None

        dots = []  # (angle, radius_norm, area, offset)
        for bi in nearby:
            if bi == ci:
                continue
            da = areas[bi]
            if not (cfg.min_dot_area_ratio * area <= da <= cfg.max_dot_area_ratio * area):
                continue
            off = cents[bi] - c_ref
            u = A_inv @ off
            rn = float(np.hypot(*u))  # in units of center radius
            if cfg.min_ring_ratio <= rn <= cfg.max_ring_ratio:
                dots.append((float(np.arctan2(u[1], u[0])), rn, da, bi))
        if len(dots) < N_RING_DOTS - 2:
            return None

        # tighten annulus around the median ring radius
        med_rn = float(np.median([d[1] for d in dots]))
        dots = [d for d in dots if abs(d[1] - med_rn) < 0.35 * med_rn]
        if not (N_RING_DOTS - 2 <= len(dots) <= N_RING_DOTS + 1):
            return None

        # split merged dot pairs (adjacent dots fused by blur)
        med_area = float(np.median([d[2] for d in dots]))
        if len(dots) < N_RING_DOTS:
            dots = self._split_merged(dots, med_area, med_rn, labels, stats, cents, A_inv, c_ref, scale)
        if len(dots) != N_RING_DOTS:
            return None

        angles = np.array([d[0] for d in dots])
        pitch = 2 * np.pi / N_SLOTS
        phi = angles / pitch
        # global phase via circular mean of fractional parts
        frac = phi * 2 * np.pi  # one turn per slot
        phase = np.arctan2(np.sin(frac).sum(), np.cos(frac).sum()) / (2 * np.pi)
        slots_f = phi - phase
        slots = np.round(slots_f).astype(int) % N_SLOTS
        resid = slots_f - np.round(slots_f)
        rms = float(np.sqrt(np.mean(resid ** 2)))
        if rms > cfg.max_slot_residual or len(set(slots.tolist())) != N_RING_DOTS:
            return None

        bits = [0] * N_SLOTS
        for s in slots:
            bits[int(s)] = 1
        decoded = self.board.codebook.decode(bits)
        if decoded is None:
            return None
        marker_id, rot = decoded

        if cfg.refine_subpixel:
            center = self._refine_center(gray, labels, stats, ci, scale)
        else:
            center = (float(c_ref[0]), float(c_ref[1]))
        return DetectedMarker(marker_id=marker_id, center=center,
                              ring_dots=len(dots), slot_rotation=rot, residual=rms)

    # ------------------------------------------------------------------
    def _center_shape(self, gray, labels, stats, ci, scale):
        """Inverse affine normalization matrix from the center blob's
        second moments, plus the blob centroid (full-res coordinates)."""
        li = ci + 1  # label index (0 is background)
        x, y, w, h = (stats[li, cv2.CC_STAT_LEFT], stats[li, cv2.CC_STAT_TOP],
                      stats[li, cv2.CC_STAT_WIDTH], stats[li, cv2.CC_STAT_HEIGHT])
        mask = (labels[y:y + h, x:x + w] == li).astype(np.uint8)
        m = cv2.moments(mask, binaryImage=True)
        if m["m00"] < 4:
            return None, None
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        mu20 = m["mu20"] / m["m00"]
        mu02 = m["mu02"] / m["m00"]
        mu11 = m["mu11"] / m["m00"]
        cov = np.array([[mu20, mu11], [mu11, mu02]])
        evals, evecs = np.linalg.eigh(cov)
        if evals[0] <= 1e-9:
            return None, None
        # a filled ellipse with semi-axes (a,b) has eigenvalues (a^2/4, b^2/4);
        # normalize so the unit shape has radius == equivalent circle radius
        r_equiv = float(np.sqrt(m["m00"] / np.pi))
        axes = 2.0 * np.sqrt(evals)  # semi-axes
        A = evecs @ np.diag(axes / r_equiv) @ evecs.T
        A_inv = np.linalg.inv(A) / r_equiv  # maps offsets to center-radius units
        c_ref = np.array([(x + cx) / scale, (y + cy) / scale])
        return A_inv * scale, c_ref

    # ------------------------------------------------------------------
    def _split_merged(self, dots, med_area, med_rn, labels, stats, cents, A_inv, c_ref, scale):
        """Split ring blobs that are two fused dots into two entries."""
        out = []
        for ang, rn, da, bi in dots:
            if da > self.cfg.split_dot_factor * med_area and len(dots) < N_RING_DOTS:
                li = bi + 1
                x, y, w, h = (stats[li, cv2.CC_STAT_LEFT], stats[li, cv2.CC_STAT_TOP],
                              stats[li, cv2.CC_STAT_WIDTH], stats[li, cv2.CC_STAT_HEIGHT])
                mask = (labels[y:y + h, x:x + w] == li).astype(np.uint8)
                m = cv2.moments(mask, binaryImage=True)
                if m["m00"] > 4:
                    mu20, mu02, mu11 = m["mu20"] / m["m00"], m["mu02"] / m["m00"], m["mu11"] / m["m00"]
                    evals, evecs = np.linalg.eigh(np.array([[mu20, mu11], [mu11, mu02]]))
                    major = evecs[:, 1] * np.sqrt(max(evals[1], 1e-9))
                    cxy = np.array([x + m["m10"] / m["m00"], y + m["m01"] / m["m00"]]) / scale
                    for sign in (-1.0, 1.0):
                        p = cxy + sign * major / scale
                        u = A_inv @ (p - c_ref)
                        out.append((float(np.arctan2(u[1], u[0])), float(np.hypot(*u)), da / 2, bi))
                    continue
            out.append((ang, rn, da, bi))
        return out

    # ------------------------------------------------------------------
    def _refine_center(self, gray, labels, stats, ci, scale):
        """Sub-pixel centroid of the center dot on the full-resolution image.

        Re-thresholds locally (Otsu) at full resolution and computes an
        intensity-weighted centroid over the dilated center component.
        Validated against the reference software's reprojections this
        yields ~0.15 px noise.
        """
        li = ci + 1
        x = stats[li, cv2.CC_STAT_LEFT]
        y = stats[li, cv2.CC_STAT_TOP]
        w = stats[li, cv2.CC_STAT_WIDTH]
        h = stats[li, cv2.CC_STAT_HEIGHT]
        cx0 = (x + w / 2.0) / scale
        cy0 = (y + h / 2.0) / scale
        win = int(max(w, h) / scale * 0.75) + 4
        x0 = max(0, int(cx0) - win)
        y0 = max(0, int(cy0) - win)
        x1 = min(gray.shape[1], int(cx0) + win + 1)
        y1 = min(gray.shape[0], int(cy0) + win + 1)
        patch = gray[y0:y1, x0:x1]
        if patch.size < 16:
            return (cx0, cy0)
        binary = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        n, lab, st, ce = cv2.connectedComponentsWithStats(binary)
        if n < 2:
            return (cx0, cy0)
        pcx, pcy = cx0 - x0, cy0 - y0
        # component nearest to the expected center, weighted towards large ones
        best = max(range(1, n), key=lambda i: st[i, cv2.CC_STAT_AREA]
                   - 10.0 * np.hypot(ce[i][0] - pcx, ce[i][1] - pcy))
        if np.hypot(ce[best][0] - pcx, ce[best][1] - pcy) > win * 0.5:
            return (cx0, cy0)
        mask = (lab == best).astype(np.uint8)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        pf = patch.astype(np.float32)
        outside = pf[mask == 0]
        bg = float(np.percentile(outside, 75)) if outside.size else float(pf.max())
        wgt = np.clip(bg - pf, 0, None) * mask
        total = float(wgt.sum())
        if total < 1e-6:
            return (float(x0 + ce[best][0]), float(y0 + ce[best][1]))
        ys, xs = np.mgrid[0:pf.shape[0], 0:pf.shape[1]]
        return (float((wgt * xs).sum() / total) + x0,
                float((wgt * ys).sum() / total) + y0)
