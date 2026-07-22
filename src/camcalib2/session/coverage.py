"""Sensor coverage tracking - the "FaceTime" progress model.

The sensor is divided into a grid of cells.  Every accepted observation
point marks its cell as covered.  In addition the board tilt direction is
tracked in a small histogram so the UI can ask for angled views (pure
frontal views cannot constrain the distortion model well).

Coverage is confined to a user-defined region of interest (:class:`Roi`):
cells outside the ROI are neither counted nor required for full coverage,
so "100 %" means every *usable* cell has been seen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Roi:
    """Region of interest in full-sensor pixel coordinates.

    Detection and calibration are confined to this rectangle.  For fisheye
    it defaults to a centered square (the usable image circle sits inside
    it); for perspective it defaults to the whole sensor.
    """

    x: float
    y: float
    w: float
    h: float

    @property
    def x1(self) -> float:
        return self.x + self.w

    @property
    def y1(self) -> float:
        return self.y + self.h

    @classmethod
    def full(cls, image_size: tuple[int, int]) -> "Roi":
        w, h = image_size
        return cls(0.0, 0.0, float(w), float(h))

    @classmethod
    def default(cls, image_size: tuple[int, int], model) -> "Roi":
        """Model-specific initial ROI: fisheye -> centered square of full
        height, perspective -> the whole sensor."""
        from ..calibration import CameraModel

        w, h = image_size
        if model is CameraModel.FISHEYE:
            side = float(min(w, h))
            return cls((w - side) / 2.0, (h - side) / 2.0, side, side)
        return cls.full(image_size)

    def clamped(self, image_size: tuple[int, int], min_size: float = 16.0) -> "Roi":
        """Keep the ROI fully inside the sensor with a minimum edge size."""
        w, h = image_size
        rw = float(np.clip(self.w, min_size, w))
        rh = float(np.clip(self.h, min_size, h))
        rx = float(np.clip(self.x, 0.0, w - rw))
        ry = float(np.clip(self.y, 0.0, h - rh))
        return Roi(rx, ry, rw, rh)

    def bbox_int(self, image_size: tuple[int, int],
                 pad: float = 0.0) -> tuple[int, int, int, int]:
        """Integer ``(x0, y0, x1, y1)`` crop box, padded and clamped to the
        frame (half-open, suitable for ``gray[y0:y1, x0:x1]``)."""
        w, h = image_size
        x0 = int(np.clip(np.floor(self.x - pad), 0, w - 1))
        y0 = int(np.clip(np.floor(self.y - pad), 0, h - 1))
        x1 = int(np.clip(np.ceil(self.x1 + pad), x0 + 1, w))
        y1 = int(np.clip(np.ceil(self.y1 + pad), y0 + 1, h))
        return x0, y0, x1, y1

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Boolean ``(N,)`` mask of points whose centers lie inside the ROI."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if pts.size == 0:
            return np.zeros(0, dtype=bool)
        return ((pts[:, 0] >= self.x) & (pts[:, 0] < self.x1)
                & (pts[:, 1] >= self.y) & (pts[:, 1] < self.y1))


def cells_in_roi(image_size: tuple[int, int], grid: tuple[int, int],
                 roi: Roi | None) -> np.ndarray:
    """``(rows, cols)`` bool mask of grid cells that intersect the ROI.

    A cell is eligible when its pixel rectangle overlaps the ROI at all;
    ``roi is None`` marks every cell eligible.
    """
    cols, rows = grid
    if roi is None:
        return np.ones((rows, cols), dtype=bool)
    w, h = image_size
    cx0 = np.arange(cols) * (w / cols)
    cx1 = (np.arange(cols) + 1) * (w / cols)
    cy0 = np.arange(rows) * (h / rows)
    cy1 = (np.arange(rows) + 1) * (h / rows)
    col_hit = (cx1 > roi.x) & (cx0 < roi.x1)  # (cols,)
    row_hit = (cy1 > roi.y) & (cy0 < roi.y1)  # (rows,)
    return row_hit[:, None] & col_hit[None, :]


@dataclass
class CoverageMap:
    image_size: tuple[int, int]  # (w, h)
    grid: tuple[int, int] = (16, 12)  # (cols, rows)
    counts: np.ndarray = field(init=False)
    #: tilt direction histogram: 8 directions + 1 frontal bin
    tilt_bins: np.ndarray = field(init=False)
    roi: Roi | None = None

    def __post_init__(self):
        cols, rows = self.grid
        self.counts = np.zeros((rows, cols), dtype=np.int32)
        self.tilt_bins = np.zeros(9, dtype=np.int32)
        self._roi_cells = cells_in_roi(self.image_size, self.grid, self.roi)

    # ------------------------------------------------------------------
    def set_roi(self, roi: Roi | None) -> None:
        """Restrict coverage to ``roi``.  ``counts`` are kept, so editing
        the ROI live never discards already-collected progress."""
        self.roi = roi
        self._roi_cells = cells_in_roi(self.image_size, self.grid, roi)

    def roi_cell_mask(self) -> np.ndarray:
        """Boolean (rows, cols) array of cells that belong to the ROI."""
        return self._roi_cells

    def _cells(self, points: np.ndarray) -> np.ndarray:
        w, h = self.image_size
        cols, rows = self.grid
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        cx = np.clip((pts[:, 0] / w * cols).astype(int), 0, cols - 1)
        cy = np.clip((pts[:, 1] / h * rows).astype(int), 0, rows - 1)
        return cy * cols + cx

    def _eligible(self, cells: np.ndarray) -> np.ndarray:
        """Keep only flat cell indices that fall inside the ROI."""
        return cells[self._roi_cells.ravel()[cells]]

    def add_points(self, points: np.ndarray) -> int:
        """Mark cells covered; returns the number of newly covered cells."""
        if len(points) == 0:
            return 0
        cells = self._eligible(np.unique(self._cells(points)))
        if cells.size == 0:
            return 0
        flat = self.counts.ravel()
        fresh = int((flat[cells] == 0).sum())
        flat[cells] += 1
        return fresh

    def new_cells(self, points: np.ndarray) -> int:
        """How many still-uncovered in-ROI cells the points would hit."""
        if len(points) == 0:
            return 0
        cells = self._eligible(np.unique(self._cells(points)))
        if cells.size == 0:
            return 0
        return int((self.counts.ravel()[cells] == 0).sum())

    def add_tilt(self, tilt_dir: float | None, tilt_mag: float) -> None:
        """Record board tilt: direction in rad, magnitude 0..1 (0=frontal)."""
        if tilt_mag < 0.15 or tilt_dir is None:
            self.tilt_bins[8] += 1
        else:
            b = int(((tilt_dir + np.pi) / (2 * np.pi) * 8)) % 8
            self.tilt_bins[b] += 1

    # ------------------------------------------------------------------
    @property
    def fraction(self) -> float:
        elig = self._roi_cells
        n = int(elig.sum())
        if n == 0:
            return 0.0
        return float(((self.counts > 0) & elig).sum() / n)

    @property
    def tilt_fraction(self) -> float:
        """Fraction of the 8 tilt directions seen at least once."""
        return float((self.tilt_bins[:8] > 0).mean())

    def mask(self) -> np.ndarray:
        """Boolean (rows, cols) array of covered cells."""
        return self.counts > 0
