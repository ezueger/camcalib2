"""Sensor coverage tracking - the "FaceTime" progress model.

The sensor is divided into a grid of cells.  Every accepted observation
point marks its cell as covered.  In addition the board tilt direction is
tracked in a small histogram so the UI can ask for angled views (pure
frontal views cannot constrain the distortion model well).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CoverageMap:
    image_size: tuple[int, int]  # (w, h)
    grid: tuple[int, int] = (16, 12)  # (cols, rows)
    counts: np.ndarray = field(init=False)
    #: tilt direction histogram: 8 directions + 1 frontal bin
    tilt_bins: np.ndarray = field(init=False)

    def __post_init__(self):
        cols, rows = self.grid
        self.counts = np.zeros((rows, cols), dtype=np.int32)
        self.tilt_bins = np.zeros(9, dtype=np.int32)

    # ------------------------------------------------------------------
    def _cells(self, points: np.ndarray) -> np.ndarray:
        w, h = self.image_size
        cols, rows = self.grid
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        cx = np.clip((pts[:, 0] / w * cols).astype(int), 0, cols - 1)
        cy = np.clip((pts[:, 1] / h * rows).astype(int), 0, rows - 1)
        return cy * cols + cx

    def add_points(self, points: np.ndarray) -> int:
        """Mark cells covered; returns the number of newly covered cells."""
        if len(points) == 0:
            return 0
        cells = np.unique(self._cells(points))
        flat = self.counts.ravel()
        fresh = int((flat[cells] == 0).sum())
        flat[cells] += 1
        return fresh

    def new_cells(self, points: np.ndarray) -> int:
        """How many still-uncovered cells the points would hit."""
        if len(points) == 0:
            return 0
        cells = np.unique(self._cells(points))
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
        return float((self.counts > 0).mean())

    @property
    def tilt_fraction(self) -> float:
        """Fraction of the 8 tilt directions seen at least once."""
        return float((self.tilt_bins[:8] > 0).mean())

    def mask(self) -> np.ndarray:
        """Boolean (rows, cols) array of covered cells."""
        return self.counts > 0
