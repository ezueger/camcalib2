"""Calibration target definitions (marker boards, checkerboards)."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np

from .dotcode import Codebook


@dataclass(frozen=True)
class MarkerBoard:
    """Coded dot-marker board: marker id -> measured 3D position (mm).

    The z coordinates carry the measured flatness deviation of the printed
    board, which is what makes the reference calibrations so accurate.
    """

    name: str
    markers: dict[int, tuple[float, float, float]]
    codebook: Codebook = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "codebook", Codebook(self.markers.keys()))

    def object_points(self, marker_ids) -> np.ndarray:
        """(N,3) float32 array of board coordinates for the given ids."""
        return np.array([self.markers[m] for m in marker_ids], dtype=np.float32)

    @classmethod
    def from_json(cls, path_or_data) -> "MarkerBoard":
        if isinstance(path_or_data, (str, Path)):
            data = json.loads(Path(path_or_data).read_text())
        else:
            data = path_or_data
        markers = {int(k): tuple(v) for k, v in data["markers"].items()}
        return cls(name=data.get("name", "marker-board"), markers=markers)

    @classmethod
    def builtin(cls, name: str = "vioso_board_196") -> "MarkerBoard":
        ref = resources.files("camcalib2.patterns") / "resources" / f"{name}.json"
        return cls.from_json(json.loads(ref.read_text()))

    @classmethod
    def from_config_xml(cls, path, name: str | None = None) -> "MarkerBoard":
        """Load marker definitions from a legacy calibration project config.xml."""
        root = ET.parse(str(path)).getroot()
        markers = {}
        for m in root.iter("Marker"):
            code = int(m.findtext("marker_code"))
            markers[code] = (
                float(m.findtext("coordinate_wks_mm_x")),
                float(m.findtext("coordinate_wks_mm_y")),
                float(m.findtext("coordinate_wks_mm_z")),
            )
        return cls(name=name or Path(path).stem, markers=markers)

    def to_json(self, path) -> None:
        data = {
            "name": self.name,
            "markers": {str(k): list(v) for k, v in sorted(self.markers.items())},
        }
        Path(path).write_text(json.dumps(data, indent=1))


@dataclass(frozen=True)
class Checkerboard:
    """Classic checkerboard target.

    ``inner_corners`` is (cols, rows) of *inner* corners,
    ``square_size`` in mm.
    """

    inner_corners: tuple[int, int]
    square_size: float
    name: str = "checkerboard"

    def object_points(self) -> np.ndarray:
        cols, rows = self.inner_corners
        grid = np.zeros((rows * cols, 3), np.float32)
        grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * self.square_size
        return grid
