"""Parsers for the legacy calibration software's file formats.

* ``config.xml``   - project file: camera info, native image size, target
  definition (dot-marker board with measured 3D coordinates, or a
  checkerboard with square size) and the per-image detections.
* ``<id>-ocv.xml``  - pinhole result (see export.py for conventions).
* ``<id>-ocam.xml`` - fisheye result in the Scaramuzza/OCamCalib model:
  ray(u,v) = (u, v, f(r)), f(r) = a0 + a2 r^2 + a3 r^3 + a4 r^4 with
  (u,v) relative to (cx,cy) after the affine correction [[c,d],[e,1]].
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..patterns.board import MarkerBoard


@dataclass
class LegacyDetections:
    """One image's detections: marker/corner id -> pixel position."""

    image: str
    points: dict[int, tuple[float, float]]
    reproj: dict[int, tuple[float, float]] = field(default_factory=dict)


@dataclass
class LegacyConfig:
    path: Path
    camera_serial: str
    camera_model: str
    image_size: tuple[int, int]  # native (w, h)
    pixel_size: tuple[float, float] | None
    #: dot-marker board (measured per print) or None for checkerboards
    board: MarkerBoard | None
    #: checkerboard square size (mm) or None for marker boards
    square_size: tuple[float, float] | None
    detections: list[LegacyDetections]

    @property
    def target_type(self) -> str:
        return "dots" if self.board is not None else "checker"

    def object_points_for(self, ids) -> np.ndarray:
        if self.board is not None:
            return self.board.object_points(ids)
        # checkerboard: id = row * 10 + col (10x10 grid)
        sx, sy = self.square_size
        out = np.array([((i % 10) * sx, (i // 10) * sy, 0.0) for i in ids],
                       dtype=np.float32)
        return out


def load_config_xml(path) -> LegacyConfig:
    path = Path(path)
    root = ET.parse(str(path)).getroot()

    w = int(root.findtext("ImageSize_Width"))
    h = int(root.findtext("ImageSize_Height"))
    px = root.findtext("Pixel_dx")
    py = root.findtext("Pixel_dy")
    pixel_size = (float(px), float(py)) if px and py else None

    board = None
    if root.find(".//Marker") is not None:
        board = MarkerBoard.from_config_xml(path)

    sq = None
    sx, sy = root.findtext(".//squareSizeX"), root.findtext(".//squareSizeY")
    if sx and sy:
        sq = (float(sx), float(sy))

    names = [im.text for im in root.find("CalibrationPatternImages")]
    detections = []
    for i, pi in enumerate(root.iter("PatternImage")):
        pts, rep = {}, {}
        for cc in pi.iter("CCMarker"):
            if cc.findtext("pixelPositionInvalid") != "0":
                continue
            mid = int(cc.findtext("id"))
            x, y = (float(v) for v in cc.findtext("pixel").split())
            pts[mid] = (x, y)
            r = cc.findtext("reproj")
            if r:
                rx, ry = (float(v) for v in r.split())
                rep[mid] = (rx, ry)
        detections.append(LegacyDetections(
            image=names[i] if i < len(names) else f"{i:03d}",
            points=pts, reproj=rep))

    return LegacyConfig(
        path=path,
        camera_serial=root.findtext("CameraSerialNumber") or "",
        camera_model=root.findtext("CameraModel") or "",
        image_size=(w, h), pixel_size=pixel_size,
        board=board, square_size=sq, detections=detections)


@dataclass
class OcvResult:
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray  # k1 k2 k3 p1 p2 (legacy order)
    rms: float
    rad: float

    @property
    def dist_opencv(self) -> np.ndarray:
        k1, k2, k3, p1, p2 = self.dist
        return np.array([k1, k2, p1, p2, k3])


def load_ocv_xml(path) -> OcvResult:
    ocv = ET.parse(str(path)).getroot().find("camera-ocv")
    f = [float(v) for v in ocv.findtext("f_xy").split()]
    c = [float(v) for v in ocv.findtext("c_xy").split()]
    d = np.array([float(v) for v in ocv.findtext("dist_coeffs").split()])
    return OcvResult(fx=f[0], fy=f[1], cx=c[0], cy=c[1], dist=d,
                     rms=float(ocv.findtext("residual_error")),
                     rad=float(ocv.findtext("rad") or 1.0))


@dataclass
class OcamResult:
    cx: float
    cy: float
    c: float
    d: float
    e: float
    poly: np.ndarray  # a0, a2, a3, a4  (a1 == 0)
    rad: float

    def f(self, r):
        a0, a2, a3, a4 = self.poly
        return a0 + a2 * r ** 2 + a3 * r ** 3 + a4 * r ** 4

    def theta(self, r):
        """Incidence angle (rad from optical axis) for image radius r."""
        return np.arctan2(r, self.f(r))

    def r_of_theta(self, thetas, r_max=4000.0, n=32768):
        """Numerically invert theta(r) on a dense grid."""
        rs = np.linspace(0.0, r_max, n)
        th = self.theta(rs)
        # theta(r) is monotonically increasing within the valid range
        cut = np.argmax(np.diff(th) <= 0) or len(th) - 1
        return np.interp(thetas, th[:cut], rs[:cut])


def load_ocam_xml(path) -> OcamResult:
    root = ET.parse(str(path)).getroot()
    g = root.findtext
    return OcamResult(
        cx=float(g("cx")), cy=float(g("cy")),
        c=float(g("c")), d=float(g("d")), e=float(g("e")),
        poly=np.array([float(g("a0")), float(g("a2")),
                       float(g("a3")), float(g("a4"))]),
        rad=float(g("rad") or 1.0))


def find_result_xml(folder: Path):
    """Locate a result XML next to/inside a dataset folder.

    Returns ("ocv"|"ocam", parsed result) or (None, None).
    """
    folder = Path(folder)
    for pattern, kind, loader in (("*-ocv.xml", "ocv", load_ocv_xml),
                                  ("*-ocam.xml", "ocam", load_ocam_xml)):
        for base in (folder, folder / "Results"):
            hits = sorted(base.glob(pattern)) if base.is_dir() else []
            if hits:
                return kind, loader(hits[0])
    return None, None
