"""Result export: legacy vendor XML, OpenCV YAML, coverage image.

Conventions of the legacy format (determined from reference data):

* ``dist_coeffs`` order is ``k1 k2 k3 p1 p2`` (NOT OpenCV's k1 k2 p1 p2 k3),
* pixel coordinates use a corner-origin convention, i.e. legacy values =
  OpenCV values + 0.5 (verified against reference detections, constant
  offset (+0.5, +0.5) with ~0.12 px spread),
* ``<f>`` is the focal length in mm (fx * pixel size),
* ``<rad>`` is the radius of the calibrated/covered region relative to
  the half diagonal of the sensor.
"""

from __future__ import annotations

import datetime as _dt
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from ..calibration import CalibrationResult, CameraModel


def _fmt(v: float) -> str:
    return f"{v:.6g}"


def coverage_radius(result: CalibrationResult, points: np.ndarray | None) -> float:
    """Max observed radius around the principal point / half diagonal."""
    w, h = result.image_size
    half_diag = float(np.hypot(w / 2.0, h / 2.0))
    if points is None or len(points) == 0:
        return 1.0
    r = np.hypot(points[:, 0] - result.cx, points[:, 1] - result.cy)
    return float(min(1.0, r.max() / half_diag))


def write_vendor_xml(result: CalibrationResult, path, camera_id: str,
                     pixel_size_mm: tuple[float, float] | None = None,
                     description: str | None = None,
                     points: np.ndarray | None = None,
                     legacy_pixel_origin: bool = True) -> None:
    """Write the legacy ``camera-calibration`` XML format."""
    if result.model is not CameraModel.PINHOLE:
        raise ValueError("vendor XML export is defined for the pinhole model")
    w, h = result.image_size
    off = 0.5 if legacy_pixel_origin else 0.0
    k1, k2, p1, p2, k3 = [float(v) for v in result.dist_coeffs[:5]]

    root = ET.Element("camera-calibration", version="1.0")
    ocv = ET.SubElement(root, "camera-ocv", version="2")
    pin = ET.SubElement(ocv, "pinhole-camera", version="2")
    ET.SubElement(pin, "id").text = camera_id
    ET.SubElement(pin, "extrinsic").text = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"
    if pixel_size_mm:
        f_mm = result.fx * pixel_size_mm[0]
        ET.SubElement(pin, "f").text = _fmt(f_mm)
        ET.SubElement(pin, "pixel_size").text = f"{pixel_size_mm[0]:g} {pixel_size_mm[1]:g}"
    ET.SubElement(pin, "resolution").text = f"{w} {h}"
    ET.SubElement(pin, "description").text = (
        description or _dt.date.today().isoformat())

    ET.SubElement(ocv, "f_xy").text = f"{_fmt(result.fx)} {_fmt(result.fy)}"
    ET.SubElement(ocv, "c_xy").text = f"{_fmt(result.cx + off)} {_fmt(result.cy + off)}"
    # legacy order: k1 k2 k3 p1 p2
    ET.SubElement(ocv, "dist_coeffs").text = " ".join(
        _fmt(v) for v in (k1, k2, k3, p1, p2))
    ET.SubElement(ocv, "residual_error").text = _fmt(result.rms)
    ET.SubElement(ocv, "rad").text = _fmt(coverage_radius(result, points))

    ET.indent(root)
    tree = ET.ElementTree(root)
    tree.write(str(path), encoding="UTF-8", xml_declaration=True)


def write_ocam_xml(ocam_result, path, camera_id: str,
                   lens_id: str = "00000000000",
                   points: np.ndarray | None = None,
                   description: str | None = None) -> None:
    """Write the legacy OCam (Scaramuzza) result format.

    ``ocam_result`` is a :class:`camcalib2.calibration.ocam.OcamCalibrationResult`.
    """
    m = ocam_result.model
    w, h = m.image_size
    root = ET.Element("calibration", version="1.0")
    ET.SubElement(root, "serialCamera").text = camera_id
    ET.SubElement(root, "serielLens").text = lens_id
    ET.SubElement(root, "cx").text = _fmt(m.cx)
    ET.SubElement(root, "cy").text = _fmt(m.cy)
    ET.SubElement(root, "c").text = _fmt(m.c)
    ET.SubElement(root, "d").text = _fmt(m.d)
    ET.SubElement(root, "e").text = _fmt(m.e)
    ET.SubElement(root, "a0").text = _fmt(m.poly[0])
    ET.SubElement(root, "a2").text = _fmt(m.poly[1])
    ET.SubElement(root, "a3").text = _fmt(m.poly[2])
    ET.SubElement(root, "a4").text = _fmt(m.poly[3])
    half_diag = float(np.hypot(w / 2.0, h / 2.0))
    if points is not None and len(points):
        r = np.hypot(points[:, 0] - m.cx, points[:, 1] - m.cy)
        rad = float(min(1.0, r.max() / half_diag))
    else:
        rad = 1.0
    ET.SubElement(root, "rad").text = _fmt(rad)
    ET.SubElement(root, "calDate").text = description or _dt.date.today().isoformat()
    ET.SubElement(root, "comment").text = (
        "- (cx, cy) is the intersection point of the optical achsis with the image plane in pixel coordinates \n"
        "    - c, d, e are affine transformation parametes describing the deviation of the fisheye image from a perfect circular \n"
        "    - a0, a2, a3, a4 are the coefficients of the polynomial which describes the camera system: "
        "f(r) = a0 + a2*r^2 + a3*r^3 + a4*r^4 \n"
        "    - rad is the confidence radius i.e. the distance from the outer most marker in one of the pattern "
        "images to the optical center")
    ET.indent(root)
    ET.ElementTree(root).write(str(path), encoding="UTF-8", xml_declaration=True)


def write_opencv_yaml(result: CalibrationResult, path) -> None:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    try:
        fs.write("model", result.model.value)
        fs.write("image_width", int(result.image_size[0]))
        fs.write("image_height", int(result.image_size[1]))
        fs.write("camera_matrix", result.camera_matrix)
        fs.write("distortion_coefficients", np.asarray(result.dist_coeffs).reshape(-1, 1))
        fs.write("rms", float(result.rms))
        fs.write("n_views", int(result.n_views))
    finally:
        fs.release()


def render_result_image(result: CalibrationResult,
                        views_points: list[np.ndarray],
                        views_errors: list[np.ndarray]) -> np.ndarray:
    """Coverage/error map in the style of the legacy ``result.jpg``:
    every observation drawn as a small circle colored by reprojection
    error (green <=1px, yellow <=2px, orange <=3px, red >3px) plus the
    calibrated-region boundary in blue."""
    w, h = result.image_size
    img = np.zeros((h, w, 3), np.uint8)

    colors = [(0, 200, 0), (0, 220, 220), (0, 140, 255), (0, 0, 255)]
    all_pts = []
    for pts, errs in zip(views_points, views_errors):
        all_pts.append(pts)
        for (x, y), e in zip(pts, errs):
            c = colors[0] if e <= 1 else colors[1] if e <= 2 else colors[2] if e <= 3 else colors[3]
            cv2.circle(img, (int(round(x)), int(round(y))), 4, c, 1, cv2.LINE_AA)
    pts = np.vstack(all_pts) if all_pts else None

    rad = coverage_radius(result, pts)
    half_diag = float(np.hypot(w / 2.0, h / 2.0))
    cv2.circle(img, (int(round(result.cx)), int(round(result.cy))),
               int(round(rad * half_diag)), (255, 80, 0), 2, cv2.LINE_AA)

    scale = max(1.0, h / 1000.0)
    legend = [("err<=1px", colors[0]), ("<=2px", colors[1]),
              ("<=3px", colors[2]), (">3px", colors[3])]
    x = int(10 * scale)
    for text, c in legend:
        cv2.putText(img, text, (x, h - int(12 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * scale, c, 1, cv2.LINE_AA)
        x += int((90 + 18 * len(text) / 8) * scale)
    return img


def export_all(result: CalibrationResult, out_dir, camera_id: str,
               views_points: list[np.ndarray], views_errors: list[np.ndarray],
               pixel_size_mm: tuple[float, float] | None = None,
               description: str | None = None,
               ocam_result=None) -> dict[str, Path]:
    """Write vendor XML (pinhole) / ocam XML (fisheye), OpenCV YAML and
    the result image."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    pts = np.vstack(views_points) if views_points else None
    if result.model is CameraModel.PINHOLE:
        p = out / f"{camera_id}-ocv.xml"
        write_vendor_xml(result, p, camera_id, pixel_size_mm, description, pts)
        written["vendor_xml"] = p
    if ocam_result is not None:
        p = out / f"{camera_id}-ocam.xml"
        write_ocam_xml(ocam_result, p, camera_id, points=pts, description=description)
        written["ocam_xml"] = p
    p = out / f"{camera_id}-opencv.yaml"
    write_opencv_yaml(result, p)
    written["opencv_yaml"] = p
    img = render_result_image(result, views_points, views_errors)
    p = out / "result.jpg"
    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    written["result_image"] = p
    return written
