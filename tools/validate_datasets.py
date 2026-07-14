#!/usr/bin/env python3
"""Validation suite: run detection + calibration over reference datasets.

Walks a directory of dataset folders (each: numbered images + optional
``config.xml`` / ``Results/<id>-ocv.xml`` from the legacy software),
auto-classifies the target type (coded dot board vs. checkerboard) and
the lens type (fisheye via image-circle vignetting), calibrates, and
prints a comparison table.  Generated ``result.jpg`` files are written
next to the table for visual comparison with the legacy output.

Usage::

    python tools/validate_datasets.py <datasets_root> [<out_dir>]
"""

from __future__ import annotations

import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from camcalib2.calibration import CameraModel, ViewObservation, calibrate
from camcalib2.detection.checkerboard import CheckerboardDetector
from camcalib2.detection.dot_marker import DotMarkerDetector, DotMarkerDetectorConfig
from camcalib2.io import export_all
from camcalib2.patterns.board import Checkerboard, MarkerBoard


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*.jpg") if p.parent.name != "Results")


def is_fisheye_circle(images: list[Path]) -> bool:
    """Detect a vignetted image circle.

    A dark studio background can make single-image corner checks
    misfire; outside a fisheye's image circle the corners are optically
    black in EVERY image, so sample several images and require it
    consistently.
    """
    idx = np.linspace(0, len(images) - 1, min(5, len(images))).astype(int)
    corner_max = 0.0
    center_ok = 0
    for i in idx:
        gray = cv2.imread(str(images[i]), cv2.IMREAD_GRAYSCALE)
        h, w = gray.shape
        k = max(8, h // 20)
        corners = [gray[:k, :k], gray[:k, -k:], gray[-k:, :k], gray[-k:, -k:]]
        corner_max = max(corner_max, max(float(c.mean()) for c in corners))
        center = gray[h // 2 - k:h // 2 + k, w // 2 - k:w // 2 + k].mean()
        if center > 40:
            center_ok += 1
    return corner_max < 12 and center_ok >= len(idx) // 2


def classify(images: list[Path], board: MarkerBoard):
    mid = cv2.imread(str(images[len(images) // 2]), cv2.IMREAD_GRAYSCALE)
    scale = 1.0 if mid.size < 4_000_000 else 0.5
    det = DotMarkerDetector(board, DotMarkerDetectorConfig(work_scale=scale))
    n_dots = len(det.detect(mid))
    if n_dots >= 20:
        return "dots", is_fisheye_circle(images), scale
    return "checker", is_fisheye_circle(images), scale


def load_reference_xml(folder: Path):
    for p in list(folder.glob("Results/*-ocv.xml")) + list(folder.glob("*-ocv.xml")):
        root = ET.parse(p).getroot()
        ocv = root.find("camera-ocv")
        if ocv is None:
            continue
        f_xy = [float(v) for v in ocv.findtext("f_xy").split()]
        c_xy = [float(v) for v in ocv.findtext("c_xy").split()]
        rms = float(ocv.findtext("residual_error"))
        return {"fx": f_xy[0], "fy": f_xy[1], "cx": c_xy[0], "cy": c_xy[1], "rms": rms}
    return None


def run_dataset(folder: Path, out_root: Path, board: MarkerBoard):
    images = list_images(folder)
    if len(images) < 5:
        return None
    first = cv2.imread(str(images[0]), cv2.IMREAD_GRAYSCALE)
    h, w = first.shape
    target, fisheye, scale = classify(images, board)
    model = CameraModel.FISHEYE if fisheye else CameraModel.PINHOLE

    views = []
    n_detected_imgs = 0
    t0 = time.time()
    if target == "dots":
        det = DotMarkerDetector(board, DotMarkerDetectorConfig(work_scale=scale))
        for p in images:
            gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            ms = det.detect(gray)
            if len(ms) < 12:
                continue
            ids = [m.marker_id for m in ms]
            pts = np.array([m.center for m in ms], np.float32)
            views.append(ViewObservation(pts, board.object_points(ids), marker_ids=ids))
            n_detected_imgs += 1
    else:
        cdet = CheckerboardDetector(Checkerboard((12, 12), 40.0))
        for p in images:
            gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            d = cdet.detect(gray)
            if d is None:
                continue
            views.append(ViewObservation(d.image_points, d.object_points))
            n_detected_imgs += 1
    t_detect = time.time() - t0

    if len(views) < 3:
        return {"name": folder.name, "target": target, "model": model.value,
                "images": len(images), "detected": n_detected_imgs, "error": "too few views"}

    t0 = time.time()
    try:
        result = calibrate(views, (w, h), model)
    except Exception as e:
        return {"name": folder.name, "target": target, "model": model.value,
                "images": len(images), "detected": n_detected_imgs, "error": str(e)[:80]}
    t_calib = time.time() - t0

    out_dir = out_root / folder.name.split()[0]
    export_all(result, out_dir, folder.name.split("_")[-1].split()[0],
               result.used_image_points, result.per_point_errors)

    row = {
        "name": folder.name, "target": target, "model": model.value,
        "images": len(images), "detected": n_detected_imgs,
        "views_used": result.n_views, "points": result.n_points,
        "fx": result.fx, "fy": result.fy, "cx": result.cx, "cy": result.cy,
        "rms": result.rms, "t_detect": t_detect, "t_calib": t_calib,
        "size": f"{w}x{h}",
    }
    ref = load_reference_xml(folder)
    if ref:
        row["ref"] = ref
    return row


def main() -> int:
    roots = [Path(a) for a in sys.argv[1:-1]] or [Path(".")]
    if len(sys.argv) >= 2:
        roots = [Path(sys.argv[1])]
    out_root = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path("validation_out")
    board = MarkerBoard.builtin()

    rows = []
    folders = sorted(p for p in roots[0].iterdir() if p.is_dir())
    for folder in folders:
        if not list_images(folder):
            # nested dataset dirs
            for sub in sorted(folder.iterdir()):
                if sub.is_dir() and list_images(sub):
                    r = run_dataset(sub, out_root, board)
                    if r:
                        rows.append(r)
                        print_row(r)
            continue
        r = run_dataset(folder, out_root, board)
        if r:
            rows.append(r)
            print_row(r)

    print("\n=== summary ===")
    for r in rows:
        print_row(r)
    return 0


def print_row(r):
    if "error" in r:
        print(f"{r['name'][:44]:44s} {r['target']:7s} {r['model']:8s} "
              f"imgs={r['images']:3d} det={r['detected']:3d}  ERROR: {r['error']}")
        return
    line = (f"{r['name'][:44]:44s} {r['target']:7s} {r['model']:8s} {r['size']:9s} "
            f"det={r['detected']:3d}/{r['images']:3d} views={r['views_used']:3d} "
            f"fx={r['fx']:8.2f} fy={r['fy']:8.2f} cx={r['cx']:8.2f} cy={r['cy']:8.2f} "
            f"rms={r['rms']:6.3f}  ({r['t_detect']:.0f}s+{r['t_calib']:.0f}s)")
    print(line, flush=True)
    if "ref" in r:
        ref = r["ref"]
        print(f"{'':44s} reference:                    "
              f"fx={ref['fx']:8.2f} fy={ref['fy']:8.2f} cx={ref['cx']:8.2f} "
              f"cy={ref['cy']:8.2f} rms={ref['rms']:6.3f}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
