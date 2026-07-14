"""Batch CLI: calibrate from a folder of images or a video file.

Examples::

    camcalib2-calibrate --images ./shots --target dots --model pinhole \
        --camera-id FBK25040148 --pixel-size 0.0024 --out ./Results

    camcalib2-calibrate --video session.avi --target checker:12x12:40 \
        --model fisheye --out ./Results
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .calibration import CameraModel
from .capture import ImageFolderSource, VideoFileSource
from .io import export_all
from .patterns.board import Checkerboard, MarkerBoard
from .session import CalibrationSession, SessionConfig
from .session.keyframes import KeyframePolicy


def parse_target(spec: str):
    if spec in ("dots", "markers"):
        return MarkerBoard.builtin()
    if spec.startswith("dots:"):
        return MarkerBoard.from_json(spec.split(":", 1)[1])
    if spec.startswith("checker"):
        # checker:<cols>x<rows>:<square_mm>
        parts = spec.split(":")
        cols, rows = (int(x) for x in parts[1].split("x")) if len(parts) > 1 else (12, 12)
        square = float(parts[2]) if len(parts) > 2 else 40.0
        return Checkerboard((cols, rows), square)
    raise argparse.ArgumentTypeError(f"unknown target spec: {spec}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", help="folder with calibration images")
    src.add_argument("--video", help="video file of the scan")
    ap.add_argument("--pattern", default="*.jpg", help="image glob (with --images)")
    ap.add_argument("--target", type=parse_target, default="dots",
                    help="dots | dots:<board.json> | checker:<cols>x<rows>:<square_mm>")
    ap.add_argument("--model", choices=[m.value for m in CameraModel], default="pinhole")
    ap.add_argument("--camera-id", default="camera")
    ap.add_argument("--pixel-size", type=float, default=None,
                    help="sensor pixel size in mm (for the vendor XML <f> entry)")
    ap.add_argument("--out", default="Results")
    ap.add_argument("--all-frames", action="store_true",
                    help="use every frame as a keyframe (classic still-image mode)")
    args = ap.parse_args(argv)

    source = (ImageFolderSource(args.images, args.pattern) if args.images
              else VideoFileSource(args.video))

    with source:
        w, h = source.image_size
        cfg = SessionConfig(model=CameraModel(args.model))
        if args.all_frames:
            cfg.keyframe_policy = KeyframePolicy(
                min_points=6, min_sharpness=0.0, min_motion_px=0.0,
                min_new_cells=0, min_interval=0.0)
        session = CalibrationSession(args.target, (w, h), cfg)

        n = 0
        while True:
            ok, frame, ts = source.read()
            if not ok:
                break
            fb = session.process(frame, ts)
            n += 1
            status = "KEY" if fb.keyframe else "   "
            print(f"frame {n:4d} {status} points={len(fb.ids):3d} "
                  f"coverage={fb.coverage*100:5.1f}% keyframes={fb.n_keyframes}"
                  + (f" rms={fb.rms:.3f}" if fb.rms else ""))

    if session.views is None or len(session.views) < 3:
        print("not enough usable views", file=sys.stderr)
        return 1

    result = session.finish()
    print(f"\nmodel={result.model.value} views={result.n_views} points={result.n_points}")
    print(f"fx={result.fx:.2f} fy={result.fy:.2f} cx={result.cx:.2f} cy={result.cy:.2f}")
    print(f"dist={np.round(result.dist_coeffs, 6)}")
    print(f"rms={result.rms:.4f} px")

    written = export_all(result, args.out, args.camera_id,
                         result.used_image_points, result.per_point_errors,
                         pixel_size_mm=(args.pixel_size, args.pixel_size) if args.pixel_size else None)
    for k, p in written.items():
        print(f"wrote {k}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
