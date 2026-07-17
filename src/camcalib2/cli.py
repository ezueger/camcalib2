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
    if spec in ("dots:auto", "auto"):
        return "auto"
    if spec.startswith("dots:"):
        ref = spec.split(":", 1)[1]
        if ref in MarkerBoard.builtin_names():
            return MarkerBoard.builtin(ref)
        return MarkerBoard.from_json(ref)
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
                    help="dots | dots:auto | dots:<builtin-name|board.json> | "
                         "checker:<cols>x<rows>:<square_mm>  "
                         f"(builtin: {', '.join(MarkerBoard.builtin_names())})")
    ap.add_argument("--model", choices=[m.value for m in CameraModel], default="pinhole")
    ap.add_argument("--camera-id", default="camera")
    ap.add_argument("--pixel-size", type=float, default=None,
                    help="sensor pixel size in mm (for the vendor XML <f> entry)")
    ap.add_argument("--out", default="Results")
    ap.add_argument("--all-frames", action="store_true",
                    help="use every frame as a keyframe (classic still-image mode)")
    ap.add_argument("--no-recover", action="store_true",
                    help="skip the model-guided recovery pass")
    args = ap.parse_args(argv)

    source = (ImageFolderSource(args.images, args.pattern) if args.images
              else VideoFileSource(args.video))

    with source:
        w, h = source.image_size
        if args.target == "auto":
            import cv2 as _cv2
            from .detection import identify_board
            ok, frame, _ = source.read()
            if not ok:
                print("no frames available", file=sys.stderr)
                return 1
            board, n = identify_board(frame)
            if board is None:
                print("could not identify a known marker board", file=sys.stderr)
                return 1
            print(f"identified board: {board.name} ({n} markers)")
            args.target = board
            source.close()
            source.open()
        cfg = SessionConfig(model=CameraModel(args.model))
        if args.all_frames:
            cfg.keyframe_policy = KeyframePolicy(
                min_points=6, min_sharpness=0.0, min_motion_px=0.0,
                min_new_cells=0, min_interval=0.0)
        session = CalibrationSession(args.target, (w, h), cfg)

        n = 0
        frames_of_keyframes = []
        while True:
            ok, frame, ts = source.read()
            if not ok:
                break
            fb = session.process(frame, ts)
            n += 1
            if fb.keyframe and not args.no_recover:
                frames_of_keyframes.append(frame)
            status = "KEY" if fb.keyframe else "   "
            print(f"frame {n:4d} {status} points={len(fb.ids):3d} "
                  f"coverage={fb.coverage*100:5.1f}% keyframes={fb.n_keyframes}"
                  + (f" rms={fb.rms:.3f}" if fb.rms else ""))

    if session.views is None or len(session.views) < 3:
        print("not enough usable views", file=sys.stderr)
        return 1

    result = session.finish()

    recovered_views = None
    if not args.no_recover:
        # model-guided recovery pass over all keyframes.
        # Recovers observations at the lens periphery that the initial
        # detection missed (weak ring dots, partial checkerboard rows).
        from .detection.recovery import (make_projector, recover_dot_markers,
                                         recover_checkerboard_corners)
        recovered = 0
        new_views = []
        for frame, view in zip(frames_of_keyframes, session.views):
            projector = make_projector(result, view)
            if projector is None:
                new_views.append(view)
                continue
            if isinstance(args.target, MarkerBoard):
                view2, k = recover_dot_markers(frame, args.target, view, projector)
            else:
                view2, k = recover_checkerboard_corners(
                    frame, view, projector, args.target.square_size)
            recovered += k
            new_views.append(view2)
        if recovered:
            print(f"recovery pass: +{recovered} observations")
            if CameraModel(args.model) is CameraModel.PINHOLE:
                # pinhole solving is robust - re-solve on the enriched views
                from .calibration import calibrate
                session.views[:] = new_views
                result = calibrate(new_views, (w, h), CameraModel.PINHOLE)
                print(f"re-solved: rms={result.rms:.4f} points={result.n_points}")
            else:
                # fisheye: OpenCV's KB init is fragile at the rim; keep the
                # KB solve on the conservative views and feed the enriched
                # views only to our own OCam bundle adjustment below
                recovered_views = new_views
    print(f"\nmodel={result.model.value} views={result.n_views} points={result.n_points}")
    print(f"fx={result.fx:.2f} fy={result.fy:.2f} cx={result.cx:.2f} cy={result.cy:.2f}")
    print(f"dist={np.round(result.dist_coeffs, 6)}")
    print(f"rms={result.rms:.4f} px")

    ocam_result = None
    views_points = result.used_image_points
    views_errors = result.per_point_errors
    views_reproj = result.used_reprojections
    if result.model is CameraModel.FISHEYE:
        from .calibration.ocam import calibrate_ocam
        ocam_result = calibrate_ocam(session.views, (w, h),
                                     refine_views=recovered_views)
        m = ocam_result.model
        print(f"ocam: cx={m.cx:.2f} cy={m.cy:.2f} a0={m.poly[0]:.3f} "
              f"rms={ocam_result.rms:.4f} points={ocam_result.n_points}")
        # result image / rad from the (possibly rim-enriched) OCam fit
        views_points = ocam_result.used_image_points
        views_errors = ocam_result.per_point_errors
        views_reproj = ocam_result.used_reprojections

    written = export_all(result, args.out, args.camera_id,
                         views_points, views_errors,
                         pixel_size_mm=(args.pixel_size, args.pixel_size) if args.pixel_size else None,
                         ocam_result=ocam_result, views_reproj=views_reproj)
    for k, p in written.items():
        print(f"wrote {k}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
