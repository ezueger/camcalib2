"""Application entry point.

Examples::

    # live GenICam camera (Daheng/Basler), coded dot board, pinhole lens
    camcalib2 --camera --target dots --model pinhole --camera-id FBK25040148

    # simulate from a folder of images (development)
    camcalib2 --images ./shots --target dots --model pinhole

    # fisheye lens with checkerboard from a video
    camcalib2 --video scan.avi --target checker:12x12:40 --model fisheye
"""

from __future__ import annotations

import argparse
import sys

from ..calibration import CameraModel
from ..cli import parse_target
from ..session import CalibrationSession, SessionConfig


def _format_camera_label(camera: dict, index: int) -> str:
    parts = [
        camera.get("vendor"),
        camera.get("model"),
    ]
    serial = camera.get("serial_number")
    if serial:
        parts.append(f"SN {serial}")
    user_name = camera.get("user_defined_name")
    if user_name:
        parts.append(f'"{user_name}"')
    transport = camera.get("tl_type")
    if transport:
        parts.append(transport)
    return f"[{index + 1}] " + " | ".join(part for part in parts if part)


def _print_camera_list(cameras: list[dict]) -> None:
    print("Gefundene GigE/GenICam-Kameras:")
    for index, camera in enumerate(cameras):
        print(" ", _format_camera_label(camera, index))


def _list_cameras(cti_files: list[str] | None) -> list[dict]:
    from ..capture import GenICamSource
    return GenICamSource.list_cameras(cti_files)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--camera", action="store_true", help="GenICam/GigE camera")
    src.add_argument("--images", help="image folder (simulation)")
    src.add_argument("--video", help="video file")
    ap.add_argument("--serial", help="camera serial number")
    ap.add_argument("--camera-index", type=int, default=None,
                    help="zero-based index of the GenICam camera to open")
    ap.add_argument("--cti", action="append", help="GenTL producer .cti path")
    ap.add_argument("--fps", type=float, default=None,
                    help="playback/acquisition frame rate")
    ap.add_argument("--list-cameras", action="store_true",
                    help="list reachable GigE/GenICam cameras and exit")
    ap.add_argument("--target", type=parse_target, default="dots",
                    help="dots | dots:<board.json> | checker:<cols>x<rows>:<square_mm>")
    ap.add_argument("--model", choices=[m.value for m in CameraModel], default="pinhole")
    ap.add_argument("--camera-id", default="camera")
    ap.add_argument("--pixel-size", type=float, default=None,
                    help="sensor pixel size in mm")
    args = ap.parse_args(argv)
    use_camera_source = args.camera or (not args.images and not args.video)
    serial = args.serial
    camera_index = args.camera_index if args.camera_index is not None else 0

    if args.list_cameras:
        try:
            cameras = _list_cameras(args.cti)
        except Exception as e:
            print(f"Kameraerkennung fehlgeschlagen: {e}", file=sys.stderr)
            return 2
        if not cameras:
            print("Keine GigE/GenICam-Kamera gefunden.", file=sys.stderr)
            return 1
        _print_camera_list(cameras)
        return 0

    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("The UI requires PySide6:  pip install camcalib2[ui]", file=sys.stderr)
        return 2
    from .main_window import MainWindow

    def make_source(selected_serial: str | None = None, source_options: dict | None = None):
        source_options = source_options or {}
        if args.images:
            from ..capture import ImageFolderSource
            return ImageFolderSource(args.images, fps=args.fps or 8.0, loop=False)
        if args.video:
            from ..capture import VideoFileSource
            return VideoFileSource(args.video, realtime=True)
        from ..capture import GenICamSource
        return GenICamSource(
            cti_files=args.cti,
            serial=selected_serial or serial,
            index=camera_index,
            reset_to_defaults=bool(source_options.get("reset_to_defaults", False)),
            exposure_us=source_options.get("exposure_us"),
            fps=args.fps,
        )

    def make_session(image_size):
        cfg = SessionConfig(model=CameraModel(args.model))
        return CalibrationSession(args.target, image_size, cfg)

    def list_cameras():
        return _list_cameras(args.cti)

    app = QApplication(sys.argv[:1])
    win = MainWindow(make_source, make_session, camera_id=args.camera_id,
                     pixel_size_mm=args.pixel_size,
                     camera_mode=use_camera_source,
                     list_cameras=list_cameras if use_camera_source else None,
                     format_camera_label=_format_camera_label,
                     initial_camera_serial=serial)
    win.resize(1280, 800)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
