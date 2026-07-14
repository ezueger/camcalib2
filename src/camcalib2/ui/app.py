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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--camera", action="store_true", help="GenICam/GigE camera")
    src.add_argument("--images", help="image folder (simulation)")
    src.add_argument("--video", help="video file")
    ap.add_argument("--serial", help="camera serial number")
    ap.add_argument("--cti", action="append", help="GenTL producer .cti path")
    ap.add_argument("--fps", type=float, default=None,
                    help="playback/acquisition frame rate")
    ap.add_argument("--target", type=parse_target, default="dots",
                    help="dots | dots:<board.json> | checker:<cols>x<rows>:<square_mm>")
    ap.add_argument("--model", choices=[m.value for m in CameraModel], default="pinhole")
    ap.add_argument("--camera-id", default="camera")
    ap.add_argument("--pixel-size", type=float, default=None,
                    help="sensor pixel size in mm")
    args = ap.parse_args(argv)

    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("The UI requires PySide6:  pip install camcalib2[ui]", file=sys.stderr)
        return 2
    from .main_window import MainWindow

    def make_source():
        if args.images:
            from ..capture import ImageFolderSource
            return ImageFolderSource(args.images, fps=args.fps or 8.0, loop=False)
        if args.video:
            from ..capture import VideoFileSource
            return VideoFileSource(args.video, realtime=True)
        from ..capture import GenICamSource
        return GenICamSource(cti_files=args.cti, serial=args.serial, fps=args.fps)

    def make_session(image_size):
        cfg = SessionConfig(model=CameraModel(args.model))
        return CalibrationSession(args.target, image_size, cfg)

    app = QApplication(sys.argv[:1])
    win = MainWindow(make_source, make_session, camera_id=args.camera_id,
                     pixel_size_mm=args.pixel_size)
    win.resize(1280, 800)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
