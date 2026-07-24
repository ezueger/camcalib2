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
from ..patterns.board import Checkerboard, MarkerBoard
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
    ip_address = camera.get("ip_address")
    nic_ip_address = camera.get("nic_ip_address")
    if ip_address and nic_ip_address:
        parts.append(f"{ip_address} via {nic_ip_address}")
    elif ip_address:
        parts.append(ip_address)
    return f"[{index + 1}] " + " | ".join(part for part in parts if part)


def _print_camera_list(cameras: list[dict]) -> None:
    print("Gefundene GigE/GenICam-Kameras:")
    for index, camera in enumerate(cameras):
        print(" ", _format_camera_label(camera, index))


def _list_cameras(cti_files: list[str] | None) -> list[dict]:
    from ..capture import GenICamSource
    return GenICamSource.list_cameras(cti_files)


def _argument_was_provided(argv, name: str) -> bool:
    args = list(sys.argv[1:] if argv is None else argv)
    return f"--{name}" in args or any(arg.startswith(f"--{name}=") for arg in args)


def _target_spec_from_value(target) -> str:
    if target == "auto":
        return "dots:auto"
    if isinstance(target, MarkerBoard):
        if target.name == "vioso_board_196":
            return "dots"
        if target.name in MarkerBoard.builtin_names():
            return f"dots:{target.name}"
        return "dots"
    if isinstance(target, Checkerboard):
        cols, rows = target.inner_corners
        return f"checker:{cols}x{rows}:{target.square_size:g}"
    return "dots"


def _default_target_spec_for_model(model_value: str) -> str:
    return "checker:10x10:40" if model_value == CameraModel.FISHEYE.value else "dots"


def _format_model_label(model_value: str) -> str:
    if model_value == CameraModel.FISHEYE.value:
        return "Fisheye"
    return "Perspektivisch"


def _format_target_label(spec: str) -> str:
    if spec == "dots":
        return "CodeMarker | Standard"
    if spec == "dots:auto":
        return "CodeMarker | Auto-Erkennung"
    if spec.startswith("dots:"):
        return f"CodeMarker | {spec.split(':', 1)[1]}"
    if spec.startswith("checker:"):
        parts = spec.split(":")
        grid = parts[1] if len(parts) > 1 else "12x12"
        square = parts[2] if len(parts) > 2 else "40"
        return f"Checkerboard | {grid} | {square} mm"
    return spec


def _target_options_for_model(model_value: str, initial_target=None) -> list[tuple[str, str]]:
    marker_specs = ["dots", "dots:auto"]
    marker_specs.extend(
        f"dots:{name}" for name in MarkerBoard.builtin_names() if name != "vioso_board_196"
    )
    checker_specs = ["checker:10x10:40", "checker:12x12:40"]

    if model_value == CameraModel.FISHEYE.value:
        specs = [*checker_specs, *marker_specs]
    else:
        specs = [*marker_specs, *checker_specs]

    initial_spec = _target_spec_from_value(initial_target) if initial_target is not None else None
    if initial_spec and initial_spec not in specs:
        specs.append(initial_spec)
    return [(spec, _format_target_label(spec)) for spec in specs]


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
    target_was_explicit = _argument_was_provided(argv, "target")
    initial_target_spec = (
        _target_spec_from_value(args.target)
        if target_was_explicit
        else _default_target_spec_for_model(args.model)
    )

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

    def make_source(selected_camera: dict | None = None, source_options: dict | None = None):
        source_options = source_options or {}
        selected_camera = selected_camera or {}
        if args.images:
            from ..capture import ImageFolderSource
            return ImageFolderSource(args.images, fps=args.fps or 8.0, loop=False)
        if args.video:
            from ..capture import VideoFileSource
            return VideoFileSource(args.video, realtime=True)
        from ..capture import GenICamSource
        return GenICamSource(
            cti_files=args.cti,
            serial=selected_camera.get("serial_number") or serial,
            index=int(selected_camera.get("camera_index", camera_index)),
            reset_to_defaults=bool(source_options.get("reset_to_defaults", False)),
            exposure_us=source_options.get("exposure_us"),
            fps=args.fps,
            ip_address=selected_camera.get("ip_address"),
            mac_address=selected_camera.get("mac_address"),
            backend=selected_camera.get("backend"),
        )

    def make_session(image_size):
        cfg = SessionConfig(model=CameraModel(args.model))
        return CalibrationSession(args.target, image_size, cfg)

    def make_session_for_target(image_size,
                                target_spec: str | None = None,
                                model_value: str | None = None):
        resolved_model = CameraModel(model_value or args.model)
        cfg = SessionConfig(model=resolved_model)
        resolved_target = target_spec or _target_spec_from_value(args.target)
        return CalibrationSession(parse_target(resolved_target), image_size, cfg)

    def list_cameras():
        return _list_cameras(args.cti)

    app = QApplication(sys.argv[:1])
    win = MainWindow(make_source, make_session_for_target, camera_id=args.camera_id,
                     pixel_size_mm=args.pixel_size,
                     camera_mode=use_camera_source,
                     list_cameras=list_cameras if use_camera_source else None,
                     format_camera_label=_format_camera_label,
                     initial_camera_serial=serial,
                     model_options=[(m.value, _format_model_label(m.value)) for m in CameraModel],
                     initial_model_value=args.model,
                     target_options_by_model={
                         CameraModel.PINHOLE.value: _target_options_for_model(
                             CameraModel.PINHOLE.value, args.target
                         ),
                         CameraModel.FISHEYE.value: _target_options_for_model(
                             CameraModel.FISHEYE.value, args.target
                         ),
                     },
                     initial_target_spec=initial_target_spec)
    win.resize(1280, 800)  # restore size when the user un-maximizes
    win.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
