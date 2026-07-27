import time

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QStyleOptionViewItem

from camcalib2.session import FrameFeedback
from camcalib2.session import SessionState
from camcalib2.ui.main_window import _ROLE_CAMERA_SUBTITLE
from camcalib2.ui.main_window import _ROLE_CAMERA_TITLE
from camcalib2.ui.main_window import CaptureWorker
from camcalib2.ui.main_window import MainWindow
from camcalib2.ui.main_window import RuntimeMetrics
from camcalib2.patterns.board import Checkerboard
from camcalib2.patterns.board import MarkerBoard
from camcalib2.ui.app import _default_target_spec_for_model
from camcalib2.ui.app import _format_camera_label
from camcalib2.ui.app import _format_model_label
from camcalib2.ui.app import _format_target_label
from camcalib2.ui.app import _target_options_for_model
from camcalib2.ui.app import _target_spec_from_value


def test_format_camera_label_includes_relevant_fields():
    label = _format_camera_label({
        "vendor": "Basler",
        "model": "acA1920-40gm",
        "serial_number": "12345",
        "user_defined_name": "LineCam",
        "tl_type": "GEV",
        "ip_address": "192.168.0.10",
        "nic_ip_address": "192.168.0.1",
    }, 1)
    assert label == '[2] Basler | acA1920-40gm | SN 12345 | "LineCam" | GEV | 192.168.0.10 via 192.168.0.1'


def test_target_spec_from_value_formats_checkerboard():
    spec = _target_spec_from_value(Checkerboard((12, 12), 40.0))
    assert spec == "checker:12x12:40"


def test_target_options_for_pinhole_prefer_code_marker():
    options = _target_options_for_model("pinhole", MarkerBoard.builtin())
    assert options[0] == ("dots", "CodeMarker | Standard")
    assert dict(options)["checker:10x10:40"] == "Checkerboard | 10x10 | 40 mm"


def test_target_options_for_fisheye_prefer_checkerboard():
    options = _target_options_for_model("fisheye", MarkerBoard.builtin())
    assert options[0] == ("checker:10x10:40", "Checkerboard | 10x10 | 40 mm")
    assert dict(options)["dots"] == "CodeMarker | Standard"


def test_default_target_spec_follows_model():
    assert _default_target_spec_for_model("pinhole") == "dots"
    assert _default_target_spec_for_model("fisheye") == "checker:10x10:40"


def test_format_labels_for_model_and_builtin_board():
    assert _format_model_label("pinhole") == "Perspektivisch"
    assert _format_model_label("fisheye") == "Fisheye"
    assert _format_target_label("dots:vioso_board_196") == "CodeMarker | vioso_board_196"


def test_capture_worker_allows_only_one_preview_in_flight():
    worker = CaptureWorker(source=None, session=None)
    assert worker._begin_preview_delivery() is True
    assert worker._begin_preview_delivery() is False
    worker.on_frame_displayed()
    assert worker._begin_preview_delivery() is True


def test_capture_worker_runtime_metrics_track_processing_and_display():
    worker = CaptureWorker(source=None, session=None)
    worker._reset_metrics()
    time.sleep(0.01)

    metrics = worker._record_processed_frame(0.012)
    worker.on_frame_displayed()
    metrics = worker.runtime_metrics()

    assert metrics.captured_frames == 1
    assert metrics.displayed_frames == 1
    assert metrics.dropped_previews == 0
    assert metrics.last_process_ms == pytest.approx(12.0)
    assert metrics.avg_process_ms == pytest.approx(12.0)
    assert metrics.max_process_ms == pytest.approx(12.0)
    assert metrics.elapsed_s > 0.0


def test_capture_worker_runtime_metrics_count_dropped_previews():
    worker = CaptureWorker(source=None, session=None)
    worker._reset_metrics()

    assert worker._begin_preview_delivery() is True
    worker._record_dropped_preview()
    metrics = worker.runtime_metrics()

    assert metrics.dropped_previews == 1
    assert metrics.displayed_frames == 0


def test_assess_calibration_verdict_and_tilt_warning():
    from types import SimpleNamespace
    from camcalib2.ui.main_window import assess_calibration

    res = SimpleNamespace(rms=0.3, per_point_errors=[np.zeros(50)])
    # good coverage + plenty of tilt -> directly usable
    verdict, _c, details = assess_calibration(
        res, None, coverage=0.9, tilt=0.75, target_coverage=0.85)
    assert verdict.startswith("Sehr gut")

    # too little tilt -> not "sehr gut", warns and gives a tilt tip
    verdict, _c, details = assess_calibration(
        res, None, coverage=0.9, tilt=0.12, target_coverage=0.85)
    assert not verdict.startswith("Sehr gut")
    assert "gekippt" in details.lower() and "kippe" in details.lower()


def test_main_window_on_frame_releases_preview_directly():
    app = QApplication.instance() or QApplication([])

    class _Coverage:
        @staticmethod
        def veil_alpha():
            return np.zeros((2, 2), dtype=float)

    class _Session:
        coverage = _Coverage()

    class _Worker:
        def __init__(self):
            self.calls = 0

        def on_frame_displayed(self):
            self.calls += 1

        def stop(self):
            pass

    win = MainWindow(lambda *_args, **_kwargs: None, lambda *_args, **_kwargs: None)
    win._session = _Session()
    win._worker = _Worker()

    fb = FrameFeedback(
        ids=[],
        points=np.empty((0, 2), dtype=np.float32),
        keyframe=False,
        reason="no_target",
        state=SessionState.WAITING,
        coverage=0.0,
        tilt_coverage=0.0,
        n_keyframes=0,
        rms=None,
        result=None,
        progress=0.0,
    )
    win._on_frame(np.zeros((4, 4), dtype=np.uint8), fb, RuntimeMetrics())

    assert win._worker.calls == 1
    win.close()
    app.processEvents()


def test_camera_dropdown_lines_show_type_and_serial():
    camera = {
        "vendor": "Daheng Imaging",
        "model": "MER2-1220-32U3M",
        "serial_number": "FBK25040148",
        "tl_type": "GEV",
        "ip_address": "10.0.0.9",
    }
    assert MainWindow._camera_title(camera, 1) == "[2] Daheng Imaging MER2-1220-32U3M"
    assert MainWindow._camera_subtitle(camera) == "S/N FBK25040148  ·  GEV  ·  10.0.0.9"


def test_camera_dropdown_subtitle_marks_missing_serial():
    assert MainWindow._camera_subtitle({"model": "acA1920"}) == "S/N ?"


def test_camera_dropdown_items_carry_both_lines():
    app = QApplication.instance() or QApplication([])
    cameras = [
        {"vendor": "Daheng Imaging", "model": "MER2-503-23GM-P",
         "serial_number": "FBF24070661", "tl_type": "GEV"},
        {"vendor": "Daheng Imaging", "model": "MER2-503-23GM-P",
         "serial_number": "FBK25040148", "tl_type": "GEV"},
    ]
    win = MainWindow(lambda *_a, **_k: None, lambda *_a, **_k: None,
                     camera_mode=True, list_cameras=lambda: cameras)
    win.refresh_cameras()

    assert win.cmb_camera.count() == 2
    # identical models - the serial is what tells them apart
    titles = [win.cmb_camera.itemData(i, _ROLE_CAMERA_TITLE) for i in range(2)]
    subtitles = [win.cmb_camera.itemData(i, _ROLE_CAMERA_SUBTITLE) for i in range(2)]
    assert titles[0] == titles[1].replace("[2]", "[1]")
    assert "FBF24070661" in subtitles[0] and "FBK25040148" in subtitles[1]

    # the delegate must be able to render both lines
    delegate = win.cmb_camera.itemDelegate()
    option = QStyleOptionViewItem()
    option.initFrom(win.cmb_camera.view())
    hint = delegate.sizeHint(option, win.cmb_camera.model().index(0, 0))
    assert hint.height() >= option.fontMetrics.height() * 2

    win.close()
    app.processEvents()
