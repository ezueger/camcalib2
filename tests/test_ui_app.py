import time

import pytest

from camcalib2.ui.main_window import CaptureWorker
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
