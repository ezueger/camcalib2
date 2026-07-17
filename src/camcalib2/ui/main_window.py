"""Qt main window: live view with FaceTime-style coverage guidance."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import (QBrush, QColor, QFont, QImage, QPainter, QPen,
                           QPixmap)
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QHBoxLayout, QLabel, QMainWindow,
                               QMessageBox, QPushButton, QStackedWidget,
                               QVBoxLayout, QWidget)

from ..calibration import CameraModel
from ..capture.source import FrameSource
from ..session import CalibrationSession, FrameFeedback, SessionConfig, SessionState


class SolveWorker(QObject):
    """Runs the final solve off the GUI thread (a fisheye finish can
    take minutes - running it inline froze the window). Computes the
    calibration + the traffic-light coverage map; writes no files."""

    finished = Signal(object, object, object)  # result, ocam_result|None, map(np.ndarray BGR)
    error = Signal(str)

    def __init__(self, session: CalibrationSession):
        super().__init__()
        self.session = session

    @Slot()
    def run(self):
        try:
            from ..io.export import render_result_image

            result = self.session.finish()
            ocam = None
            views_points = result.used_image_points
            views_errors = result.per_point_errors
            views_reproj = result.used_reprojections
            if result.model.value == "fisheye":
                from ..calibration.ocam import calibrate_ocam
                ocam = calibrate_ocam(self.session.views, self.session.image_size)
                views_points = ocam.used_image_points
                views_errors = ocam.per_point_errors
                views_reproj = ocam.used_reprojections
            img = render_result_image(result, views_points, views_errors,
                                      views_reproj)
            self.finished.emit(result, ocam, img)
        except Exception as e:
            self.error.emit(str(e))


def write_calibration_files(result, ocam_result, xml_path: str,
                            camera_id: str, pixel_size_mm: float | None,
                            result_map) -> dict:
    """Write the already-computed calibration next to the chosen XML path."""
    from pathlib import Path
    import cv2 as _cv2
    from ..io.export import (write_ocam_xml, write_opencv_yaml,
                             write_vendor_xml)

    out = Path(xml_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = {}
    pts = np.vstack(result.used_image_points) if result.used_image_points else None
    errs = (np.concatenate(result.per_point_errors)
            if result.per_point_errors else None)
    if ocam_result is not None:
        opts = (np.vstack(ocam_result.used_image_points)
                if ocam_result.used_image_points else pts)
        write_ocam_xml(ocam_result, out, camera_id, points=opts)
        written["ocam_xml"] = out
    else:
        ps = (pixel_size_mm, pixel_size_mm) if pixel_size_mm else None
        write_vendor_xml(result, out, camera_id, pixel_size_mm=ps,
                         points=pts, errors=errs)
        written["vendor_xml"] = out
    yaml_path = out.with_name(f"{camera_id}-opencv.yaml")
    write_opencv_yaml(result, yaml_path)
    written["opencv_yaml"] = yaml_path
    img_path = out.with_name(f"{camera_id}-result.jpg")
    _cv2.imwrite(str(img_path), result_map, [_cv2.IMWRITE_JPEG_QUALITY, 92])
    written["result_image"] = img_path
    return written


@dataclass(frozen=True)
class RuntimeMetrics:
    elapsed_s: float = 0.0
    captured_frames: int = 0
    displayed_frames: int = 0
    dropped_previews: int = 0
    capture_fps: float = 0.0
    display_fps: float = 0.0
    avg_process_ms: float = 0.0
    max_process_ms: float = 0.0
    last_process_ms: float = 0.0


class CaptureWorker(QObject):
    """Runs the source + session loop in a QThread."""

    frameProcessed = Signal(np.ndarray, object, object)  # gray frame, FrameFeedback, RuntimeMetrics
    finished = Signal()
    error = Signal(str)

    def __init__(self, source: FrameSource, session: CalibrationSession):
        super().__init__()
        self.source = source
        self.session = session
        self._running = False
        self._preview_lock = threading.Lock()
        self._preview_in_flight = False
        self._metrics_lock = threading.Lock()
        self._started_at = 0.0
        self._captured_frames = 0
        self._displayed_frames = 0
        self._dropped_previews = 0
        self._total_process_s = 0.0
        self._max_process_s = 0.0
        self._last_process_s = 0.0

    @Slot()
    def run(self):
        self._running = True
        self._reset_metrics()
        try:
            while self._running:
                ok, frame, ts = self.source.read()
                if not ok:
                    break
                process_started = time.perf_counter()
                fb = self.session.process(frame, ts)
                process_s = time.perf_counter() - process_started
                metrics = self._record_processed_frame(process_s)
                if self._begin_preview_delivery():
                    self.frameProcessed.emit(frame, fb, metrics)
                else:
                    self._record_dropped_preview()
        except Exception as e:  # surface errors instead of dying silently
            if self._running:
                self.error.emit(str(e))
        finally:
            self.finished.emit()

    def stop(self):
        self._running = False
        self._finish_preview_delivery()

    def _begin_preview_delivery(self) -> bool:
        with self._preview_lock:
            if self._preview_in_flight:
                return False
            self._preview_in_flight = True
            return True

    def _finish_preview_delivery(self):
        with self._preview_lock:
            self._preview_in_flight = False

    @Slot()
    def on_frame_displayed(self):
        self._record_displayed_frame()
        self._finish_preview_delivery()

    def _reset_metrics(self):
        with self._metrics_lock:
            self._started_at = time.perf_counter()
            self._captured_frames = 0
            self._displayed_frames = 0
            self._dropped_previews = 0
            self._total_process_s = 0.0
            self._max_process_s = 0.0
            self._last_process_s = 0.0

    def _record_processed_frame(self, process_s: float) -> RuntimeMetrics:
        with self._metrics_lock:
            self._captured_frames += 1
            self._total_process_s += process_s
            self._last_process_s = process_s
            if process_s > self._max_process_s:
                self._max_process_s = process_s
        return self.runtime_metrics()

    def _record_dropped_preview(self):
        with self._metrics_lock:
            self._dropped_previews += 1

    def _record_displayed_frame(self):
        with self._metrics_lock:
            self._displayed_frames += 1

    def runtime_metrics(self) -> RuntimeMetrics:
        with self._metrics_lock:
            elapsed_s = max(time.perf_counter() - self._started_at, 1e-9)
            captured_frames = self._captured_frames
            displayed_frames = self._displayed_frames
            dropped_previews = self._dropped_previews
            total_process_s = self._total_process_s
            max_process_s = self._max_process_s
            last_process_s = self._last_process_s

        avg_process_ms = (total_process_s / captured_frames * 1000.0) if captured_frames else 0.0
        return RuntimeMetrics(
            elapsed_s=elapsed_s,
            captured_frames=captured_frames,
            displayed_frames=displayed_frames,
            dropped_previews=dropped_previews,
            capture_fps=captured_frames / elapsed_s,
            display_fps=displayed_frames / elapsed_s,
            avg_process_ms=avg_process_ms,
            max_process_ms=max_process_s * 1000.0,
            last_process_ms=last_process_s * 1000.0,
        )


class LiveView(QLabel):
    """Video widget with coverage veil, marker dots and progress ring."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #101014;")
        self._frame: np.ndarray | None = None
        self._fb: FrameFeedback | None = None
        self._coverage_mask: np.ndarray | None = None

    def update_frame(self, frame: np.ndarray, fb: FrameFeedback, coverage_mask):
        self._frame = frame
        self._fb = fb
        self._coverage_mask = coverage_mask
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._frame is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        h, w = self._frame.shape[:2]
        img = QImage(self._frame.data, w, h, self._frame.strides[0],
                     QImage.Format_Grayscale8)
        scale = min(self.width() / w, self.height() / h)
        dw, dh = int(w * scale), int(h * scale)
        ox, oy = (self.width() - dw) // 2, (self.height() - dh) // 2
        painter.drawImage(ox, oy, img.scaled(dw, dh, Qt.KeepAspectRatio,
                                             Qt.SmoothTransformation))
        fb = self._fb
        if fb is None:
            painter.end()
            return

        # --- coverage overlay: uncovered cells clearly red, covered
        #     cells a light green tint - full coverage reads as a
        #     uniformly light-green image
        if self._coverage_mask is not None:
            rows, cols = self._coverage_mask.shape
            cw, ch = dw / cols, dh / rows
            painter.setPen(Qt.NoPen)
            red = QBrush(QColor(225, 45, 45, 130))
            green = QBrush(QColor(70, 220, 120, 45))
            for r in range(rows):
                for c in range(cols):
                    painter.setBrush(green if self._coverage_mask[r, c] else red)
                    painter.drawRect(int(ox + c * cw), int(oy + r * ch),
                                     int(cw + 1), int(ch + 1))

        # --- detected points
        color = QColor(80, 255, 120) if fb.keyframe else QColor(120, 220, 255)
        painter.setPen(QPen(color, 2))
        painter.setBrush(Qt.NoBrush)
        for x, y in fb.points:
            painter.drawEllipse(int(ox + x * scale) - 3, int(oy + y * scale) - 3, 6, 6)

        # --- FaceTime-style progress ring (center)
        ring_r = int(min(dw, dh) * 0.16)
        cx, cy = ox + dw // 2, oy + dh // 2
        painter.setPen(QPen(QColor(255, 255, 255, 60), 6))
        painter.drawEllipse(cx - ring_r, cy - ring_r, 2 * ring_r, 2 * ring_r)
        painter.setPen(QPen(QColor(90, 250, 140), 6, Qt.SolidLine, Qt.RoundCap))
        span = int(-360 * 16 * fb.progress)
        painter.drawArc(cx - ring_r, cy - ring_r, 2 * ring_r, 2 * ring_r, 90 * 16, span)

        # --- status text
        painter.setFont(QFont("Sans", 11, QFont.Bold))
        painter.setPen(QColor(240, 240, 240))
        msg = {
            SessionState.WAITING: "Pattern vor die Kamera halten …",
            SessionState.SCANNING: self._hint(fb),
            SessionState.CONVERGED: "Fertig! Kalibrierung konvergiert ✓",
            SessionState.FINISHED: "Abgeschlossen.",
        }[fb.state]
        painter.drawText(ox + 12, oy + dh - 14, msg)
        painter.end()

    @staticmethod
    def _hint(fb: FrameFeedback) -> str:
        if fb.reason == "blurry":
            return "Bewegung zu schnell - langsamer bewegen"
        if fb.coverage < 0.5:
            return "Kamera bewegen: Pattern über den ganzen Sensor führen"
        if fb.tilt_coverage < 0.5:
            return "Pattern aus schrägeren Winkeln aufnehmen"
        return "Weiter abtasten - Ecken und Ränder nicht vergessen"


class ResultView(QWidget):
    """Post-scan review page: traffic-light coverage/error map with the
    calibration summary. Green = good, used observations; red = bad."""

    backToScan = Signal()
    finishExport = Signal()

    def __init__(self):
        super().__init__()
        self._pixmap: QPixmap | None = None
        self.image = QLabel("")
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setMinimumSize(640, 480)
        self.image.setStyleSheet("background-color: #101014;")
        self.summary = QLabel("")
        self.summary.setStyleSheet("font-family: monospace; padding: 6px;")
        self.summary.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.btn_back = QPushButton("Back to Scan")
        self.btn_export = QPushButton("Finish and Create Calibrationfile")
        self.btn_back.clicked.connect(self.backToScan.emit)
        self.btn_export.clicked.connect(self.finishExport.emit)

        buttons = QHBoxLayout()
        buttons.addWidget(self.btn_back)
        buttons.addStretch(1)
        buttons.addWidget(self.btn_export)
        lay = QVBoxLayout()
        lay.addWidget(self.image, 1)
        lay.addWidget(self.summary)
        lay.addLayout(buttons)
        self.setLayout(lay)

    def set_result(self, result, ocam_result, bgr_map: np.ndarray):
        h, w = bgr_map.shape[:2]
        qimg = QImage(bgr_map.data, w, h, bgr_map.strides[0],
                      QImage.Format_BGR888)
        self._pixmap = QPixmap.fromImage(qimg.copy())
        self._rescale()
        errs = (np.concatenate(result.per_point_errors)
                if result.per_point_errors else np.empty(0))
        good = int((errs <= 1.0).sum())
        bad = int((errs > 2.0).sum())
        rms = ocam_result.rms if ocam_result is not None else result.rms
        model = "OCam (Fisheye)" if ocam_result is not None else "Pinhole"
        self.summary.setText(
            f"Modell: {model}   Views: {result.n_views}   "
            f"Punkte: {len(errs)}  (gut <=1px: {good}, schlecht >2px: {bad})\n"
            f"fx={result.fx:.2f}  fy={result.fy:.2f}  "
            f"cx={result.cx:.2f}  cy={result.cy:.2f}   RMS={rms:.3f} px\n"
            f"Grün = gute, verwendete Punkte - Rot = schlechte Punkte. "
            f"Dünn besetzte/rote Zonen? -> Back to Scan und dort nachscannen.")

    def _rescale(self):
        if self._pixmap is not None:
            self.image.setPixmap(self._pixmap.scaled(
                self.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()


class MainWindow(QMainWindow):
    def __init__(self, make_source, make_session, camera_id: str = "camera",
                 pixel_size_mm: float | None = None,
                 camera_mode: bool = False,
                 list_cameras=None,
                 format_camera_label=None,
                 initial_camera_serial: str | None = None,
                 model_options: list[tuple[str, str]] | None = None,
                 initial_model_value: str | None = None,
                 target_options_by_model: dict[str, list[tuple[str, str]]] | None = None,
                 initial_target_spec: str | None = None):
        """``make_source``/``make_session`` are factories so a session can
        be restarted without rebuilding the window."""
        super().__init__()
        self.setWindowTitle("camcalib2 - Kamera-Kalibrierung")
        self._make_source = make_source
        self._make_session = make_session
        self._camera_id = camera_id
        self._pixel_size = pixel_size_mm
        self._camera_mode = camera_mode
        self._list_cameras = list_cameras
        self._format_camera_label = format_camera_label or self._default_camera_label
        self._initial_camera_serial = initial_camera_serial
        self._model_options = model_options or [(CameraModel.PINHOLE.value, "Perspektivisch")]
        self._initial_model_value = initial_model_value or self._model_options[0][0]
        self._target_options_by_model = target_options_by_model or {
            self._initial_model_value: [("dots", "CodeMarker | Standard")]
        }
        self._initial_target_spec = initial_target_spec or (
            self._target_options_by_model[self._initial_model_value][0][0]
        )
        self._cameras: list[dict] = []
        self._thread: QThread | None = None
        self._worker: CaptureWorker | None = None
        self._session: CalibrationSession | None = None
        self._source: FrameSource | None = None
        self._capture_running = False
        self._last_stats_update = 0.0
        self._solve_thread: QThread | None = None
        self._solve_worker: SolveWorker | None = None
        self._result = None
        self._ocam_result = None
        self._result_map = None

        self.view = LiveView()
        self.result_view = ResultView()
        self.result_view.backToScan.connect(self.back_to_scan)
        self.result_view.finishExport.connect(self.export_calibration)
        self.stats = QLabel("-")
        self.stats.setStyleSheet("font-family: monospace; padding: 6px;")
        self.stats.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.stats.setWordWrap(True)
        self.stats.setMinimumHeight(320)
        self.lbl_model = QLabel("Linsenart")
        self.cmb_model = QComboBox()
        self.lbl_target = QLabel("Testpattern")
        self.cmb_target = QComboBox()
        self.lbl_camera = QLabel("Kamera")
        self.cmb_camera = QComboBox()
        self.lbl_camera_status = QLabel("")
        self.lbl_camera_status.setWordWrap(True)
        self.btn_refresh_cameras = QPushButton("Kameras aktualisieren")
        self.chk_reset_defaults = QCheckBox("Auf Kamera-Defaults zuruecksetzen")
        self.chk_reset_defaults.setChecked(True)
        self.chk_custom_exposure = QCheckBox("Belichtung manuell setzen")
        self.chk_custom_exposure.setChecked(True)
        self.spn_exposure_ms = QDoubleSpinBox()
        self.spn_exposure_ms.setDecimals(1)
        self.spn_exposure_ms.setRange(0.1, 10_000.0)
        self.spn_exposure_ms.setSingleStep(0.1)
        self.spn_exposure_ms.setValue(10.0)
        self.spn_exposure_ms.setSuffix(" ms")
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop && Kamera freigeben")
        self.btn_stop.setEnabled(False)
        self.btn_finish = QPushButton("Ergebnis berechnen")
        self.btn_finish.setEnabled(False)
        self.cmb_model.currentIndexChanged.connect(self._on_model_changed)
        self.cmb_target.currentIndexChanged.connect(self._on_target_changed)
        self.cmb_camera.currentIndexChanged.connect(self._on_camera_changed)
        self.btn_refresh_cameras.clicked.connect(self.refresh_cameras)
        self.chk_custom_exposure.toggled.connect(self.spn_exposure_ms.setEnabled)
        self.chk_custom_exposure.toggled.connect(lambda _checked: self._update_camera_status())
        self.chk_reset_defaults.toggled.connect(lambda _checked: self._update_camera_status())
        self.spn_exposure_ms.valueChanged.connect(lambda _value: self._update_camera_status())
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_finish.clicked.connect(self.finish)
        self.spn_exposure_ms.setEnabled(self.chk_custom_exposure.isChecked())

        side = QVBoxLayout()
        side.addWidget(self.lbl_model)
        side.addWidget(self.cmb_model)
        side.addWidget(self.lbl_target)
        side.addWidget(self.cmb_target)
        if self._camera_mode:
            side.addWidget(self.lbl_camera)
            side.addWidget(self.cmb_camera)
            side.addWidget(self.btn_refresh_cameras)
            side.addWidget(self.chk_reset_defaults)
            side.addWidget(self.chk_custom_exposure)
            side.addWidget(self.spn_exposure_ms)
            side.addWidget(self.lbl_camera_status)
        side.addWidget(self.stats)
        side.addStretch(1)
        side.addWidget(self.btn_start)
        side.addWidget(self.btn_stop)
        side.addWidget(self.btn_finish)
        sidew = QWidget()
        sidew.setLayout(side)
        sidew.setFixedWidth(280)

        self._stack = QStackedWidget()
        self._stack.addWidget(self.view)         # 0: live scan
        self._stack.addWidget(self.result_view)  # 1: result review
        lay = QHBoxLayout()
        lay.addWidget(self._stack, 1)
        lay.addWidget(sidew)
        central = QWidget()
        central.setLayout(lay)
        self.setCentralWidget(central)
        self._populate_model_options()
        self._populate_target_options()
        if self._camera_mode:
            self.refresh_cameras()

    # ------------------------------------------------------------------
    @Slot()
    def start(self):
        self.stop()
        self._stack.setCurrentIndex(0)
        self._result = self._ocam_result = self._result_map = None
        selected_camera = self._selected_camera()
        if self._camera_mode and self.cmb_camera.count() == 0:
            QMessageBox.warning(self, "Keine Kamera",
                                "Es ist keine GigE/GenICam-Kamera ausgewaehlt.")
            return
        source_options = self._source_options()
        try:
            self._source = self._make_source(selected_camera, source_options)
            self._source.open()
            self._session = self._make_session(
                self._source.image_size,
                self._selected_target_spec(),
                self._selected_model_value(),
            )
        except Exception as e:
            QMessageBox.critical(self, "Fehler", str(e))
            return
        self._worker = CaptureWorker(self._source, self._session)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.frameProcessed.connect(self._on_frame)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._on_capture_finished)
        self._thread.start()
        self._capture_running = True
        self._last_stats_update = 0.0
        self.btn_start.setText("Neu starten")
        self.btn_stop.setEnabled(True)
        self.btn_finish.setEnabled(True)
        self._set_camera_controls_enabled(False)
        self.cmb_model.setEnabled(False)
        self.cmb_target.setEnabled(False)

    def stop(self):
        was_running = self._capture_running
        worker = self._worker
        if self._worker:
            self._worker.stop()
        self._release_source()
        if self._thread:
            self._thread.quit()
            self._thread.wait(3000)
        self._worker = self._thread = None
        self._capture_running = False
        self.btn_start.setText("Start")
        self.btn_stop.setEnabled(False)
        self._set_camera_controls_enabled(True)
        self.cmb_model.setEnabled(True)
        self.cmb_target.setEnabled(True)
        if was_running and self._camera_mode and self.cmb_camera.count() > 0:
            self._update_camera_status()
            self.lbl_camera_status.setText(
                "Kamera freigegeben. Du kannst sie jetzt nachstellen.\n"
                + self.lbl_camera_status.text()
            )

    def _release_source(self):
        if self._source:
            try:
                self._source.close()
            except Exception:
                pass
            finally:
                self._source = None

    @Slot(np.ndarray, object, object)
    def _on_frame(self, frame, fb: FrameFeedback, metrics: RuntimeMetrics):
        try:
            mask = self._session.coverage.mask() if self._session else None
            self.view.update_frame(frame, fb, mask)
            now = time.perf_counter()
            if now - self._last_stats_update >= 0.25:
                lines = [
                    f"Status {fb.state.value} | Marker {len(fb.ids)} | KF {fb.n_keyframes}",
                    f"Abd {fb.coverage*100:.0f}% | Winkel {fb.tilt_coverage*100:.0f}%",
                    f"Laufz {metrics.elapsed_s:6.1f}s | Cap {metrics.capture_fps:4.1f} fps | UI {metrics.display_fps:4.1f} fps",
                    f"Frames {metrics.captured_frames}/{metrics.displayed_frames} | Drops {metrics.dropped_previews}",
                    f"Proc {metrics.last_process_ms:6.1f} ms | Avg {metrics.avg_process_ms:5.1f} | Peak {metrics.max_process_ms:6.1f}",
                ]
                if fb.result is not None:
                    r = fb.result
                    lines += [
                        "",
                        f"fx {r.fx:8.2f} | fy {r.fy:8.2f} | RMS {r.rms:6.3f} px",
                        f"cx {r.cx:8.2f} | cy {r.cy:8.2f} | Views {r.n_views}",
                    ]
                self.stats.setText("\n".join(lines))
                self._last_stats_update = now
        finally:
            # Call directly from the UI thread: queued delivery back into the
            # worker thread would stall because the capture loop keeps that
            # thread busy and the release signal would never be processed.
            worker = self._worker
            if worker is not None:
                worker.on_frame_displayed()

    @Slot(str)
    def _on_error(self, msg):
        self.stop()
        QMessageBox.critical(self, "Fehler", msg)

    @Slot()
    def _on_capture_finished(self):
        self._capture_running = False
        self._release_source()
        self.btn_start.setText("Start")
        self.btn_stop.setEnabled(False)
        self._set_camera_controls_enabled(True)
        self.cmb_model.setEnabled(True)
        self.cmb_target.setEnabled(True)

    def _populate_model_options(self):
        self.cmb_model.blockSignals(True)
        self.cmb_model.clear()
        selected_index = 0
        for index, (model_value, label) in enumerate(self._model_options):
            self.cmb_model.addItem(label, model_value)
            if model_value == self._initial_model_value:
                selected_index = index
        self.cmb_model.setCurrentIndex(selected_index)
        self.cmb_model.blockSignals(False)

    def _populate_target_options(self, preferred_spec: str | None = None):
        model_value = self._selected_model_value()
        target_options = self._target_options_by_model.get(model_value, [])
        self.cmb_target.blockSignals(True)
        self.cmb_target.clear()
        selected_index = 0
        for index, (spec, label) in enumerate(target_options):
            self.cmb_target.addItem(label, spec)
            if spec == (preferred_spec or self._initial_target_spec):
                selected_index = index
        self.cmb_target.setCurrentIndex(selected_index)
        self.cmb_target.blockSignals(False)

    @Slot()
    def refresh_cameras(self):
        if not self._camera_mode or self._list_cameras is None:
            return
        current_camera = self._selected_camera()
        wanted_key = self._camera_selection_key(current_camera)
        if wanted_key is None and self._initial_camera_serial:
            wanted_key = f"serial_number:{self._initial_camera_serial}"
        try:
            cameras = self._list_cameras()
        except Exception as e:
            self._cameras = []
            self.cmb_camera.clear()
            self.lbl_camera_status.setText(str(e))
            self.btn_start.setEnabled(False)
            return

        self.cmb_camera.blockSignals(True)
        self.cmb_camera.clear()
        self._cameras = []
        for index, camera in enumerate(cameras):
            camera_entry = dict(camera)
            camera_entry["camera_index"] = index
            self.cmb_camera.addItem(self._format_camera_label(camera_entry, index))
            self._cameras.append(camera_entry)
        self.cmb_camera.blockSignals(False)

        if not cameras:
            self.lbl_camera_status.setText("Keine GigE/GenICam-Kamera gefunden.")
            self.btn_start.setEnabled(False)
            return

        selected_index = 0
        if wanted_key:
            for index, camera in enumerate(self._cameras):
                if self._camera_selection_key(camera) == wanted_key:
                    selected_index = index
                    break
        self.cmb_camera.setCurrentIndex(selected_index)
        self._initial_camera_serial = self._cameras[selected_index].get("serial_number") or None
        self.btn_start.setEnabled(True)
        self._update_camera_status()

    @Slot(int)
    def _on_camera_changed(self, _index: int):
        self._update_camera_status()

    @Slot(int)
    def _on_model_changed(self, _index: int):
        preferred_spec = self._recommended_target_spec_for_model(self._selected_model_value())
        self._populate_target_options(preferred_spec=preferred_spec)
        self._update_camera_status()

    @Slot(int)
    def _on_target_changed(self, _index: int):
        self._update_camera_status()

    def _selected_camera(self) -> dict | None:
        index = self.cmb_camera.currentIndex()
        if 0 <= index < len(self._cameras):
            return self._cameras[index]
        return None

    def _selected_target_spec(self) -> str:
        data = self.cmb_target.currentData()
        if isinstance(data, str) and data:
            return data
        return self._initial_target_spec

    def _selected_model_value(self) -> str:
        data = self.cmb_model.currentData()
        if isinstance(data, str) and data:
            return data
        return self._initial_model_value

    @staticmethod
    def _recommended_target_spec_for_model(model_value: str) -> str:
        if model_value == CameraModel.FISHEYE.value:
            return "checker:10x10:40"
        return "dots"

    @staticmethod
    def _camera_selection_key(camera: dict | None) -> str | None:
        if not camera:
            return None
        for key in ("serial_number", "ip_address", "mac_address", "display_name"):
            value = camera.get(key)
            if value:
                return f"{key}:{value}"
        return None

    def _update_camera_status(self):
        if not self._camera_mode:
            return
        if self.cmb_camera.count() == 0:
            self.lbl_camera_status.setText("Keine Kamera verfuegbar.")
            return
        model_label = self.cmb_model.currentText()
        label = self.cmb_camera.currentText()
        target_label = self.cmb_target.currentText()
        exposure = "Kamera-Default"
        if self.chk_custom_exposure.isChecked():
            exposure = f"{self.spn_exposure_ms.value():.1f} ms"
        reset_info = "mit Reset auf Defaults" if self.chk_reset_defaults.isChecked() else "ohne Reset"
        self.lbl_camera_status.setText(
            f"Linsenart: {model_label}\nPattern: {target_label}\nAusgewaehlt: {label}\n"
            f"Start: {reset_info}, Belichtung {exposure}"
        )

    def _set_camera_controls_enabled(self, enabled: bool):
        if not self._camera_mode:
            return
        self.cmb_camera.setEnabled(enabled)
        self.btn_refresh_cameras.setEnabled(enabled)
        self.chk_reset_defaults.setEnabled(enabled)
        self.chk_custom_exposure.setEnabled(enabled)
        self.spn_exposure_ms.setEnabled(enabled and self.chk_custom_exposure.isChecked())

    def _source_options(self) -> dict:
        if not self._camera_mode:
            return {}
        exposure_us = None
        if self.chk_custom_exposure.isChecked():
            exposure_us = float(self.spn_exposure_ms.value()) * 1000.0
        return {
            "reset_to_defaults": self.chk_reset_defaults.isChecked(),
            "exposure_us": exposure_us,
        }

    @staticmethod
    def _default_camera_label(camera: dict, index: int) -> str:
        return camera.get("serial_number") or f"Kamera {index + 1}"

    # ------------------------------------------------------------------
    # result step: scan -> compute -> review page -> back to scan / export
    @Slot()
    def finish(self):
        if not self._session or len(self._session.views) < 3:
            QMessageBox.warning(self, "Zu wenig Daten",
                                "Es wurden noch nicht genug Keyframes gesammelt.")
            return
        self.stop()

        # final solve in a worker thread - the fisheye finish can take a
        # while and used to freeze the GUI here
        self.btn_finish.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.stats.setText("Ergebnis wird berechnet …\n(kann bei Fisheye"
                           " einige Minuten dauern)")
        self._solve_worker = SolveWorker(self._session)
        self._solve_thread = QThread()
        self._solve_worker.moveToThread(self._solve_thread)
        self._solve_thread.started.connect(self._solve_worker.run)
        self._solve_worker.finished.connect(self._on_solve_done)
        self._solve_worker.error.connect(self._on_solve_error)
        self._solve_worker.finished.connect(self._solve_thread.quit)
        self._solve_worker.error.connect(self._solve_thread.quit)
        self._solve_thread.start()

    @Slot(object, object, object)
    def _on_solve_done(self, result, ocam_result, bgr_map):
        self.btn_finish.setEnabled(True)
        self.btn_start.setEnabled(True)
        self._result = result
        self._ocam_result = ocam_result
        self._result_map = bgr_map
        self.result_view.set_result(result, ocam_result, bgr_map)
        self._stack.setCurrentIndex(1)
        self.stats.setText(
            "Ergebnis prüfen:\nGrün = gute Punkte, Rot = schlechte.\n\n"
            "Back to Scan  -> weiter scannen und verbessern\n"
            "Finish and Create Calibrationfile -> abschließen")

    @Slot(str)
    def _on_solve_error(self, msg):
        self.btn_finish.setEnabled(True)
        self.btn_start.setEnabled(True)
        QMessageBox.critical(self, "Kalibrierung fehlgeschlagen", msg)

    @Slot()
    def back_to_scan(self):
        """Resume scanning with the existing session - keyframes,
        coverage and the current model are kept and improved."""
        self._stack.setCurrentIndex(0)
        if not self._session:
            return
        self._session.resume()
        selected_camera = self._selected_camera()
        try:
            self._source = self._make_source(selected_camera, self._source_options())
            self._source.open()
        except Exception as e:
            QMessageBox.critical(self, "Fehler", str(e))
            return
        self._worker = CaptureWorker(self._source, self._session)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.frameProcessed.connect(self._on_frame)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._on_capture_finished)
        self._thread.start()
        self._capture_running = True
        self._last_stats_update = 0.0
        self.btn_stop.setEnabled(True)
        self.btn_finish.setEnabled(True)
        self._set_camera_controls_enabled(False)
        self.cmb_model.setEnabled(False)
        self.cmb_target.setEnabled(False)

    @Slot()
    def export_calibration(self):
        if self._result is None:
            return
        from pathlib import Path
        suffix = "ocam" if self._ocam_result is not None else "ocv"
        default = str(Path.home() / "Documents" / f"{self._camera_id}-{suffix}.xml")
        xml_path, _ = QFileDialog.getSaveFileName(
            self, "Kalibrierung exportieren", default, "XML-Datei (*.xml)")
        if not xml_path:
            return
        try:
            written = write_calibration_files(
                self._result, self._ocam_result, xml_path,
                self._camera_id, self._pixel_size, self._result_map)
        except Exception as e:
            QMessageBox.critical(self, "Export fehlgeschlagen", str(e))
            return
        r = self._result
        rms = self._ocam_result.rms if self._ocam_result is not None else r.rms
        QMessageBox.information(
            self, "Export abgeschlossen",
            f"fx={r.fx:.2f} fy={r.fy:.2f}\n"
            f"cx={r.cx:.2f} cy={r.cy:.2f}\n"
            f"RMS={rms:.3f} px ({r.n_views} Views)\n\n"
            + "\n".join(str(p) for p in written.values()))

    def closeEvent(self, event):
        self.stop()
        super().closeEvent(event)
