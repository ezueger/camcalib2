"""Qt main window: live view with FaceTime-style coverage guidance."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
import numpy as np
from PySide6.QtCore import (QObject, QPoint, QPointF, QRect, QRectF, Qt, QThread,
                            Signal, Slot)
from PySide6.QtGui import (QBrush, QColor, QFont, QImage, QPainter,
                           QPainterPath, QPen, QPixmap, QPolygonF)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                               QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel,
                               QMainWindow, QMessageBox, QPlainTextEdit,
                               QPushButton, QStackedWidget, QTabWidget,
                               QVBoxLayout, QWidget)

from ..calibration import CameraModel
from ..capture.source import FrameSource
from ..session import (CalibrationSession, FrameFeedback, REASON_TEXT, Roi,
                       SessionConfig, SessionState)


class SolveWorker(QObject):
    """Runs the final solve off the GUI thread (a fisheye finish can
    take minutes - running it inline froze the window). Computes the
    calibration + the traffic-light coverage map; writes no files.

    Reports phase-weighted progress (0..100 %) via ``progress`` and can be
    aborted via ``cancel()`` (a threading.Event checked inside the solve)."""

    finished = Signal(object, object, object)  # result, ocam_result|None, map(np.ndarray BGR)
    error = Signal(str)
    progress = Signal(float, str)  # percent 0..100, phase message
    cancelled = Signal()

    def __init__(self, session: CalibrationSession):
        super().__init__()
        self.session = session
        self.cancel_event = threading.Event()

    def cancel(self):
        self.cancel_event.set()

    def _band(self, lo, hi):
        """A progress_cb mapping a phase's local 0..1 into a global slice."""
        def cb(frac, msg):
            frac = max(0.0, min(1.0, frac))
            self.progress.emit((lo + (hi - lo) * frac) * 100.0, msg)
        return cb

    @Slot()
    def run(self):
        from ..calibration.solver import SolveCancelled
        try:
            from ..io.export import render_result_image

            fisheye = self.session.cfg.model is CameraModel.FISHEYE
            # top-level split: KB solve | OCam refine | result render
            kb_hi = 0.30 if fisheye else 0.90
            result = self.session.finish(
                progress_cb=self._band(0.0, kb_hi),
                cancel_event=self.cancel_event)
            ocam = None
            views_points = result.used_image_points
            views_errors = result.per_point_errors
            views_reproj = result.used_reprojections
            if fisheye:
                from ..calibration.ocam import calibrate_ocam
                # reuse the KB solve just computed (same views/image_size)
                ocam = calibrate_ocam(
                    self.session.views, self.session.image_size, kb=result,
                    progress_cb=self._band(kb_hi, 0.95),
                    cancel_event=self.cancel_event)
                views_points = ocam.used_image_points
                views_errors = ocam.per_point_errors
                views_reproj = ocam.used_reprojections
            self.progress.emit(96.0, "Ergebnisbild …")
            img = render_result_image(result, views_points, views_errors,
                                      views_reproj)
            self.progress.emit(100.0, "Fertig")
            self.finished.emit(result, ocam, img)
        except SolveCancelled:
            self.cancelled.emit()
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
    #: every frame at camera fps: frame, StillTick, RuntimeMetrics
    framePreview = Signal(np.ndarray, object, object)

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
        """Still-capture loop: every frame is displayed at camera fps
        with only a cheap motion tick; the expensive CV evaluation runs
        in a side thread exactly once per hold-still phase."""
        self._running = True
        self._reset_metrics()
        self._eval_thread: threading.Thread | None = None
        self._eval_result: tuple | None = None
        try:
            monotonic0 = time.perf_counter()
            while self._running:
                ok, frame, ts = self.source.read()
                if not ok:
                    break
                if ts is None or ts <= 0:
                    ts = time.perf_counter() - monotonic0
                process_started = time.perf_counter()

                if self.session is None:
                    # preview-only: show the live stream, no measurement
                    metrics = self._record_processed_frame(
                        time.perf_counter() - process_started)
                    if self._begin_preview_delivery():
                        self.framePreview.emit(frame, None, metrics)
                    else:
                        self._record_dropped_preview()
                    continue

                # deliver a completed evaluation (coverage/points update)
                if self._eval_result is not None:
                    eframe, efb = self._eval_result
                    self._eval_result = None
                    metrics = self._record_processed_frame(
                        time.perf_counter() - process_started)
                    self._begin_preview_delivery()  # released by _on_frame
                    self.frameProcessed.emit(eframe, efb, metrics)

                tick = self.session.tick(frame, ts)
                if (tick.evaluate and
                        (self._eval_thread is None or not self._eval_thread.is_alive())):
                    def _evaluate(f=frame, t=ts):
                        try:
                            fb = self.session.evaluate(f, t)
                            self._eval_result = (f, fb)
                        except Exception as e:
                            self._eval_result = None
                            if self._running:
                                self.error.emit(str(e))
                    self._eval_thread = threading.Thread(target=_evaluate, daemon=True)
                    self._eval_thread.start()

                process_s = time.perf_counter() - process_started
                metrics = self._record_processed_frame(process_s)
                if self._begin_preview_delivery():
                    self.framePreview.emit(frame, tick, metrics)
                else:
                    self._record_dropped_preview()
        except Exception as e:  # surface errors instead of dying silently
            if self._running:
                self.error.emit(str(e))
        finally:
            t = getattr(self, "_eval_thread", None)
            if t is not None and t.is_alive():
                t.join(timeout=5.0)
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

    #: emitted while the user drags the ROI (payload: the new Roi)
    roiChanged = Signal(object)

    _HANDLE = 10  # resize-handle hit/draw size in widget pixels

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #101014;")
        self._frame: np.ndarray | None = None
        self._fb: FrameFeedback | None = None
        self._tick = None  # StillTick of the latest preview frame
        self._coverage_veil: np.ndarray | None = None  # (rows,cols) red opacity 0..1
        # region of interest editing
        self._roi: Roi | None = None
        self._roi_edit = False
        self._roi_square = False
        self._roi_elliptical = False  # fisheye: coverage over the image circle
        self._drag_mode: str | None = None
        self._drag_start_img: tuple[float, float] | None = None
        self._drag_start_roi: Roi | None = None
        self._hover_mode: str | None = None
        # solve progress overlay
        self._computing = False
        self._compute_progress = 0.0
        self._compute_text = ""

    def update_frame(self, frame: np.ndarray, fb: FrameFeedback, coverage_veil):
        self._frame = frame
        self._fb = fb
        self._coverage_veil = coverage_veil
        self.update()

    def update_preview(self, frame: np.ndarray, tick, coverage_veil):
        """Every camera frame: fresh image + capture-state overlay; the
        last evaluation's points/coverage stay visible."""
        self._frame = frame
        self._tick = tick
        if coverage_veil is not None:
            self._coverage_veil = coverage_veil
        self.update()

    def reset_overlay(self):
        """Drop any scan overlay (last feedback, coverage veil, ROI) - used
        for a plain camera preview that shows only the live image."""
        self._fb = None
        self._tick = None
        self._coverage_veil = None
        self._roi = None
        self.update()

    def set_computing(self, active: bool, progress: float = 0.0, text: str = ""):
        """Show/update the solve-progress overlay on the frozen frame."""
        self._computing = active
        self._compute_progress = progress
        self._compute_text = text
        self.update()

    # --- ROI editing --------------------------------------------------
    def set_roi(self, roi, square: bool = False, elliptical: bool = False):
        self._roi = roi
        self._roi_square = square
        self._roi_elliptical = elliptical
        self.update()

    def set_roi_edit(self, on: bool):
        self._roi_edit = bool(on)
        self.setMouseTracking(self._roi_edit)
        if not self._roi_edit:
            self._drag_mode = None
            self._hover_mode = None
            self.unsetCursor()
        self.update()

    def _image_size(self):
        h, w = self._frame.shape[:2]
        return (w, h)

    def _image_transform(self):
        """(scale, ox, oy, dw, dh): full-sensor pixels <-> widget pixels."""
        h, w = self._frame.shape[:2]
        scale = min(self.width() / w, self.height() / h)
        dw, dh = int(w * scale), int(h * scale)
        ox, oy = (self.width() - dw) // 2, (self.height() - dh) // 2
        return scale, ox, oy, dw, dh

    def _image_to_widget(self, x, y):
        scale, ox, oy, _, _ = self._image_transform()
        return ox + x * scale, oy + y * scale

    def _widget_to_image(self, wx, wy):
        scale, ox, oy, _, _ = self._image_transform()
        return (wx - ox) / scale, (wy - oy) / scale

    def _roi_handles(self):
        """name -> (image_x, image_y) for the active resize handles."""
        r = self._roi
        cx, cy = r.x + r.w / 2, r.y + r.h / 2
        pts = {"nw": (r.x, r.y), "ne": (r.x1, r.y),
               "sw": (r.x, r.y1), "se": (r.x1, r.y1)}
        if not self._roi_square:  # free rectangle also gets edge midpoints
            pts.update({"n": (cx, r.y), "s": (cx, r.y1),
                        "w": (r.x, cy), "e": (r.x1, cy)})
        return pts

    _CURSORS = {
        "nw": Qt.SizeFDiagCursor, "se": Qt.SizeFDiagCursor,
        "ne": Qt.SizeBDiagCursor, "sw": Qt.SizeBDiagCursor,
        "n": Qt.SizeVerCursor, "s": Qt.SizeVerCursor,
        "w": Qt.SizeHorCursor, "e": Qt.SizeHorCursor,
        "move": Qt.SizeAllCursor,
    }

    def _hit_test(self, pos):
        """Return handle name / 'move' / None for a widget-space point."""
        if self._roi is None or self._frame is None:
            return None
        wx, wy = pos.x(), pos.y()
        for name, (ix, iy) in self._roi_handles().items():
            hx, hy = self._image_to_widget(ix, iy)
            if abs(wx - hx) <= self._HANDLE and abs(wy - hy) <= self._HANDLE:
                return name
        x0, y0 = self._image_to_widget(self._roi.x, self._roi.y)
        x1, y1 = self._image_to_widget(self._roi.x1, self._roi.y1)
        if x0 <= wx <= x1 and y0 <= wy <= y1:
            return "move"
        return None

    def _resize_roi(self, r0, mode, dx, dy):
        min_size = 16.0
        if mode == "move":
            return Roi(r0.x + dx, r0.y + dy, r0.w, r0.h)
        if self._roi_square:
            # anchor the opposite corner, keep it a square
            anchors = {"nw": (r0.x1, r0.y1), "ne": (r0.x, r0.y1),
                       "sw": (r0.x1, r0.y), "se": (r0.x, r0.y)}
            ax, ay = anchors[mode]
            cur_x = self._drag_start_img[0] + dx
            cur_y = self._drag_start_img[1] + dy
            side = max(abs(cur_x - ax), abs(cur_y - ay))
            w_img, h_img = self._image_size()
            side = min(side, ax if "w" in mode else w_img - ax)
            side = min(side, ay if "n" in mode else h_img - ay)
            side = max(side, min_size)
            nx = ax - side if "w" in mode else ax
            ny = ay - side if "n" in mode else ay
            return Roi(nx, ny, side, side)
        # free rectangle: move the dragged edge(s) only
        x, y, x1, y1 = r0.x, r0.y, r0.x1, r0.y1
        if "w" in mode:
            x = min(r0.x + dx, x1 - min_size)
        if "e" in mode:
            x1 = max(r0.x1 + dx, x + min_size)
        if "n" in mode:
            y = min(r0.y + dy, y1 - min_size)
        if "s" in mode:
            y1 = max(r0.y1 + dy, y + min_size)
        return Roi(x, y, x1 - x, y1 - y)

    def mousePressEvent(self, event):
        if not (self._roi_edit and self._roi is not None
                and self._frame is not None):
            return
        if event.button() != Qt.LeftButton:
            return
        mode = self._hit_test(event.position().toPoint())
        if mode is None:
            return
        self._drag_mode = mode
        self._drag_start_img = self._widget_to_image(
            event.position().x(), event.position().y())
        self._drag_start_roi = self._roi
        event.accept()

    def mouseMoveEvent(self, event):
        if not (self._roi_edit and self._roi is not None
                and self._frame is not None):
            return
        if self._drag_mode is None:  # hover: reflect the action in the cursor
            mode = self._hit_test(event.position().toPoint())
            if mode != self._hover_mode:
                self._hover_mode = mode
                self.setCursor(self._CURSORS.get(mode, Qt.ArrowCursor))
            return
        ix, iy = self._widget_to_image(event.position().x(),
                                       event.position().y())
        dx = ix - self._drag_start_img[0]
        dy = iy - self._drag_start_img[1]
        new = self._resize_roi(self._drag_start_roi, self._drag_mode, dx, dy)
        self._roi = new.clamped(self._image_size())
        self.update()
        self.roiChanged.emit(self._roi)
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._drag_mode is None:
            return
        self._drag_mode = None
        if self._roi is not None:
            self._roi = self._roi.clamped(self._image_size())
            self.roiChanged.emit(self._roi)
        event.accept()

    _STATE_COLORS = {
        "move": QColor(255, 170, 40),      # orange: bewegen
        "hold": QColor(250, 220, 60),      # gelb: still halten
        "evaluating": QColor(90, 170, 255),  # blau: rechnet
        "captured": QColor(80, 230, 120),  # grün: aufgenommen
        "full": QColor(80, 230, 120),
    }

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._frame is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        h, w = self._frame.shape[:2]
        img = QImage(self._frame.data, w, h, self._frame.strides[0],
                     QImage.Format_Grayscale8)
        scale, ox, oy, dw, dh = self._image_transform()
        painter.drawImage(ox, oy, img.scaled(dw, dh, Qt.KeepAspectRatio,
                                             Qt.SmoothTransformation))

        # --- solve-progress overlay on the frozen frame (suppresses the
        #     stale scan overlays while the final calibration runs)
        if self._computing:
            painter.fillRect(ox, oy, dw, dh, QColor(0, 0, 0, 140))
            ring_r = int(min(dw, dh) * 0.16)
            cx, cy = ox + dw // 2, oy + dh // 2
            painter.setPen(QPen(QColor(255, 255, 255, 60), 6))
            painter.drawEllipse(cx - ring_r, cy - ring_r, 2 * ring_r, 2 * ring_r)
            painter.setPen(QPen(QColor(90, 250, 140), 6, Qt.SolidLine, Qt.RoundCap))
            painter.drawArc(cx - ring_r, cy - ring_r, 2 * ring_r, 2 * ring_r,
                            90 * 16, int(-360 * 16 * self._compute_progress))
            painter.setFont(QFont("Sans", 15, QFont.Bold))
            painter.setPen(QColor(240, 240, 240))
            painter.drawText(QRect(ox, cy + ring_r + 10, dw, 60),
                             Qt.AlignHCenter | Qt.AlignTop, self._compute_text)
            painter.end()
            return

        # --- still-capture state: colored frame border + message
        tick = self._tick
        if tick is not None:
            state = getattr(tick.capture_state, "value", str(tick.capture_state))
            color = self._STATE_COLORS.get(state, QColor(200, 200, 200))
            painter.setPen(QPen(color, 6))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(ox + 3, oy + 3, dw - 6, dh - 6)
            if state == "hold" and tick.hold_progress > 0:
                # settle progress arc top-center
                r = int(min(dw, dh) * 0.06)
                painter.setPen(QPen(color, 5, Qt.SolidLine, Qt.RoundCap))
                painter.drawArc(ox + dw // 2 - r, oy + 16, 2 * r, 2 * r,
                                90 * 16, int(-360 * 16 * tick.hold_progress))
            painter.setFont(QFont("Sans", 13, QFont.Bold))
            painter.setPen(color)
            painter.drawText(ox + 14, oy + 30, tick.message)

        # --- ROI: dim outside + boundary/handles. Drawn regardless of a
        #     FrameFeedback, so the ROI is visible and adjustable already in
        #     the measurement-free preview (where fb is None).
        roi = self._roi
        roi_rect = None
        if roi is not None:
            rx0, ry0 = self._image_to_widget(roi.x, roi.y)
            rx1, ry1 = self._image_to_widget(roi.x1, roi.y1)
            roi_rect = QRectF(rx0, ry0, rx1 - rx0, ry1 - ry0)
            dim = QPainterPath()
            dim.addRect(QRectF(ox, oy, dw, dh))
            dim.addRect(roi_rect)
            dim.setFillRule(Qt.OddEvenFill)
            painter.fillPath(dim, QColor(10, 10, 14, 110))

        fb = self._fb
        # coverage veil (scan only): uncovered cells carry a red tint whose
        # opacity grows with under-coverage; a cell covered this run is clear.
        if fb is not None and self._coverage_veil is not None:
            rows, cols = self._coverage_veil.shape
            cw, ch = dw / cols, dh / rows
            painter.setPen(Qt.NoPen)
            for r in range(rows):
                for c in range(cols):
                    a = float(self._coverage_veil[r, c])
                    if a > 0.01:
                        painter.setBrush(QBrush(QColor(225, 45, 45, int(40 + 150 * a))))
                        painter.drawRect(int(ox + c * cw), int(oy + r * ch),
                                         int(cw + 1), int(ch + 1))

        if roi_rect is not None:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(90, 200, 255), 2))
            painter.drawRect(roi_rect)
            if self._roi_edit:
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(90, 200, 255)))
                hs = self._HANDLE
                for _name, (ix, iy) in self._roi_handles().items():
                    hx, hy = self._image_to_widget(ix, iy)
                    painter.drawRect(int(hx - hs / 2), int(hy - hs / 2), hs, hs)

        if fb is None:
            painter.end()
            return

        # --- detected points
        color = QColor(80, 255, 120) if fb.keyframe else QColor(120, 220, 255)
        painter.setPen(QPen(color, 2))
        painter.setBrush(Qt.NoBrush)
        for x, y in fb.points:
            painter.drawEllipse(int(ox + x * scale) - 3, int(oy + y * scale) - 3, 6, 6)

        # --- center: progress ring while scanning, a big success check once
        #     the scan is complete (so "nothing turns green anymore" reads as
        #     DONE, not broken)
        ring_r = int(min(dw, dh) * 0.16)
        cx, cy = ox + dw // 2, oy + dh // 2
        converged = fb.state in (SessionState.CONVERGED, SessionState.FINISHED)
        if converged:
            painter.setBrush(QBrush(QColor(20, 40, 25, 190)))
            painter.setPen(QPen(QColor(80, 230, 120), 8))
            painter.drawEllipse(cx - ring_r, cy - ring_r, 2 * ring_r, 2 * ring_r)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(120, 240, 150), 12, Qt.SolidLine,
                                Qt.RoundCap, Qt.RoundJoin))
            painter.drawPolyline(QPolygonF([
                QPointF(cx - 0.42 * ring_r, cy + 0.02 * ring_r),
                QPointF(cx - 0.12 * ring_r, cy + 0.34 * ring_r),
                QPointF(cx + 0.46 * ring_r, cy - 0.34 * ring_r)]))
        else:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(255, 255, 255, 60), 6))
            painter.drawEllipse(cx - ring_r, cy - ring_r, 2 * ring_r, 2 * ring_r)
            painter.setPen(QPen(QColor(90, 250, 140), 6, Qt.SolidLine, Qt.RoundCap))
            span = int(-360 * 16 * fb.progress)
            painter.drawArc(cx - ring_r, cy - ring_r, 2 * ring_r, 2 * ring_r, 90 * 16, span)

        # --- status text (bottom)
        if converged:
            painter.setFont(QFont("Sans", 15, QFont.Bold))
            painter.setPen(QColor(120, 240, 150))
            painter.drawText(QRect(ox, cy + ring_r + 14, dw, 34),
                             Qt.AlignHCenter | Qt.AlignTop,
                             "Scan erfolgreich – Auswertung kann beginnen")
            painter.setFont(QFont("Sans", 10))
            painter.setPen(QColor(215, 225, 215))
            painter.drawText(QRect(ox, cy + ring_r + 46, dw, 28),
                             Qt.AlignHCenter | Qt.AlignTop,
                             "Rechts „Abschließen zur Berechnung“ – oder weiter verdichten")
        else:
            painter.setFont(QFont("Sans", 11, QFont.Bold))
            painter.setPen(QColor(240, 240, 240))
            msg = ("Pattern vor die Kamera halten …"
                   if fb.state is SessionState.WAITING else self._hint(fb))
            painter.drawText(ox + 12, oy + dh - 34, msg)
            # why the last settled evaluation did NOT capture (still-capture
            # feedback - otherwise a rejection silently re-arms "hold still")
            if fb.state is SessionState.SCANNING and not fb.keyframe:
                txt = REASON_TEXT.get(fb.reason, fb.reason)
                painter.setFont(QFont("Sans", 11, QFont.Bold))
                painter.setPen(QColor(255, 175, 90))
                painter.drawText(ox + 12, oy + dh - 14,
                                 f"Zuletzt nicht aufgenommen: {txt}")
        painter.end()

    @staticmethod
    def _hint(fb: FrameFeedback) -> str:
        if fb.reason == "blurry":
            return "Bewegung zu schnell - langsamer bewegen"
        if fb.coverage < 0.5:
            return "Kamera bewegen: Pattern über den ganzen Sensor führen"
        if fb.tilt_coverage < 0.5:
            return "Pattern aus schrägeren Winkeln aufnehmen"
        return ("Ränder/Ecken abdecken - Pattern darf über den Bildrand "
                "hinausragen, sichtbare Marker genügen")


def assess_calibration(result, ocam_result, *, coverage=None, tilt=None,
                       target_coverage: float = 0.85):
    """Layman-friendly quality assessment.

    Returns ``(verdict, color_hex, details)`` where ``details`` explains what
    the numbers mean and, when the result is weak, what to do on a follow-up
    scan. ``coverage``/``tilt`` are the session's sensor-coverage and tilt
    fractions (0..1); pass ``None`` if unknown.
    """
    rms = ocam_result.rms if ocam_result is not None else result.rms
    errs = (np.concatenate(result.per_point_errors)
            if result.per_point_errors else np.empty(0))
    total = int(errs.size)
    bad = int((errs > 2.0).sum())
    bad_frac = bad / total if total else 0.0

    if rms < 0.5:
        rms_word = "sehr gut (unter 0,5 px)"
    elif rms < 1.0:
        rms_word = "gut"
    elif rms < 2.0:
        rms_word = "brauchbar, aber verbesserbar"
    else:
        rms_word = "hoch – deutet auf Ausreißer oder zu wenig Vielfalt hin"

    lines = [
        "Was das Ergebnis bedeutet:",
        f"• Genauigkeit (mittl. Reprojektionsfehler): {rms:.2f} px – {rms_word}. "
        "Kleiner ist besser: der typische Abstand zwischen erkanntem und "
        "vom Modell vorhergesagtem Punkt.",
    ]
    if total:
        lines.append(f"• Punkte: {total} verwendet, davon {bad} auffällig "
                     f"(>2 px) = {bad_frac * 100:.0f} %.")
    if coverage is not None:
        cov_word = ("gut abgedeckt" if coverage >= 0.9 * target_coverage
                    else "lückenhaft")
        lines.append(f"• Abdeckung des Bildbereichs: {coverage * 100:.0f} % – {cov_word}.")
    if tilt is not None:
        tilt_word = "genug Winkel" if tilt >= 0.5 else "zu wenig gekippt"
        lines.append(f"• Blickwinkel-Vielfalt: {tilt * 100:.0f} % der Richtungen – {tilt_word}.")

    tips = []
    if tilt is not None and tilt < 0.5:
        tips.append("Kippe das Board deutlich stärker (nach links/rechts/oben/"
                    "unten neigen, nicht nur frontal halten) – die schrägen "
                    "Ansichten bestimmen die Linsenverzeichnung und senken den "
                    "Fehler am meisten.")
    if coverage is not None and coverage < 0.9 * target_coverage:
        tips.append("Führe das Board bis an die Bildränder und in die Ecken "
                    "(bei Fisheye bis zum Rand des Bildkreises) – dunkelrote "
                    "Zonen zeigen, wo noch etwas fehlt.")
    if bad_frac > 0.15:
        tips.append("Scanne langsamer und halte beim Aufnehmen kurz still – "
                    "viele auffällige Punkte kommen von Bewegungsunschärfe.")

    n = len(tips)
    if rms < 1.0 and n == 0:
        verdict, color = "Sehr gut – direkt verwendbar", "#2e9e4f"
    elif rms < 2.0 and n <= 1:
        verdict, color = "Brauchbar – für mehr Genauigkeit nachscannen", "#c9a227"
    else:
        verdict, color = "Besser nachscannen", "#d9534f"

    lines.append("")
    if tips:
        lines.append('So verbesserst du es – „Zurück zum Scannen" und dort '
                     '„Zusätzlich drüberscannen":')
        lines.extend(f"→ {t}" for t in tips)
    else:
        lines.append("Keine Schwächen erkannt – die Kalibrierung kann so "
                     "übernommen werden.")
    return verdict, color, "\n".join(lines)


def calibration_strings(result, ocam_result, camera_id: str,
                        pixel_size_mm: float | None = None) -> tuple[str, str]:
    """``(json_str, xml_str)`` of the calibration for the dataset viewer.

    The XML is produced by the very writers used for the exported file (via a
    throwaway temp file) so what the user copies matches the exported file;
    the JSON is a compact parameter dump.
    """
    import json
    import os
    import tempfile
    from pathlib import Path
    from ..io.export import write_ocam_xml, write_vendor_xml

    data = {
        "camera_id": camera_id,
        "model": result.model.value,
        "image_width": int(result.image_size[0]),
        "image_height": int(result.image_size[1]),
        "fx": round(float(result.fx), 6),
        "fy": round(float(result.fy), 6),
        "cx": round(float(result.cx), 6),
        "cy": round(float(result.cy), 6),
        "dist_coeffs": [round(float(v), 8)
                        for v in np.asarray(result.dist_coeffs).ravel()],
        "rms_px": round(float(result.rms), 6),
        "n_views": int(result.n_views),
        "n_points": int(result.n_points),
    }
    if ocam_result is not None:
        m = ocam_result.model
        data["ocam"] = {
            "cx": round(float(m.cx), 6), "cy": round(float(m.cy), 6),
            "c": round(float(m.c), 8), "d": round(float(m.d), 8),
            "e": round(float(m.e), 8),
            "poly": [round(float(v), 8) for v in m.poly],
            "rms_px": round(float(ocam_result.rms), 6),
        }
    json_str = json.dumps(data, indent=2)

    pts = np.vstack(result.used_image_points) if result.used_image_points else None
    errs = (np.concatenate(result.per_point_errors)
            if result.per_point_errors else None)
    fd, tmp = tempfile.mkstemp(suffix=".xml")
    os.close(fd)
    try:
        if ocam_result is not None:
            opts = (np.vstack(ocam_result.used_image_points)
                    if ocam_result.used_image_points else pts)
            write_ocam_xml(ocam_result, tmp, camera_id, points=opts)
        else:
            ps = (pixel_size_mm, pixel_size_mm) if pixel_size_mm else None
            write_vendor_xml(result, tmp, camera_id, pixel_size_mm=ps,
                             points=pts, errors=errs)
        xml_str = Path(tmp).read_text(encoding="utf-8")
    finally:
        os.remove(tmp)
    return json_str, xml_str


def error_stats_text(result, ocam_result) -> str:
    """One-line-ish reprojection-error breakdown for the result dataset view,
    from the same source (OCam for fisheye, else pinhole) as the map."""
    src = ocam_result if ocam_result is not None else result
    errs = (np.concatenate(src.per_point_errors)
            if getattr(src, "per_point_errors", None) else np.empty(0))
    if errs.size == 0:
        return "Keine Fehlerdaten verfügbar."
    n = int(errs.size)

    def pct(mask):
        return 100.0 * int(np.count_nonzero(mask)) / n
    return (
        f"Reprojektionsfehler über {n} Punkte (Abstand erkannt ↔ vorhergesagt):\n"
        f"  Median {np.median(errs):.2f} px   Mittel {errs.mean():.2f} px   "
        f"Max {errs.max():.2f} px\n"
        f"  ≤1px: {pct(errs <= 1):.0f}%    ≤2px: {pct(errs <= 2):.0f}%    "
        f"≤3px: {pct(errs <= 3):.0f}%    >3px: {pct(errs > 3):.0f}%\n"
        "Die Karte zeigt jeden Punkt als Fehlervektor (grün ≤1px … rot >3px); "
        "blauer Kreis = Vertrauensradius um den Bildmittelpunkt.")


class ResultDatasetDialog(QDialog):
    """Popup with the calibration dataset: XML and JSON tabs (each with a
    copy-to-clipboard button) plus a 'Fehlerkarte' tab (coverage/error map +
    error breakdown)."""

    def __init__(self, xml_str: str, json_str: str, result_map=None,
                 stats_text: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Kalibrierergebnis")
        self.resize(700, 580)
        tabs = QTabWidget()
        for label, text in (("XML", xml_str), ("JSON", json_str)):
            page = QWidget()
            edit = QPlainTextEdit(text)
            edit.setReadOnly(True)
            edit.setLineWrapMode(QPlainTextEdit.NoWrap)
            edit.setStyleSheet("font-family: monospace;")
            btn = QPushButton(f"{label} in Zwischenablage kopieren")
            btn.clicked.connect(lambda _checked=False, t=text, b=btn:
                                self._copy(t, b))
            v = QVBoxLayout()
            v.addWidget(edit, 1)
            v.addWidget(btn)
            page.setLayout(v)
            tabs.addTab(page, label)

        if result_map is not None:
            page = QWidget()
            v = QVBoxLayout()
            img = QLabel()
            img.setAlignment(Qt.AlignCenter)
            img.setMinimumSize(480, 360)
            img.setStyleSheet("background-color: #101014;")
            h, w = result_map.shape[:2]
            qimg = QImage(result_map.data, w, h, result_map.strides[0],
                          QImage.Format_BGR888)
            img.setPixmap(QPixmap.fromImage(qimg.copy()).scaled(
                600, 440, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            v.addWidget(img, 1)
            info = QLabel(stats_text)
            info.setWordWrap(True)
            info.setStyleSheet("font-family: monospace; padding: 6px;")
            v.addWidget(info)
            page.setLayout(v)
            tabs.addTab(page, "Fehlerkarte")

        close = QPushButton("Schließen")
        close.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(close)
        lay = QVBoxLayout()
        lay.addWidget(tabs, 1)
        lay.addLayout(row)
        self.setLayout(lay)

    @staticmethod
    def _copy(text: str, btn: QPushButton):
        QApplication.clipboard().setText(text)
        old = btn.text()
        btn.setText("Kopiert ✓")
        # revert the label after a moment (no-op if the dialog is gone)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(1200, lambda: btn.setText(old))


class ResultView(QWidget):
    """Post-scan review page: traffic-light coverage/error map with the
    calibration summary. Green = good, used observations; red = bad."""

    backToScan = Signal()
    finishExport = Signal()
    showDataset = Signal()

    def __init__(self):
        super().__init__()
        self._pixmap: QPixmap | None = None
        self.image = QLabel("")
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setMinimumSize(640, 480)
        self.image.setStyleSheet("background-color: #101014;")
        self.verdict = QLabel("")
        self.verdict.setAlignment(Qt.AlignHCenter)
        self.verdict.setStyleSheet("font-size:16px; font-weight:bold; padding:6px;")
        self.summary = QLabel("")
        self.summary.setStyleSheet("font-family: monospace; padding: 2px 6px;")
        self.summary.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.details = QLabel("")
        self.details.setWordWrap(True)
        self.details.setStyleSheet("padding: 2px 6px;")
        self.details.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.btn_back = QPushButton("Zurück zum Scannen")
        self.btn_back.setToolTip(
            "Zurück zum Scannen und ein neuer Abtast-Durchgang, um schwache "
            "Bereiche nachzulegen. Alle bisherigen Aufnahmen bleiben erhalten.")
        self.btn_show = QPushButton("Kalibrierergebnis anzeigen")
        self.btn_show.setToolTip("Zeigt den Kalibrier-Datensatz als JSON und "
                                 "XML zum Kopieren.")
        self.btn_export = QPushButton("Kalibrierdatei erstellen")
        self.btn_back.clicked.connect(self.backToScan.emit)
        self.btn_show.clicked.connect(self.showDataset.emit)
        self.btn_export.clicked.connect(self.finishExport.emit)
        for _b, _bg in ((self.btn_back, "#2d6cdf"), (self.btn_show, "#2e9e4f"),
                        (self.btn_export, "#2e9e4f")):
            _b.setStyleSheet(f"QPushButton{{font-weight:bold;padding:10px;"
                             f"background:{_bg};color:white;border-radius:4px;}}")

        buttons = QHBoxLayout()
        buttons.addWidget(self.btn_back)
        buttons.addStretch(1)
        buttons.addWidget(self.btn_show)
        buttons.addWidget(self.btn_export)
        lay = QVBoxLayout()
        lay.addWidget(self.image, 1)
        lay.addWidget(self.verdict)
        lay.addWidget(self.summary)
        lay.addWidget(self.details)
        lay.addLayout(buttons)
        self.setLayout(lay)

    def set_result(self, result, ocam_result, bgr_map: np.ndarray, *,
                   coverage=None, tilt=None, target_coverage: float = 0.85):
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
            f"Punkte: {len(errs)}  (gut ≤1px: {good}, schlecht >2px: {bad})\n"
            f"fx={result.fx:.2f}  fy={result.fy:.2f}  "
            f"cx={result.cx:.2f}  cy={result.cy:.2f}   RMS={rms:.3f} px\n"
            f"Grün = gute, verwendete Punkte · Rot = schlechte Punkte.")
        verdict, color, details = assess_calibration(
            result, ocam_result, coverage=coverage, tilt=tilt,
            target_coverage=target_coverage)
        self.verdict.setText(verdict)
        self.verdict.setStyleSheet(
            f"font-size:16px; font-weight:bold; padding:6px; color:{color};")
        self.details.setText(details)

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
        #: ROI chosen during preview, applied to the session on Start
        self._pending_roi: Roi | None = None
        self._source: FrameSource | None = None
        self._capture_running = False
        self._last_stats_update = 0.0
        self._last_fb = None
        self._solve_thread: QThread | None = None
        self._solve_worker: SolveWorker | None = None
        self._solving = False
        self._result = None
        self._ocam_result = None
        self._result_map = None

        self.view = LiveView()
        self.result_view = ResultView()
        self.result_view.backToScan.connect(self.back_to_scan)
        self.result_view.showDataset.connect(self.show_dataset)
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
        self.chk_roi = QCheckBox("ROI anpassen")
        self.chk_roi.setEnabled(False)
        self.lbl_camera = QLabel("Kamera")
        self.cmb_camera = QComboBox()
        self.lbl_camera_status = QLabel("")
        self.lbl_camera_status.setWordWrap(True)
        self.btn_refresh_cameras = QPushButton("Kameras aktualisieren")
        self.btn_preview = QPushButton("Vorschau")
        self.btn_preview.setToolTip("Zeigt das Livebild der gewählten Kamera "
                                    "ohne Messung – zum Ausrichten/Prüfen von "
                                    "Bild und Belichtung. Mit „Stop“ beenden.")
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
        # densification: re-open novelty so already-covered - especially
        # thinly-covered - areas can be scanned over again for more keyframes
        self.btn_densify = QPushButton("Zusätzlich drüberscannen")
        self.btn_densify.setToolTip(
            "Legt eine neue Abtast-Schicht an: bereits erfasste Bereiche "
            "zählen wieder als neu, sodass du dünn besetzte Zonen gezielt "
            "verdichten kannst. Die bisherigen Aufnahmen bleiben erhalten.")
        self.btn_densify.setEnabled(False)
        self.btn_densify.setStyleSheet(
            "QPushButton{font-weight:bold;padding:10px;background:#2d6cdf;"
            "color:white;border-radius:4px;}"
            "QPushButton:disabled{background:#33384a;color:#8890a0;}")
        self.btn_finish = QPushButton("Abschließen zur Berechnung")
        self.btn_finish.setToolTip(
            "Beendet das Scannen und berechnet die Kalibrierung aus den "
            "gesammelten Aufnahmen.")
        self.btn_finish.setEnabled(False)
        self.btn_finish.setStyleSheet(
            "QPushButton{font-weight:bold;padding:10px;background:#2e9e4f;"
            "color:white;border-radius:4px;}"
            "QPushButton:disabled{background:#33384a;color:#8890a0;}")
        self.btn_cancel = QPushButton("Abbrechen")
        self.btn_cancel.setEnabled(False)
        self.cmb_model.currentIndexChanged.connect(self._on_model_changed)
        self.cmb_target.currentIndexChanged.connect(self._on_target_changed)
        self.cmb_camera.currentIndexChanged.connect(self._on_camera_changed)
        self.btn_refresh_cameras.clicked.connect(self.refresh_cameras)
        self.btn_preview.clicked.connect(self.preview)
        self.chk_custom_exposure.toggled.connect(self.spn_exposure_ms.setEnabled)
        self.chk_custom_exposure.toggled.connect(lambda _checked: self._update_camera_status())
        self.chk_reset_defaults.toggled.connect(lambda _checked: self._update_camera_status())
        self.spn_exposure_ms.valueChanged.connect(lambda _value: self._update_camera_status())
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_densify.clicked.connect(self.densify)
        self.btn_finish.clicked.connect(self.finish)
        self.btn_cancel.clicked.connect(self.cancel_solve)
        self.chk_roi.toggled.connect(self.view.set_roi_edit)
        self.view.roiChanged.connect(self._on_roi_changed)
        self.spn_exposure_ms.setEnabled(self.chk_custom_exposure.isChecked())

        side = QVBoxLayout()
        side.addWidget(self.lbl_model)
        side.addWidget(self.cmb_model)
        side.addWidget(self.lbl_target)
        side.addWidget(self.cmb_target)
        if self._camera_mode:
            side.addWidget(self.lbl_camera)
            side.addWidget(self.cmb_camera)
            cam_row = QHBoxLayout()
            cam_row.addWidget(self.btn_refresh_cameras)
            cam_row.addWidget(self.btn_preview)
            side.addLayout(cam_row)
            side.addWidget(self.chk_roi)
            side.addWidget(self.chk_reset_defaults)
            side.addWidget(self.chk_custom_exposure)
            side.addWidget(self.spn_exposure_ms)
            side.addWidget(self.lbl_camera_status)
        else:
            side.addWidget(self.chk_roi)
        side.addWidget(self.stats)
        side.addStretch(1)
        side.addWidget(self.btn_start)
        side.addWidget(self.btn_densify)
        side.addWidget(self.btn_finish)
        side.addWidget(self.btn_stop)
        side.addWidget(self.btn_cancel)
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
    def preview(self):
        """Show the selected camera's live stream without any measurement -
        for framing/exposure checks. „Start" begins the real capture, „Stop"
        releases the camera."""
        self.stop()
        self._stack.setCurrentIndex(0)
        self._result = self._ocam_result = self._result_map = None
        self._last_fb = None
        self._session = None
        if self._camera_mode and self.cmb_camera.count() == 0:
            QMessageBox.warning(self, "Keine Kamera",
                                "Es ist keine GigE/GenICam-Kamera ausgewaehlt.")
            return
        try:
            self._source = self._make_source(self._selected_camera(),
                                             self._source_options())
            self._source.open()
        except Exception as e:
            QMessageBox.critical(self, "Fehler", str(e))
            return
        self.view.reset_overlay()
        # allow ROI adjustment already during preview: seed a default ROI for
        # the selected model and remember edits for the upcoming session
        model = CameraModel(self._selected_model_value())
        is_fish = model is CameraModel.FISHEYE
        self._pending_roi = Roi.default(self._source.image_size, model)
        self.view.set_roi(self._pending_roi, square=is_fish, elliptical=is_fish)
        self.chk_roi.setEnabled(True)
        self._worker = CaptureWorker(self._source, None)  # None -> preview only
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.framePreview.connect(self._on_preview)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._on_capture_finished)
        self._thread.start()
        self._capture_running = True
        self._last_stats_update = 0.0
        self.btn_stop.setEnabled(True)
        self.btn_densify.setEnabled(False)
        self.btn_finish.setEnabled(False)
        self._set_camera_controls_enabled(False)
        self.stats.setText("Vorschau – Livebild, keine Messung.\n"
                           "Start beginnt die Kalibrier-Aufnahme, "
                           "Stop gibt die Kamera frei.")

    @Slot()
    def start(self):
        self.stop()
        self._stack.setCurrentIndex(0)
        self._result = self._ocam_result = self._result_map = None
        self._last_fb = None
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
        # carry over an ROI adjusted during preview
        if self._pending_roi is not None:
            self._session.set_roi(self._pending_roi)
        self._worker = CaptureWorker(self._source, self._session)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.frameProcessed.connect(self._on_frame)
        self._worker.framePreview.connect(self._on_preview)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._on_capture_finished)
        self._thread.start()
        self._capture_running = True
        self._last_stats_update = 0.0
        self.view.set_roi(
            self._session.roi,
            square=self._session.cfg.model is CameraModel.FISHEYE,
            elliptical=self._session.cfg.model is CameraModel.FISHEYE)
        self.chk_roi.setEnabled(True)
        self.btn_start.setText("Neu starten")
        self.btn_stop.setEnabled(True)
        self.btn_densify.setEnabled(True)
        self.btn_finish.setEnabled(True)
        self._set_camera_controls_enabled(False)
        self.cmb_model.setEnabled(False)
        self.cmb_target.setEnabled(False)

    @Slot()
    def densify(self):
        """Start a new densification run on the live session (see
        CalibrationSession.start_run): already-covered areas count as new
        again, so scanning over thinly-covered zones adds more keyframes."""
        if self._session is None:
            return
        self._session.start_run()
        self.stats.setText(
            "Neuer Abtast-Durchgang.\nFahr jetzt vor allem die dunkelroten "
            "(dünn besetzten) Bereiche erneut ab - dort werden wieder "
            "Aufnahmen angenommen.")

    @Slot(object)
    def _on_roi_changed(self, roi):
        # remember it so a preview-time ROI carries over to the session
        self._pending_roi = roi
        if self._session is not None:
            self._session.set_roi(roi)

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
        self.btn_densify.setEnabled(False)
        self.chk_roi.setChecked(False)
        self.chk_roi.setEnabled(False)
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
        """A completed evaluation (still-capture) - update coverage/points."""
        try:
            veil = self._session.coverage.veil_alpha() if self._session else None
            self.view.update_frame(frame, fb, veil)
            self._last_fb = fb
            self._update_stats(fb, metrics)
        finally:
            # release directly from the UI thread (see _on_preview)
            worker = self._worker
            if worker is not None:
                worker.on_frame_displayed()

    @Slot(np.ndarray, object, object)
    def _on_preview(self, frame, tick, metrics: RuntimeMetrics):
        """Every camera frame at native fps - cheap overlay only."""
        try:
            veil = self._session.coverage.veil_alpha() if self._session else None
            self.view.update_preview(frame, tick, veil)
            fb = getattr(self, "_last_fb", None)
            if fb is not None and tick is not None:
                self._update_stats(fb, metrics, tick)
            elif time.perf_counter() - self._last_stats_update >= 0.25:
                msg = tick.message if tick is not None else "Vorschau – Livebild (keine Messung)"
                self.stats.setText(
                    f"Cap {metrics.capture_fps:4.1f} fps | UI {metrics.display_fps:4.1f} fps\n"
                    f"{msg}")
                self._last_stats_update = time.perf_counter()
        finally:
            # Call directly from the UI thread: queued delivery back into the
            # worker thread would stall because the capture loop keeps that
            # thread busy and the release signal would never be processed.
            worker = self._worker
            if worker is not None:
                worker.on_frame_displayed()

    def _update_stats(self, fb: FrameFeedback, metrics: RuntimeMetrics, tick=None):
        now = time.perf_counter()
        if now - self._last_stats_update < 0.25:
            return
        outcome = ("aufgenommen ✓" if fb.keyframe
                   else f"nicht aufgenommen – {REASON_TEXT.get(fb.reason, fb.reason)}")
        lines = [
            f"Status {fb.state.value} | Marker {len(fb.ids)} | KF {fb.n_keyframes}",
            f"Letzte Auswertung: {len(fb.points)} Punkte, {outcome}",
            f"Abd {fb.coverage*100:.0f}% | Winkel {fb.tilt_coverage*100:.0f}%",
            f"Laufz {metrics.elapsed_s:6.1f}s | Cap {metrics.capture_fps:4.1f} fps | UI {metrics.display_fps:4.1f} fps",
            f"Frames {metrics.captured_frames}/{metrics.displayed_frames} | Drops {metrics.dropped_previews}",
        ]
        if tick is not None:
            lines.append(f"Motion {tick.motion:5.2f} | {tick.message}")
        if fb.result is not None:
            r = fb.result
            lines += [
                "",
                f"fx {r.fx:8.2f} | fy {r.fy:8.2f} | RMS {r.rms:6.3f} px",
                f"cx {r.cx:8.2f} | cy {r.cy:8.2f} | Views {r.n_views}",
            ]
        self.stats.setText("\n".join(lines))
        self._last_stats_update = now

    @Slot(str)
    def _on_error(self, msg):
        self.stop()
        QMessageBox.critical(self, "Fehler", msg)

    @Slot()
    def _on_capture_finished(self):
        self._capture_running = False
        self._release_source()
        if self._solving:
            return  # solve overlay owns the UI; keep Start/Finish disabled
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

    def _camera_identifier(self) -> str:
        """Identifier used in exported filenames and the XML camera id -
        the selected camera's serial number when known, so results are
        traceable to the physical camera."""
        cam = self._selected_camera() or {}
        return (cam.get("serial_number") or self._initial_camera_serial
                or self._camera_id or "camera")

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
        self.btn_preview.setEnabled(enabled)
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
        self._solving = True
        self.stop()  # also drops ROI edit mode; frozen frame stays on screen

        # final solve in a worker thread - the fisheye finish can take a
        # while and used to freeze the GUI here. Progress and a cancel
        # action are shown as an overlay on the frozen camera image.
        self.btn_finish.setEnabled(False)
        self.btn_densify.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.btn_cancel.setText("Abbrechen")
        self.btn_cancel.setEnabled(True)
        self.stats.setText("Ergebnis wird berechnet …")
        self.view.set_computing(True, 0.0, "Berechne… 0%")
        self._solve_worker = SolveWorker(self._session)
        self._solve_thread = QThread()
        self._solve_worker.moveToThread(self._solve_thread)
        self._solve_thread.started.connect(self._solve_worker.run)
        self._solve_worker.progress.connect(self._on_solve_progress)
        self._solve_worker.finished.connect(self._on_solve_done)
        self._solve_worker.error.connect(self._on_solve_error)
        self._solve_worker.cancelled.connect(self._on_solve_cancelled)
        self._solve_worker.finished.connect(self._solve_thread.quit)
        self._solve_worker.error.connect(self._solve_thread.quit)
        self._solve_worker.cancelled.connect(self._solve_thread.quit)
        self._solve_thread.start()

    def _finish_solve_ui(self):
        """Shared teardown for every terminal solve outcome."""
        self._solving = False
        self.view.set_computing(False)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setText("Abbrechen")
        self.btn_finish.setEnabled(True)
        self.btn_start.setEnabled(True)

    @Slot(float, str)
    def _on_solve_progress(self, percent, text):
        self.view.set_computing(True, percent / 100.0,
                                f"Berechne… {percent:.0f}%\n{text}")

    @Slot()
    def cancel_solve(self):
        if self._solve_worker is not None:
            self._solve_worker.cancel()
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setText("Abbrechen…")

    @Slot(object, object, object)
    def _on_solve_done(self, result, ocam_result, bgr_map):
        self._finish_solve_ui()
        self._result = result
        self._ocam_result = ocam_result
        self._result_map = bgr_map
        cov = tilt = None
        target_cov = 0.85
        if self._session is not None:
            cov = self._session.coverage.fraction
            tilt = self._session.coverage.tilt_fraction
            target_cov = self._session._target_coverage()
        self.result_view.set_result(result, ocam_result, bgr_map,
                                    coverage=cov, tilt=tilt,
                                    target_coverage=target_cov)
        self._stack.setCurrentIndex(1)
        self.stats.setText(
            "Ergebnis prüfen. Die Bewertung unter dem Bild sagt, ob es reicht "
            "oder was du nachscannen solltest.\n\n"
            "„Zurück zum Scannen“ → schwache Bereiche nachlegen\n"
            "„Kalibrierdatei erstellen“ → abschließen")

    @Slot()
    def _on_solve_cancelled(self):
        self._finish_solve_ui()
        self.stats.setText("Berechnung abgebrochen.\n"
                           "Ergebnis berechnen -> erneut versuchen\n"
                           "Start -> neu scannen")

    @Slot(str)
    def _on_solve_error(self, msg):
        self._finish_solve_ui()
        QMessageBox.critical(self, "Kalibrierung fehlgeschlagen", msg)

    @Slot()
    def back_to_scan(self):
        """Resume scanning with the existing session - keyframes,
        coverage and the current model are kept and improved."""
        self._stack.setCurrentIndex(0)
        if not self._session:
            return
        # start_run (not resume): coming back after a result, re-open coverage
        # novelty so the user can immediately scan more into thin/empty areas
        self._session.start_run()
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
        self._worker.framePreview.connect(self._on_preview)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._on_capture_finished)
        self._thread.start()
        self._capture_running = True
        self._last_stats_update = 0.0
        self.view.set_roi(
            self._session.roi,
            square=self._session.cfg.model is CameraModel.FISHEYE,
            elliptical=self._session.cfg.model is CameraModel.FISHEYE)
        self.chk_roi.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_densify.setEnabled(True)
        self.btn_finish.setEnabled(True)
        self._set_camera_controls_enabled(False)
        self.cmb_model.setEnabled(False)
        self.cmb_target.setEnabled(False)

    @Slot()
    def show_dataset(self):
        if self._result is None:
            return
        try:
            json_str, xml_str = calibration_strings(
                self._result, self._ocam_result, self._camera_identifier(),
                self._pixel_size)
        except Exception as e:
            QMessageBox.critical(self, "Fehler", str(e))
            return
        stats = error_stats_text(self._result, self._ocam_result)
        ResultDatasetDialog(xml_str, json_str, self._result_map, stats,
                            parent=self).exec()

    @Slot()
    def export_calibration(self):
        if self._result is None:
            return
        from pathlib import Path
        ident = self._camera_identifier()
        suffix = "ocam" if self._ocam_result is not None else "ocv"
        default = str(Path.home() / "Documents" / f"{ident}-{suffix}.xml")
        xml_path, _ = QFileDialog.getSaveFileName(
            self, "Kalibrierung exportieren", default, "XML-Datei (*.xml)")
        if not xml_path:
            return
        try:
            written = write_calibration_files(
                self._result, self._ocam_result, xml_path,
                ident, self._pixel_size, self._result_map)
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
