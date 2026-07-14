"""Qt main window: live view with FaceTime-style coverage guidance."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import (QBrush, QColor, QFont, QImage, QPainter, QPen,
                           QPixmap)
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel,
                               QMainWindow, QMessageBox, QPushButton,
                               QVBoxLayout, QWidget)

from ..calibration import CameraModel
from ..capture.source import FrameSource
from ..io import export_all
from ..session import CalibrationSession, FrameFeedback, SessionConfig, SessionState


class CaptureWorker(QObject):
    """Runs the source + session loop in a QThread."""

    frameProcessed = Signal(np.ndarray, object)  # gray frame, FrameFeedback
    finished = Signal()
    error = Signal(str)

    def __init__(self, source: FrameSource, session: CalibrationSession):
        super().__init__()
        self.source = source
        self.session = session
        self._running = False

    @Slot()
    def run(self):
        self._running = True
        try:
            while self._running:
                ok, frame, ts = self.source.read()
                if not ok:
                    break
                fb = self.session.process(frame, ts)
                self.frameProcessed.emit(frame, fb)
        except Exception as e:  # surface errors instead of dying silently
            self.error.emit(str(e))
        finally:
            self.finished.emit()

    def stop(self):
        self._running = False


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

        # --- coverage veil: darken cells not yet covered
        if self._coverage_mask is not None:
            rows, cols = self._coverage_mask.shape
            cw, ch = dw / cols, dh / rows
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(10, 10, 20, 150))
            for r in range(rows):
                for c in range(cols):
                    if not self._coverage_mask[r, c]:
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


class MainWindow(QMainWindow):
    def __init__(self, make_source, make_session, camera_id: str = "camera",
                 pixel_size_mm: float | None = None):
        """``make_source``/``make_session`` are factories so a session can
        be restarted without rebuilding the window."""
        super().__init__()
        self.setWindowTitle("camcalib2 - Kamera-Kalibrierung")
        self._make_source = make_source
        self._make_session = make_session
        self._camera_id = camera_id
        self._pixel_size = pixel_size_mm
        self._thread: QThread | None = None
        self._worker: CaptureWorker | None = None
        self._session: CalibrationSession | None = None
        self._source: FrameSource | None = None

        self.view = LiveView()
        self.stats = QLabel("-")
        self.stats.setStyleSheet("font-family: monospace; padding: 6px;")
        self.stats.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.stats.setWordWrap(True)
        self.stats.setMinimumHeight(320)
        self.btn_start = QPushButton("Start")
        self.btn_finish = QPushButton("Fertigstellen && Export")
        self.btn_finish.setEnabled(False)
        self.btn_start.clicked.connect(self.start)
        self.btn_finish.clicked.connect(self.finish)

        side = QVBoxLayout()
        side.addWidget(self.stats)
        side.addStretch(1)
        side.addWidget(self.btn_start)
        side.addWidget(self.btn_finish)
        sidew = QWidget()
        sidew.setLayout(side)
        sidew.setFixedWidth(280)

        lay = QHBoxLayout()
        lay.addWidget(self.view, 1)
        lay.addWidget(sidew)
        central = QWidget()
        central.setLayout(lay)
        self.setCentralWidget(central)

    # ------------------------------------------------------------------
    @Slot()
    def start(self):
        self.stop()
        try:
            self._source = self._make_source()
            self._source.open()
            self._session = self._make_session(self._source.image_size)
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
        self._thread.start()
        self.btn_start.setText("Neu starten")
        self.btn_finish.setEnabled(True)

    def stop(self):
        if self._worker:
            self._worker.stop()
        if self._thread:
            self._thread.quit()
            self._thread.wait(3000)
        if self._source:
            try:
                self._source.close()
            except Exception:
                pass
        self._worker = self._thread = self._source = None

    @Slot(np.ndarray, object)
    def _on_frame(self, frame, fb: FrameFeedback):
        mask = self._session.coverage.mask() if self._session else None
        self.view.update_frame(frame, fb, mask)
        lines = [
            f"Status:     {fb.state.value}",
            f"Marker:     {len(fb.ids)}",
            f"Keyframes:  {fb.n_keyframes}",
            f"Abdeckung:  {fb.coverage*100:.0f} %",
            f"Winkel:     {fb.tilt_coverage*100:.0f} %",
        ]
        if fb.result is not None:
            r = fb.result
            lines += [
                "", "-- Intrinsiken (live) --",
                f"fx: {r.fx:9.2f}", f"fy: {r.fy:9.2f}",
                f"cx: {r.cx:9.2f}", f"cy: {r.cy:9.2f}",
                f"RMS: {r.rms:7.3f} px",
                f"Views: {r.n_views}",
            ]
        self.stats.setText("\n".join(lines))

    @Slot(str)
    def _on_error(self, msg):
        QMessageBox.critical(self, "Fehler", msg)

    @Slot()
    def finish(self):
        if not self._session or len(self._session.views) < 3:
            QMessageBox.warning(self, "Zu wenig Daten",
                                "Es wurden noch nicht genug Keyframes gesammelt.")
            return
        self.stop()
        try:
            result = self._session.finish()
        except Exception as e:
            QMessageBox.critical(self, "Kalibrierung fehlgeschlagen", str(e))
            return
        out = QFileDialog.getExistingDirectory(self, "Export-Verzeichnis wählen")
        if not out:
            return
        ps = (self._pixel_size, self._pixel_size) if self._pixel_size else None
        written = export_all(result, out, self._camera_id,
                             result.used_image_points, result.per_point_errors,
                             pixel_size_mm=ps)
        QMessageBox.information(
            self, "Export abgeschlossen",
            f"fx={result.fx:.2f} fy={result.fy:.2f}\n"
            f"cx={result.cx:.2f} cy={result.cy:.2f}\n"
            f"RMS={result.rms:.3f} px ({result.n_views} Views)\n\n"
            + "\n".join(str(p) for p in written.values()))

    def closeEvent(self, event):
        self.stop()
        super().closeEvent(event)
