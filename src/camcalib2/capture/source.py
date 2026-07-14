"""Frame sources: GenICam cameras, video files, image folders."""

from __future__ import annotations

import glob
import os
import time
from abc import ABC, abstractmethod

import cv2
import numpy as np


class FrameSource(ABC):
    """A source of grayscale/BGR frames with timestamps."""

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None, float]:
        """Returns (ok, frame, timestamp_seconds)."""

    @abstractmethod
    def close(self) -> None: ...

    @property
    def image_size(self) -> tuple[int, int] | None:
        """(width, height) if known."""
        return None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()


class ImageFolderSource(FrameSource):
    """Plays back a folder of images - simulation source for development
    and for re-running recorded sessions."""

    def __init__(self, folder: str, pattern: str = "*.jpg", fps: float | None = None,
                 loop: bool = False):
        self.folder = folder
        self.pattern = pattern
        self.fps = fps
        self.loop = loop
        self._files: list[str] = []
        self._idx = 0
        self._size = None

    def open(self) -> None:
        self._files = sorted(
            f for f in glob.glob(os.path.join(self.folder, self.pattern))
            if os.path.basename(os.path.dirname(f)) != "Results")
        if not self._files:
            raise FileNotFoundError(f"no images matching {self.pattern} in {self.folder}")
        first = cv2.imread(self._files[0], cv2.IMREAD_GRAYSCALE)
        self._size = (first.shape[1], first.shape[0])
        self._idx = 0

    def read(self):
        if self._idx >= len(self._files):
            if not self.loop:
                return False, None, 0.0
            self._idx = 0
        frame = cv2.imread(self._files[self._idx], cv2.IMREAD_GRAYSCALE)
        ts = self._idx / self.fps if self.fps else float(self._idx)
        self._idx += 1
        if self.fps:
            time.sleep(1.0 / self.fps)
        return True, frame, ts

    def close(self) -> None:
        self._files = []

    @property
    def image_size(self):
        return self._size


class VideoFileSource(FrameSource):
    def __init__(self, path: str, realtime: bool = False):
        self.path = path
        self.realtime = realtime
        self._cap: cv2.VideoCapture | None = None
        self._t0 = 0.0

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise IOError(f"cannot open video {self.path}")
        self._t0 = time.monotonic()

    def read(self):
        ok, frame = self._cap.read()
        if not ok:
            return False, None, 0.0
        ts = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if self.realtime:
            lag = ts - (time.monotonic() - self._t0)
            if lag > 0:
                time.sleep(lag)
        return True, frame, ts

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def image_size(self):
        if self._cap is None:
            return None
        return (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
