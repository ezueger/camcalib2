"""Generic GenICam/GigE camera source via GenTL producers (harvesters).

Works with any vendor that ships a GenTL producer (`.cti` file), tested
targets: Daheng Galaxy and Basler pylon.  Typical producer locations:

* Daheng:  ``C:/Program Files/Daheng Imaging/GalaxySDK/GenTL/...`` or
  ``/usr/lib/libgxgentl*.cti``
* Basler:  ``C:/Program Files/Basler/pylon */Runtime/x64/ProducerGEV.cti``
  or ``/opt/pylon/lib/gentlproducer/gtl/ProducerGEV.cti``

The ``GENICAM_GENTL64_PATH`` environment variable is also honoured
(standard GenTL discovery).
"""

from __future__ import annotations

import glob
import os
import time

import numpy as np

from .source import FrameSource

_DEFAULT_CTI_GLOBS = [
    # environment (colon/semicolon separated directories)
    *(os.path.join(p, "*.cti")
      for p in os.environ.get("GENICAM_GENTL64_PATH", "").replace(";", ":").split(":") if p),
    # Daheng
    "/usr/lib/libgxgentl*.cti",
    "/opt/DahengImaging/**/*.cti",
    "C:/Program Files/Daheng Imaging/GalaxySDK/GenTL/*/*.cti",
    # Basler
    "/opt/pylon/lib/gentlproducer/gtl/*.cti",
    "C:/Program Files/Basler/pylon*/Runtime/x64/*.cti",
]


def discover_producers() -> list[str]:
    found = []
    for pattern in _DEFAULT_CTI_GLOBS:
        found.extend(glob.glob(pattern, recursive=True))
    return sorted(set(found))


class GenICamSource(FrameSource):
    """Frame source for any GenICam camera reachable through a GenTL producer."""

    def __init__(self, cti_files: list[str] | None = None,
                 serial: str | None = None, index: int = 0,
                 pixel_format: str = "Mono8",
                 reset_to_defaults: bool = False,
                 exposure_us: float | None = None,
                 gain_db: float | None = None,
                 fps: float | None = None):
        self.cti_files = cti_files
        self.serial = serial
        self.index = index
        self.pixel_format = pixel_format
        self.reset_to_defaults = reset_to_defaults
        self.exposure_us = exposure_us
        self.gain_db = gain_db
        self.fps = fps
        self._harvester = None
        self._ia = None
        self._size = None

    # ------------------------------------------------------------------
    def open(self) -> None:
        try:
            from harvesters.core import Harvester
        except ImportError as e:
            raise RuntimeError(
                "GenICam support requires the 'harvesters' package "
                "(pip install camcalib2[genicam])") from e

        ctis = self.cti_files or discover_producers()
        if not ctis:
            raise RuntimeError(
                "no GenTL producer (.cti) found - install the Daheng Galaxy "
                "or Basler pylon SDK, or set GENICAM_GENTL64_PATH")

        self._harvester = Harvester()
        for cti in ctis:
            self._harvester.add_file(cti)
        self._harvester.update()
        if not self._harvester.device_info_list:
            raise RuntimeError("no GenICam camera found")

        kwargs = {}
        if self.serial:
            kwargs["search_key"] = {"serial_number": self.serial}
            self._ia = self._harvester.create(kwargs["search_key"])
        else:
            self._ia = self._harvester.create(self.index)

        nm = self._ia.remote_device.node_map
        if self.reset_to_defaults:
            self._load_default_user_set(nm)
        # Keep the runtime stream predictable for the app even after a reset.
        self._try_set(nm, "PixelFormat", self.pixel_format)
        if self.exposure_us is not None:
            self._try_set(nm, "ExposureAuto", "Off")
            self._try_set(nm, "ExposureTime", float(self.exposure_us))
        if self.gain_db is not None:
            self._try_set(nm, "GainAuto", "Off")
            self._try_set(nm, "Gain", float(self.gain_db))
        if self.fps is not None:
            self._try_set(nm, "AcquisitionFrameRateEnable", True)
            self._try_set(nm, "AcquisitionFrameRate", float(self.fps))
        try:
            self._size = (int(nm.Width.value), int(nm.Height.value))
        except Exception:
            self._size = None
        self._ia.start()

    @staticmethod
    def _try_set(node_map, name: str, value) -> bool:
        try:
            node = getattr(node_map, name)
            node.value = value
            return True
        except Exception:
            return False

    @staticmethod
    def _try_command(node_map, name: str) -> bool:
        try:
            getattr(node_map, name).execute()
            return True
        except Exception:
            return False

    @staticmethod
    def _default_user_set_candidates(default_selector: str | None) -> list[str]:
        candidates = []
        if default_selector:
            candidates.append(default_selector)
        for candidate in ("Default", "Factory"):
            if candidate not in candidates:
                candidates.append(candidate)
        return candidates

    @classmethod
    def _load_default_user_set(cls, node_map) -> bool:
        default_selector = None
        try:
            default_selector = str(getattr(node_map, "UserSetDefaultSelector").value)
        except Exception:
            default_selector = None
        for selector in cls._default_user_set_candidates(default_selector):
            if not cls._try_set(node_map, "UserSetSelector", selector):
                continue
            if cls._try_command(node_map, "UserSetLoad"):
                return True
        return False

    @staticmethod
    def _safe_device_info_value(device_info, attribute: str, default: str = "") -> str:
        try:
            value = getattr(device_info, attribute)
        except Exception:
            return default
        return default if value is None else value

    # ------------------------------------------------------------------
    def read(self):
        if self._ia is None:
            return False, None, 0.0
        try:
            with self._ia.fetch(timeout=3.0) as buffer:
                comp = buffer.payload.components[0]
                w, h = comp.width, comp.height
                data = np.asarray(comp.data, dtype=np.uint8)[: w * h].reshape(h, w).copy()
            return True, data, time.monotonic()
        except Exception:
            return False, None, 0.0

    def close(self) -> None:
        if self._ia is not None:
            try:
                self._ia.stop()
                self._ia.destroy()
            finally:
                self._ia = None
        if self._harvester is not None:
            self._harvester.reset()
            self._harvester = None

    @property
    def image_size(self):
        return self._size

    @staticmethod
    def list_cameras(cti_files: list[str] | None = None) -> list[dict]:
        """Enumerate reachable cameras (vendor, model, serial)."""
        from harvesters.core import Harvester
        h = Harvester()
        for cti in cti_files or discover_producers():
            h.add_file(cti)
        h.update()
        out = []
        for d in h.device_info_list:
            out.append({
                "vendor": GenICamSource._safe_device_info_value(d, "vendor"),
                "model": GenICamSource._safe_device_info_value(d, "model"),
                "serial_number": GenICamSource._safe_device_info_value(d, "serial_number"),
                "user_defined_name": GenICamSource._safe_device_info_value(d, "user_defined_name"),
                "display_name": GenICamSource._safe_device_info_value(d, "display_name"),
                "tl_type": GenICamSource._safe_device_info_value(d, "tl_type"),
            })
        h.reset()
        return out
