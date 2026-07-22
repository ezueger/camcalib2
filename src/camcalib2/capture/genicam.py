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
import importlib
import os
import re
import sys
import time

import numpy as np

from .source import FrameSource

_ENV_CTI_VARS = (
    "GENICAM_GENTL64_PATH",
    "GENICAM_GENTL32_PATH",
    "GENICAM_GENTL_PATH",
)
_STATIC_CTI_GLOBS = [
    # Daheng
    "/usr/lib/libgxgentl*.cti",
    "/opt/DahengImaging/**/*.cti",
    "C:/Program Files/Daheng Imaging/GalaxySDK/GenTL/*/*.cti",
    # Basler
    "/opt/pylon/lib/gentlproducer/gtl/*.cti",
    "C:/Program Files/Basler/pylon*/Runtime/x64/*.cti",
]
_DAHENG_GXIPY_SEARCH_ROOTS = [
    "C:/Program Files/Daheng Imaging/GalaxySDK/Samples/Python SDK",
]
_DAHENG_DLL_DIRS = [
    "C:/Program Files/Daheng Imaging/GalaxySDK/APIDll/Win64",
    "C:/Program Files/Daheng Imaging/GalaxySDK/GenTL/Win64",
]
_DISPLAY_NAME_IP_MAC_RE = re.compile(r"\((?P<ip>[0-9.]+)\[(?P<mac>[^\]]+)\]\)")
_GXIPY_MODULE = None
_GXIPY_DLL_HANDLES = []


def _env_cti_globs() -> list[str]:
    globs = []
    for env_var in _ENV_CTI_VARS:
        raw = os.environ.get(env_var, "")
        for path in raw.split(os.pathsep):
            if path:
                globs.append(os.path.join(path, "*.cti"))
    return globs


def _prepare_daheng_sdk_environment() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    if _GXIPY_DLL_HANDLES:
        return
    for dll_dir in _DAHENG_DLL_DIRS:
        if not os.path.isdir(dll_dir):
            continue
        try:
            _GXIPY_DLL_HANDLES.append(os.add_dll_directory(dll_dir))
        except OSError:
            continue


def _load_daheng_gxipy():
    global _GXIPY_MODULE
    if _GXIPY_MODULE is not None:
        return _GXIPY_MODULE

    _prepare_daheng_sdk_environment()
    for root in _DAHENG_GXIPY_SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            _GXIPY_MODULE = importlib.import_module("gxipy")
            return _GXIPY_MODULE
        except Exception:
            continue
    return None


def _camera_identifiers(camera: dict) -> set[str]:
    identifiers = set()
    for key in ("serial_number", "mac_address", "ip_address", "display_name", "device_id"):
        value = camera.get(key)
        if value:
            identifiers.add(f"{key}:{str(value).strip().lower()}")
    return identifiers


def _merge_camera_records(primary: dict, secondary: dict) -> dict:
    merged = dict(primary)
    for key, value in secondary.items():
        if key not in merged or merged.get(key) in ("", None):
            merged[key] = value
    return merged


def _merge_discovered_cameras(primary: list[dict], additions: list[dict]) -> list[dict]:
    merged = list(primary)
    for camera in additions:
        candidate_ids = _camera_identifiers(camera)
        matched_index = None
        for index, existing in enumerate(merged):
            if candidate_ids and candidate_ids.intersection(_camera_identifiers(existing)):
                matched_index = index
                break
        if matched_index is None:
            merged.append(camera)
        else:
            merged[matched_index] = _merge_camera_records(merged[matched_index], camera)
    return merged


def _parse_display_name_ip_mac(display_name: str | None) -> tuple[str | None, str | None]:
    if not display_name:
        return None, None
    match = _DISPLAY_NAME_IP_MAC_RE.search(display_name)
    if not match:
        return None, None
    return match.group("ip"), match.group("mac")


def discover_producers() -> list[str]:
    found = []
    for pattern in [*_env_cti_globs(), *_STATIC_CTI_GLOBS]:
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
                 fps: float | None = None,
                 ip_address: str | None = None,
                 mac_address: str | None = None,
                 backend: str | None = None):
        self.cti_files = cti_files
        self.serial = serial
        self.index = index
        self.pixel_format = pixel_format
        self.reset_to_defaults = reset_to_defaults
        self.exposure_us = exposure_us
        self.gain_db = gain_db
        self.fps = fps
        self.ip_address = ip_address
        self.mac_address = mac_address
        self.backend = backend or "auto"
        self._harvester = None
        self._ia = None
        self._size = None
        self._gx = None
        self._gx_manager = None
        self._gx_device = None

    # ------------------------------------------------------------------
    def open(self) -> None:
        gentl_error = None
        if self.backend != "daheng_gxipy":
            try:
                self._open_with_harvester()
                return
            except Exception as e:
                gentl_error = e
                if not self._should_try_daheng_sdk():
                    raise

        try:
            self._open_with_daheng_sdk()
        except Exception:
            if gentl_error is not None:
                raise gentl_error
            raise

    def _open_with_harvester(self) -> None:
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
            try:
                self._harvester.add_file(cti)
            except Exception:
                # A broken/conflicting producer (e.g. from an unrelated
                # vendor SDK installed on the same machine) must not stop
                # discovery through the other producers.
                continue
        self._harvester.update()
        if not self._harvester.device_info_list:
            raise RuntimeError("no GenICam camera found")

        kwargs = {}
        if self.serial:
            kwargs["search_key"] = {"serial_number": self.serial}
            self._ia = self._harvester.create(kwargs["search_key"])
        else:
            self._ia = self._harvester.create(self.index)
        try:
            # small buffer queue = low latency: the live loop always sees
            # a near-current frame instead of a stale queued one
            self._ia.num_buffers = 2
        except Exception:
            pass

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

    def _should_try_daheng_sdk(self) -> bool:
        if self.backend == "daheng_gxipy":
            return True
        return bool(self.serial or self.ip_address or self.mac_address or _load_daheng_gxipy())

    def _open_with_daheng_sdk(self) -> None:
        gx = _load_daheng_gxipy()
        if gx is None:
            raise RuntimeError(
                "Daheng GalaxySDK Python support not found - install GalaxySDK "
                "with the Python SDK samples.")

        manager = gx.DeviceManager()
        update = getattr(manager, "update_all_device_list", manager.update_device_list)
        device_count, device_info_list = update(500)
        if device_count <= 0 or not device_info_list:
            raise RuntimeError("no Daheng GigE camera found")

        selected = None
        if self.serial:
            selected = next((d for d in device_info_list if d.get("sn") == self.serial), None)
        if selected is None and self.ip_address:
            selected = next((d for d in device_info_list if d.get("ip") == self.ip_address), None)
        if selected is None and self.mac_address:
            selected = next((d for d in device_info_list if d.get("mac") == self.mac_address), None)
        if selected is None:
            if not (0 <= self.index < len(device_info_list)):
                raise RuntimeError("no Daheng GigE camera found")
            selected = device_info_list[self.index]

        if self.serial and selected.get("sn"):
            device = manager.open_device_by_sn(selected["sn"])
        elif self.ip_address and selected.get("ip"):
            device = manager.open_device_by_ip(selected["ip"])
        elif self.mac_address and selected.get("mac"):
            device = manager.open_device_by_mac(selected["mac"])
        else:
            device = manager.open_device_by_index(int(selected["index"]))

        self._gx = gx
        self._gx_manager = manager
        self._gx_device = device

        self._apply_runtime_gx_settings()
        device.stream_on()
        try:
            self._size = (int(device.Width.get()), int(device.Height.get()))
        except Exception:
            self._size = None

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

    @staticmethod
    def _try_gx_set(feature, value) -> bool:
        try:
            feature.set(value)
            return True
        except Exception:
            return False

    @staticmethod
    def _try_gx_command(feature) -> bool:
        try:
            feature.send_command()
            return True
        except Exception:
            return False

    @classmethod
    def _apply_daheng_defaults(cls, device, gx) -> None:
        cls._try_gx_set(device.TriggerMode, gx.GxSwitchEntry.OFF)
        if cls._try_gx_set(device.UserSetSelector, gx.GxUserSetEntry.DEFAULT):
            cls._try_gx_command(device.UserSetLoad)
        cls._try_gx_set(device.PixelFormat, gx.GxPixelFormatEntry.MONO8)
        if cls._try_gx_set(device.ExposureAuto, gx.GxAutoEntry.OFF) and cls._try_gx_set(
            device.GainAuto, gx.GxAutoEntry.OFF
        ):
            pass

    def _apply_runtime_gx_settings(self) -> None:
        if self._gx_device is None or self._gx is None:
            return
        if self.reset_to_defaults:
            self._apply_daheng_defaults(self._gx_device, self._gx)
        else:
            self._try_gx_set(self._gx_device.TriggerMode, self._gx.GxSwitchEntry.OFF)
            self._try_gx_set(self._gx_device.PixelFormat, self._gx.GxPixelFormatEntry.MONO8)
        if self.exposure_us is not None:
            self._try_gx_set(self._gx_device.ExposureAuto, self._gx.GxAutoEntry.OFF)
            self._try_gx_set(self._gx_device.ExposureTime, float(self.exposure_us))
        if self.gain_db is not None:
            self._try_gx_set(self._gx_device.GainAuto, self._gx.GxAutoEntry.OFF)
            self._try_gx_set(self._gx_device.Gain, float(self.gain_db))
        if self.fps is not None:
            self._try_gx_set(self._gx_device.AcquisitionFrameRate, float(self.fps))

    # ------------------------------------------------------------------
    def read(self):
        if self._gx_device is not None:
            try:
                raw_image = self._gx_device.data_stream[0].get_image(3000)
                if raw_image is None:
                    return False, None, 0.0
                data = raw_image.get_numpy_array()
                if data is None:
                    return False, None, 0.0
                frame = np.asarray(data)
                if frame.ndim == 3:
                    frame = np.mean(frame, axis=2).astype(np.uint8)
                return True, frame.copy(), time.monotonic()
            except Exception:
                return False, None, 0.0
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
        if self._gx_device is not None:
            try:
                try:
                    self._gx_device.stream_off()
                except Exception:
                    pass
                self._gx_device.close_device()
            finally:
                self._gx_device = None
                self._gx_manager = None
                self._gx = None
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
    def _camera_from_device_info(device_info, cti_file: str) -> dict:
        display_name = GenICamSource._safe_device_info_value(device_info, "display_name")
        ip_address, mac_address = _parse_display_name_ip_mac(display_name)
        properties = getattr(device_info, "property_dict", {}) or {}
        return {
            "vendor": GenICamSource._safe_device_info_value(device_info, "vendor"),
            "model": GenICamSource._safe_device_info_value(device_info, "model"),
            "serial_number": GenICamSource._safe_device_info_value(device_info, "serial_number"),
            "user_defined_name": GenICamSource._safe_device_info_value(device_info, "user_defined_name"),
            "display_name": display_name,
            "device_id": properties.get("id_"),
            "tl_type": GenICamSource._safe_device_info_value(device_info, "tl_type"),
            "ip_address": ip_address,
            "mac_address": mac_address,
            "access_status": properties.get("access_status"),
            "backend": "gentl",
            "cti_file": cti_file,
        }

    @staticmethod
    def _list_gentl_cameras(cti_files: list[str] | None = None) -> list[dict]:
        """Enumerate reachable cameras through GenTL producers."""
        from harvesters.core import Harvester
        out: list[dict] = []
        for cti in cti_files or discover_producers():
            h = Harvester()
            try:
                h.add_file(cti)
                h.update()
                for device_info in h.device_info_list:
                    out = _merge_discovered_cameras(
                        out,
                        [GenICamSource._camera_from_device_info(device_info, cti)],
                    )
            except Exception:
                # A broken/conflicting producer (e.g. from an unrelated
                # vendor SDK installed on the same machine) must not stop
                # discovery through the other producers.
                continue
            finally:
                h.reset()
        return out

    @staticmethod
    def _list_daheng_sdk_cameras() -> list[dict]:
        gx = _load_daheng_gxipy()
        if gx is None:
            return []

        manager = gx.DeviceManager()
        update = getattr(manager, "update_all_device_list", manager.update_device_list)
        try:
            _count, device_info_list = update(500)
        except Exception:
            return []

        out = []
        for info in device_info_list or []:
            ip_address = info.get("ip") or None
            nic_ip_address = info.get("nic_ip") or None
            out.append({
                "vendor": info.get("vendor_name") or "Daheng Imaging",
                "model": info.get("model_name") or "",
                "serial_number": info.get("sn") or "",
                "user_defined_name": info.get("user_id") or "",
                "display_name": info.get("display_name") or "",
                "device_id": info.get("device_id") or "",
                "tl_type": "GEV",
                "ip_address": ip_address,
                "mac_address": info.get("mac") or "",
                "subnet_mask": info.get("subnet_mask") or "",
                "gateway": info.get("gateway") or "",
                "nic_ip_address": nic_ip_address,
                "nic_subnet_mask": info.get("nic_subnet_mask") or "",
                "nic_gateway": info.get("nic_gateWay") or "",
                "nic_description": info.get("nic_description") or "",
                "access_status": info.get("access_status"),
                "backend": "daheng_gxipy",
            })
        return out

    @staticmethod
    def list_cameras(cti_files: list[str] | None = None) -> list[dict]:
        """Enumerate reachable cameras (vendor, model, serial, network info)."""
        cameras = GenICamSource._list_gentl_cameras(cti_files)
        return _merge_discovered_cameras(cameras, GenICamSource._list_daheng_sdk_cameras())
