import sys
import types

from camcalib2.capture.genicam import GenICamSource
from camcalib2.capture.genicam import _merge_discovered_cameras


def test_default_user_set_candidates_prefers_camera_default():
    assert GenICamSource._default_user_set_candidates("UserSet2") == [
        "UserSet2",
        "Default",
        "Factory",
    ]


def test_default_user_set_candidates_falls_back_to_generic_defaults():
    assert GenICamSource._default_user_set_candidates(None) == [
        "Default",
        "Factory",
    ]


class _UnsupportedInfo:
    @property
    def serial_number(self):
        raise RuntimeError("not implemented")


def test_safe_device_info_value_returns_default_for_unsupported_fields():
    assert GenICamSource._safe_device_info_value(_UnsupportedInfo(), "serial_number") == ""


def test_merge_discovered_cameras_deduplicates_by_mac_and_ip():
    merged = _merge_discovered_cameras(
        [{
            "vendor": "Daheng Imaging",
            "model": "MER2-503-23GM-P",
            "display_name": "MER2-503-23GM-P(10.0.0.8[00-21-49-06-39-D9])",
            "ip_address": "10.0.0.8",
            "mac_address": "00-21-49-06-39-D9",
            "backend": "gentl",
        }],
        [{
            "vendor": "Daheng Imaging",
            "model": "MER2-503-23GM-P",
            "serial_number": "FBF24070661",
            "ip_address": "10.0.0.8",
            "mac_address": "00-21-49-06-39-D9",
            "backend": "daheng_gxipy",
        }],
    )
    assert len(merged) == 1
    assert merged[0]["serial_number"] == "FBF24070661"
    assert merged[0]["backend"] == "gentl"


class _FakeDeviceInfo:
    vendor = "Daheng Imaging"
    model = "MER2-503-23GM-P"
    serial_number = "FBF24070661"
    user_defined_name = ""
    display_name = "MER2-503-23GM-P(10.0.0.8[00-21-49-06-39-D9])"
    tl_type = "GEV"
    property_dict = {}


class _FakeHarvester:
    def __init__(self):
        self.device_info_list = []

    def add_file(self, cti):
        if "broken" in cti:
            raise RuntimeError("the target port does not hold any URL")

    def update(self):
        self.device_info_list = [_FakeDeviceInfo()]

    def reset(self):
        pass


def test_list_gentl_cameras_skips_broken_producer_without_losing_others(monkeypatch):
    fake_core = types.SimpleNamespace(Harvester=_FakeHarvester)
    fake_harvesters = types.ModuleType("harvesters")
    fake_harvesters.core = fake_core
    monkeypatch.setitem(sys.modules, "harvesters", fake_harvesters)
    monkeypatch.setitem(sys.modules, "harvesters.core", fake_core)

    cameras = GenICamSource._list_gentl_cameras(["broken_vendor.cti", "good_vendor.cti"])

    assert len(cameras) == 1
    assert cameras[0]["serial_number"] == "FBF24070661"
