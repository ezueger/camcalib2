import sys
import types

import pytest

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


class _MultiDeviceInfo:
    def __init__(self, model, serial, display_name="", device_id=""):
        self.vendor = "Daheng Imaging"
        self.model = model
        self.serial_number = serial
        self.user_defined_name = ""
        self.display_name = display_name
        self.tl_type = "GEV"
        self.property_dict = {"id_": device_id}


_MULTI_DEVICES = [
    _MultiDeviceInfo("MER2-503-23GM-P", "FBF24070661",
                     "MER2-503-23GM-P(10.0.0.8[00-21-49-06-39-D9])", "dev-a"),
    _MultiDeviceInfo("MER2-1220-32U3M", "FBK25040148",
                     "MER2-1220-32U3M(10.0.0.9[00-21-49-06-39-DA])", "dev-b"),
    _MultiDeviceInfo("MER2-1220-32U3M", "FBK25040149",
                     "MER2-1220-32U3M(10.0.0.10[00-21-49-06-39-DB])", "dev-c"),
]


class _NodeMap:
    class _Node:
        value = 0

    Width = _Node()
    Height = _Node()


class _FakeAcquirer:
    def __init__(self, device_info):
        self.device_info = device_info
        self.remote_device = types.SimpleNamespace(node_map=_NodeMap())
        self.started = False

    def start(self):
        self.started = True


class _MultiHarvester:
    """Mimics harvesters: refuses to choose when the key is ambiguous."""

    def __init__(self):
        self.device_info_list = []

    def add_file(self, cti):
        pass

    def update(self):
        self.device_info_list = list(_MULTI_DEVICES)

    def create(self, search_key=None):
        if search_key is None:
            raise ValueError("multiple devices found: provide sufficient search key")
        if isinstance(search_key, _MultiDeviceInfo):
            return _FakeAcquirer(search_key)
        if isinstance(search_key, dict):
            hits = [d for d in _MULTI_DEVICES
                    if all(d.property_dict.get(k) == v for k, v in search_key.items())]
            if len(hits) != 1:
                raise ValueError("multiple devices found: provide sufficient search key")
            return _FakeAcquirer(hits[0])
        return _FakeAcquirer(_MULTI_DEVICES[search_key])

    def reset(self):
        pass


def _install_multi_harvester(monkeypatch):
    fake_core = types.SimpleNamespace(Harvester=_MultiHarvester)
    fake_harvesters = types.ModuleType("harvesters")
    fake_harvesters.core = fake_core
    monkeypatch.setitem(sys.modules, "harvesters", fake_harvesters)
    monkeypatch.setitem(sys.modules, "harvesters.core", fake_core)


def _open_source(monkeypatch, **kwargs):
    _install_multi_harvester(monkeypatch)
    source = GenICamSource(cti_files=["fake.cti"], **kwargs)
    source.open()
    return source._ia.device_info


def test_open_without_search_key_uses_index_instead_of_failing(monkeypatch):
    # regression: harvesters used to abort with "multiple devices found"
    assert _open_source(monkeypatch).serial_number == "FBF24070661"
    assert _open_source(monkeypatch, index=2).serial_number == "FBK25040149"


def test_open_selects_camera_by_serial_ip_and_mac(monkeypatch):
    assert _open_source(monkeypatch, serial="FBK25040148").serial_number == "FBK25040148"
    assert _open_source(monkeypatch, ip_address="10.0.0.10").serial_number == "FBK25040149"
    assert _open_source(
        monkeypatch, mac_address="00-21-49-06-39-d9").serial_number == "FBF24070661"


def test_open_reports_available_cameras_when_selection_misses(monkeypatch):
    for kwargs in ({"serial": "nope"}, {"ip_address": "10.9.9.9"}, {"index": 9}):
        with pytest.raises(RuntimeError) as excinfo:
            _open_source(monkeypatch, **kwargs)
        message = str(excinfo.value)
        assert "available" in message
        assert "FBK25040148" in message
        assert "multiple devices found" not in message
