from camcalib2.capture.genicam import GenICamSource


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
