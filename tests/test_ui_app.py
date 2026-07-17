from camcalib2.ui.app import _format_camera_label


def test_format_camera_label_includes_relevant_fields():
    label = _format_camera_label({
        "vendor": "Basler",
        "model": "acA1920-40gm",
        "serial_number": "12345",
        "user_defined_name": "LineCam",
        "tl_type": "GEV",
    }, 1)
    assert label == '[2] Basler | acA1920-40gm | SN 12345 | "LineCam" | GEV'
