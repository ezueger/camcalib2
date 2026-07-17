import xml.etree.ElementTree as ET

import numpy as np

from camcalib2.calibration import CalibrationResult, CameraModel
from camcalib2.io import render_result_image, write_opencv_yaml, write_vendor_xml


def make_result():
    K = np.array([[1515.0, 0, 1501.0], [0, 1515.5, 1062.0], [0, 0, 1]])
    dist = np.array([-0.042, 0.097, 0.0004, -0.0004, -0.047])
    return CalibrationResult(
        model=CameraModel.PINHOLE, image_size=(3088, 2064),
        camera_matrix=K, dist_coeffs=dist, rms=0.5,
        per_view_rms=[0.5], n_views=1, n_points=100)


def test_vendor_xml(tmp_path):
    r = make_result()
    p = tmp_path / "cam-ocv.xml"
    write_vendor_xml(r, p, "FBK25040148", pixel_size_mm=(0.0024, 0.0024))
    root = ET.parse(p).getroot()
    assert root.tag == "camera-calibration"
    ocv = root.find("camera-ocv")
    f_xy = [float(v) for v in ocv.findtext("f_xy").split()]
    assert f_xy == [1515.0, 1515.5]
    # unbiased OpenCV principal point (the legacy software's +0.5 offset
    # is a quantization artifact of its integer detections, no convention)
    c_xy = [float(v) for v in ocv.findtext("c_xy").split()]
    assert c_xy == [1501.0, 1062.0]
    # legacy coefficient order: k1 k2 k3 p1 p2
    d = [float(v) for v in ocv.findtext("dist_coeffs").split()]
    assert d == [-0.042, 0.097, -0.047, 0.0004, -0.0004]
    pin = ocv.find("pinhole-camera")
    assert pin.findtext("resolution") == "3088 2064"
    assert abs(float(pin.findtext("f")) - 1515.0 * 0.0024) < 1e-9


def test_vendor_xml_legacy_offset(tmp_path):
    r = make_result()
    p = tmp_path / "cam-ocv.xml"
    write_vendor_xml(r, p, "CAM", legacy_pixel_origin=True)
    ocv = ET.parse(p).getroot().find("camera-ocv")
    c_xy = [float(v) for v in ocv.findtext("c_xy").split()]
    assert c_xy == [1501.5, 1062.5]


def test_opencv_yaml(tmp_path):
    r = make_result()
    p = tmp_path / "cam.yaml"
    write_opencv_yaml(r, p)
    import cv2
    fs = cv2.FileStorage(str(p), cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    assert abs(K[0, 0] - 1515.0) < 1e-9
    fs.release()


def test_result_image():
    r = make_result()
    pts = np.array([[100.0, 100], [3000, 2000], [1500, 1000]])
    errs = np.array([0.5, 1.5, 3.5])
    img = render_result_image(r, [pts], [errs])
    assert img.shape == (2064, 3088, 3)
    assert img.sum() > 0
