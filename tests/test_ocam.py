"""OCam (Scaramuzza) model and solver tests."""

import cv2
import numpy as np
import pytest

from camcalib2.calibration import ViewObservation
from camcalib2.calibration.ocam import OcamModel, calibrate_ocam
from camcalib2.io import write_ocam_xml
from camcalib2.io.legacy import load_ocam_xml


@pytest.fixture(scope="module")
def model():
    # realistic parameters (MER-531-like)
    return OcamModel(image_size=(2592, 2048), cx=1277.3, cy=1055.4,
                     c=1.0001, d=-4e-05, e=-3.7e-05,
                     poly=np.array([558.3, -6.76e-4, 3.08e-7, -4.02e-10]))


def test_project_roundtrip(model):
    rng = np.random.default_rng(1)
    # random rays up to ~85 deg off-axis
    theta = rng.uniform(0.05, np.deg2rad(85), 500)
    phi = rng.uniform(-np.pi, np.pi, 500)
    pts = np.stack([np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)], axis=1) * 1000.0
    px = model.world2cam(pts)
    rays = model.cam2world(px)
    dirs = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((rays * dirs).sum(1), -1, 1)))
    assert np.max(ang) < 0.01  # < 0.01 deg round-trip error


def test_theta_beyond_90deg(model):
    """FOV > 180 deg must be representable (f(rho) goes negative)."""
    rho = model._rho_valid_max() * 0.999
    assert np.degrees(model.theta(rho)) > 95


def test_ocam_solver_synthetic(model):
    rng = np.random.default_rng(5)
    g = np.zeros((100, 3))
    g[:, :2] = np.mgrid[0:10, 0:10].T.reshape(-1, 2) * 100.0
    g[:, :2] -= g[:, :2].mean(0)
    views = []
    w, h = model.image_size
    while len(views) < 14:
        rvec = rng.uniform(-0.7, 0.7, 3)
        tvec = np.array([rng.uniform(-300, 300), rng.uniform(-300, 300),
                         rng.uniform(500, 1500)])
        R, _ = cv2.Rodrigues(rvec)
        pc = g @ R.T + tvec
        if (pc[:, 2] < 50).any():
            continue
        px = model.world2cam(pc)
        inside = ((px[:, 0] > 5) & (px[:, 0] < w - 5)
                  & (px[:, 1] > 5) & (px[:, 1] < h - 5))
        if inside.sum() < 40:
            continue
        noise = rng.normal(0, 0.1, (int(inside.sum()), 2))
        views.append(ViewObservation((px[inside] + noise).astype(np.float32),
                                     g[inside].astype(np.float32)))
    r = calibrate_ocam(views, model.image_size)
    m = r.model
    assert r.rms < 0.25
    assert abs(m.cx - model.cx) < 2.0
    assert abs(m.cy - model.cy) < 2.0
    assert abs(m.poly[0] - model.poly[0]) / model.poly[0] < 0.01
    # geometric agreement of the radial curve
    thetas = np.linspace(0.05, np.deg2rad(80), 100)
    dr = np.abs(m.r_of_theta(thetas) - model.r_of_theta(thetas))
    assert dr.max() < 1.0


def test_ocam_xml_roundtrip(model, tmp_path):
    from camcalib2.calibration.ocam import OcamCalibrationResult
    res = OcamCalibrationResult(model=model, rms=0.5, per_view_rms=[0.5],
                                n_views=1, n_points=100, rvecs=[], tvecs=[])
    p = tmp_path / "CAM-ocam.xml"
    pts = np.array([[100.0, 100.0]])
    write_ocam_xml(res, p, "CAM123", points=pts)
    back = load_ocam_xml(p)
    assert abs(back.cx - model.cx) < 1e-3
    assert abs(back.poly[0] - model.poly[0]) < 1e-2
    assert abs(back.c - model.c) < 1e-6
