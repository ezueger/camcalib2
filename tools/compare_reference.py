#!/usr/bin/env python3
"""Compare our solvers against legacy reference results.

For every legacy project folder (containing ``config.xml`` plus
``<id>-ocv.xml`` or ``<id>-ocam.xml``) this tool:

1. rebuilds the views from the legacy software's OWN detections (native
   resolution, its pixel convention) and runs OUR solver on them,
2. compares pinhole results parameter-by-parameter against the ocv XML,
3. compares fisheye results against the OCam (Scaramuzza) polynomial via
   the radial mapping r(theta) inside the confidence radius - the
   models differ (Kannala-Brandt vs polynomial), so geometry, not
   coefficients, is what must agree.

Usage::

    python tools/compare_reference.py <calib_root>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from camcalib2.calibration import CameraModel, ViewObservation, calibrate
from camcalib2.io.legacy import find_result_xml, load_config_xml


def views_from_config(cfg):
    views = []
    for det in cfg.detections:
        if len(det.points) < 12:
            continue
        ids = sorted(det.points)
        pts = np.array([det.points[i] for i in ids], np.float32)
        views.append(ViewObservation(pts, cfg.object_points_for(ids), marker_ids=ids))
    return views


def compare_pinhole(cfg, ref, views):
    r = calibrate(views, cfg.image_size, CameraModel.PINHOLE)
    print(f"  ours: fx={r.fx:8.2f} fy={r.fy:8.2f} cx={r.cx:8.2f} cy={r.cy:8.2f} "
          f"rms={r.rms:6.3f}  (views={r.n_views}, pts={r.n_points})")
    print(f"  ref : fx={ref.fx:8.2f} fy={ref.fy:8.2f} cx={ref.cx:8.2f} cy={ref.cy:8.2f} "
          f"rms={ref.rms:6.3f}")
    dist_ours = r.dist_coeffs  # k1 k2 p1 p2 k3
    dist_ref = ref.dist_opencv
    print(f"  dist ours: {np.array2string(dist_ours, precision=5, suppress_small=False)}")
    print(f"  dist ref : {np.array2string(dist_ref, precision=5, suppress_small=False)}")
    dfx = 100 * abs(r.fx - ref.fx) / ref.fx
    dfy = 100 * abs(r.fy - ref.fy) / ref.fy
    dc = float(np.hypot(r.cx - ref.cx, r.cy - ref.cy))
    print(f"  ==> dfx={dfx:.3f}%  dfy={dfy:.3f}%  |dc|={dc:.2f}px")
    return {"dfx_pct": dfx, "dfy_pct": dfy, "dc_px": dc, "rms": r.rms, "ref_rms": ref.rms}


def compare_fisheye(cfg, ref, views):
    w, h = cfg.image_size
    r = calibrate(views, cfg.image_size, CameraModel.FISHEYE)
    print(f"  ours (Kannala-Brandt): fx={r.fx:8.2f} fy={r.fy:8.2f} "
          f"cx={r.cx:8.2f} cy={r.cy:8.2f} rms={r.rms:6.3f} "
          f"(views={r.n_views}/{len(views)})")
    print(f"  ref  (OCam):           cx={ref.cx:8.2f} cy={ref.cy:8.2f} "
          f"a0={ref.poly[0]:.1f} rad={ref.rad:.3f}")
    dc = float(np.hypot(r.cx - ref.cx, r.cy - ref.cy))

    # geometric comparison r(theta) within the confidence radius
    half_diag = float(np.hypot(w / 2, h / 2))
    r_conf = ref.rad * half_diag
    theta_max = float(ref.theta(r_conf))
    thetas = np.linspace(0.01, theta_max, 200)
    r_ref = ref.r_of_theta(thetas, r_max=half_diag * 1.5)
    k = r.dist_coeffs
    th_d = thetas * (1 + k[0] * thetas**2 + k[1] * thetas**4
                     + k[2] * thetas**6 + k[3] * thetas**8)
    f_iso = 0.5 * (r.fx + r.fy)
    r_ours = f_iso * th_d
    dr = np.abs(r_ours - r_ref)
    print(f"  ==> |dc|={dc:.2f}px  r(theta) deviation inside confidence radius "
          f"({np.degrees(theta_max):.0f} deg): mean={dr.mean():.2f}px  max={dr.max():.2f}px")
    return {"dc_px": dc, "dr_mean_px": float(dr.mean()),
            "dr_max_px": float(dr.max()), "theta_max_deg": float(np.degrees(theta_max)),
            "rms": r.rms}


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    folders = sorted(p for p in root.rglob("config.xml"))
    summary = []
    for cfgpath in folders:
        folder = cfgpath.parent
        kind, ref = find_result_xml(folder)
        if kind is None:
            continue
        cfg = load_config_xml(cfgpath)
        print(f"\n=== {folder.name}  [{cfg.camera_model} {cfg.camera_serial}, "
              f"{cfg.image_size[0]}x{cfg.image_size[1]}, {cfg.target_type}, {kind}] ===")
        views = views_from_config(cfg)
        try:
            if kind == "ocv":
                res = compare_pinhole(cfg, ref, views)
            else:
                res = compare_fisheye(cfg, ref, views)
            summary.append((folder.name, kind, res))
        except Exception as e:
            print(f"  ERROR: {e}")
            summary.append((folder.name, kind, {"error": str(e)[:100]}))

    print("\n================= SUMMARY =================")
    for name, kind, res in summary:
        if "error" in res:
            print(f"{name[:46]:48s} {kind:4s} ERROR {res['error']}")
        elif kind == "ocv":
            print(f"{name[:46]:48s} ocv  dfx={res['dfx_pct']:.3f}% "
                  f"|dc|={res['dc_px']:5.2f}px rms {res['rms']:.3f} vs ref {res['ref_rms']:.3f}")
        else:
            print(f"{name[:46]:48s} ocam |dc|={res['dc_px']:5.2f}px "
                  f"dr mean={res['dr_mean_px']:5.2f}px max={res['dr_max_px']:5.2f}px "
                  f"(bis {res['theta_max_deg']:.0f} deg), rms KB {res['rms']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
