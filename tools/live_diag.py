"""Live diagnostic for the adaptive keyframe staffelung / coverage.

Opens the real GenICam camera and runs every frame through the exact same
``CalibrationSession.process()`` pipeline the GUI uses, but instead of a
polished UI it (a) shows a simple preview window so the operator can aim
and move the board, and (b) streams compact per-frame telemetry so the
detection/keyframe behaviour can be watched and analysed:

    * how many points are detected right now,
    * the phase  BOOTSTRAP (no model yet, near-complete board required)
      vs REFINED (model exists, partial views down to the floor allowed),
    * the effective point floor,
    * every keyframe acceptance with its trigger, and every rejection with
      its reason (too_few_points / partial_static / detection_rejected ...),
    * coverage / tilt / #keyframes / rms.

It also writes an annotated snapshot (``live_snapshot.png`` next to this
file) every few seconds - handy for inspecting the coverage veil offline.

The veil shades by cumulative density: a fresh run re-tints everything
lightly and you "rub it free" as you cover cells this run, while globally
sparse cells stay darker so you can target them for another pass.

Keys in the preview window:
    q = quit
    n = new densification run (keep keyframes/coverage, re-open novelty)
    r = full reset (new session from scratch)

Examples::

    python tools/live_diag.py                         # fisheye + 10x10 checker
    python tools/live_diag.py --model pinhole --target dots
    python tools/live_diag.py --serial FBK25040148

This is a developer aid, not part of the shipped app.
"""
from __future__ import annotations

import argparse
import os
import time
from collections import Counter

import cv2
import numpy as np

from camcalib2.calibration import CameraModel
from camcalib2.capture import GenICamSource
from camcalib2.cli import parse_target
from camcalib2.session import CalibrationSession, SessionConfig

SNAPSHOT = os.path.join(os.path.dirname(__file__), "live_snapshot.png")

# plain-language explanation of each accept/reject reason code (ASCII only -
# the OpenCV Hershey font can't render umlauts)
REASON_DE = {
    "no_target": "kein Board erkannt (0 Punkte)",
    "too_few_points": "zu wenige Punkte",
    "too_soon": "zu schnell hintereinander",
    "blurry": "unscharf / zu viel Bewegung",
    "static": "nichts Neues (Zelle schon bedeckt)",
    "partial_static": "Teil-Ansicht ohne neue Flaeche",
    "detection_rejected": "Gitter passt nicht zum Modell (PnP-Gate)",
    "max_keyframes": "Budget voll (nur Verdichtung blockiert)",
    "precise_detect_failed": "Feindetektion fehlgeschlagen",
    "new_coverage": "neue Flaeche",
    "motion": "Bewegung",
    "new_tilt": "neuer Winkel",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default=None)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--model", choices=[m.value for m in CameraModel], default="fisheye")
    ap.add_argument("--target", default="checker:10x10:40")
    ap.add_argument("--preview-width", type=int, default=960)
    args = ap.parse_args()

    log(f"opening camera (index={args.index} serial={args.serial}) ...")
    src = GenICamSource(serial=args.serial, index=args.index)
    src.open()
    w, h = src.image_size
    log(f"camera open: {w}x{h}")

    target = parse_target(args.target)
    cfg = SessionConfig(model=CameraModel(args.model))
    session = CalibrationSession(target, (w, h), cfg)
    pol = cfg.keyframe_policy
    log(f"session: model={args.model} target={args.target} "
        f"roi={tuple(round(v) for v in (session.roi.x, session.roi.y, session.roi.w, session.roi.h))} "
        f"floor bootstrap={pol.min_points_bootstrap} refined={pol.min_points}")

    win = "camcalib2 live-diag"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.preview_width, int(args.preview_width * h / w))

    scale = args.preview_width / w
    reasons = Counter()
    kf_total = kf_partial = 0
    run_no = 1
    last_reason = None
    last_phase = None
    last_hb = 0.0
    last_snap = 0.0
    fail_reads = 0

    try:
        while True:
            ok, frame, ts = src.read()
            if not ok:
                fail_reads += 1
                if fail_reads % 30 == 1:
                    log("no frame from camera ...")
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break
                continue
            fail_reads = 0

            fb = session.process(frame, ts)
            model_exists = fb.result is not None
            floor, full = session._point_floor(model_exists)
            phase = "REFINED" if model_exists else "BOOTSTRAP"
            n = len(fb.ids)

            # --- events -------------------------------------------------
            if phase != last_phase:
                log(f"=== phase -> {phase}  (floor={floor}, full={full})"
                    + (f"  first model rms={fb.rms:.3f}" if fb.rms else ""))
                last_phase = phase

            if fb.keyframe:
                kf_total += 1
                partial = n < full
                if partial:
                    kf_partial += 1
                tag = "PARTIAL" if partial else "full"
                log(f"KEYFRAME #{fb.n_keyframes:2d} [{tag}] pts={n} "
                    f"trigger={fb.reason} cov={fb.coverage*100:.0f}% "
                    f"tilt={fb.tilt_coverage*100:.0f}%"
                    + (f" rms={fb.rms:.3f}" if fb.rms else ""))
            else:
                reasons[fb.reason] += 1
                if fb.reason != last_reason and n > 0:
                    log(f"    reject: {fb.reason:20s} pts={n} phase={phase} "
                        f"floor={floor} cov={fb.coverage*100:.0f}%")
                    last_reason = fb.reason

            now = time.monotonic()
            if now - last_hb > 2.0:
                log(f"hb  pts={n:3d} {phase:9s} floor={floor} "
                    f"cov={fb.coverage*100:4.0f}% tilt={fb.tilt_coverage*100:4.0f}% "
                    f"kf={fb.n_keyframes} (partial={kf_partial})"
                    + (f" rms={fb.rms:.3f}" if fb.rms else "")
                    + f"  state={fb.state.value}  last={fb.reason}")
                last_hb = now

            # --- preview ------------------------------------------------
            vis = cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale),
                               cv2.COLOR_GRAY2BGR)
            # coverage veil: opacity by density - clear where covered this
            # run, darker red the sparser a cell is overall (fisheye: only
            # inside the inscribed image circle)
            alpha = session.coverage.veil_alpha()
            rows, cols = alpha.shape
            red = np.array([40, 40, 200], np.float32)  # BGR
            cw, ch = vis.shape[1] / cols, vis.shape[0] / rows
            for r in range(rows):
                for c in range(cols):
                    a = float(alpha[r, c])
                    if a > 0.01:
                        x0, y0 = int(c * cw), int(r * ch)
                        sub = vis[y0:y0 + int(ch) + 1, x0:x0 + int(cw) + 1]
                        sub[:] = ((1.0 - a) * sub + a * red).astype(np.uint8)
            # detected points: green if this frame is a keyframe, else blue
            col = (80, 255, 80) if fb.keyframe else (255, 200, 90)
            for x, y in fb.points:
                cv2.circle(vis, (int(x * scale), int(y * scale)), 4, col, 1, cv2.LINE_AA)
            hud = [
                f"{phase}  run={run_no}  floor={floor}  pts={n}",
                f"cov={fb.coverage*100:.0f}%  tilt={fb.tilt_coverage*100:.0f}%  "
                f"kf={fb.n_keyframes} (partial={kf_partial})"
                + (f"  rms={fb.rms:.3f}" if fb.rms else ""),
            ]
            for i, line in enumerate(hud):
                y = 22 + i * 22
                cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 1, cv2.LINE_AA)
            # prominent accept/reject banner (bottom): green keyframe / red
            # rejection, always with the reason so it's clear why
            if fb.keyframe:
                banner, bcol = f"KEYFRAME #{fb.n_keyframes}  ({fb.reason})", (80, 255, 80)
            else:
                txt = REASON_DE.get(fb.reason, fb.reason)
                banner, bcol = f"REJECT: {fb.reason} - {txt}", (60, 60, 255)  # BGR red
            by = vis.shape[0] - 16
            cv2.putText(vis, banner, (10, by), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(vis, banner, (10, by), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, bcol, 2, cv2.LINE_AA)
            cv2.imshow(win, vis)
            if now - last_snap > 3.0:
                cv2.imwrite(SNAPSHOT, vis)
                last_snap = now

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("n"):
                session.start_run()
                run_no += 1
                log(f"=== new run #{run_no} (densify) - keyframes so far="
                    f"{fb.n_keyframes}, cov={fb.coverage*100:.0f}% ===")
            if k == ord("r"):
                session = CalibrationSession(target, (w, h), cfg)
                reasons.clear()
                kf_total = kf_partial = 0
                run_no = 1
                last_reason = last_phase = None
                log("=== session reset ===")
    finally:
        cv2.destroyAllWindows()
        src.close()
        log(f"done. keyframes={kf_total} (partial={kf_partial})  "
            f"reject reasons={dict(reasons)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
