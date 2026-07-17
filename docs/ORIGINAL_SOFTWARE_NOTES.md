# Notes on the original "CameraCalibrator" (calib1) source

Findings from reviewing the original Qt/C++ application and its
`develop/` solvers, and what camcalib2 does differently. All numeric
claims were verified against the 15 reference datasets.

## Detection

* Coded markers are detected by a Fraunhofer IFF plugin
  (`PluginDetectMarkerIFF*.xalg`; pipeline: Gauss -> Canny -> ellipse fit
  "FindEllipseTeutschMod" -> EllipseOptimize -> ring analysis). The
  plugin returns *sub-pixel* centers (see `Marker_out_01.xml`), but
  `PatternDetector.cpp` stores them into `cv::Point2i` - **all marker
  detections are truncated to integer pixels**. With image downscaling
  (out-of-memory fallback) the quantization becomes 2 px.
* Checkerboards use `cv::findChessboardCorners` **without**
  `cornerSubPix` refinement.
* Consequence: the original residuals (~0.4-0.5 px pinhole) are
  dominated by quantization noise. camcalib2 keeps full sub-pixel
  precision (~0.15 px detector noise) and reaches ~0.26-0.33 px.
* Outlier handling: `eliminateWronglyDetectedMarkers` gates at
  image_width/20 (~154 px!) after a first calibration;
  `refineMarkerDetection` re-associates undecoded detections with
  predicted reprojections (a good recovery idea worth porting).

## Pinhole solver (CvCalibrator)

* Identical strategy to camcalib2: pass 1 with z=0, pass 2 with the
  measured 3D board and `CALIB_USE_INTRINSIC_GUESS` - hence our
  bit-identical reproductions on clean datasets.
* No solver-level outlier rejection (camcalib2: 3-sigma point drop +
  re-solve).

## OCam solver (OcamCalibrator)

* Center estimation: iterative 6x6 grid search (shrinking ROI, tolerance
  0.5 px) minimizing the SSRE of the *linear* Scaramuzza solution;
  center is then only touched by a multiplicative scale in the LM.
* **The nonlinear refinement is effectively disabled**: both LM passes
  run with `lm.parameters.maxfev = 3` (the commented-out intended value
  was 2000), inside an alternation loop of max 10 iterations that stops
  on any SSE increase without reverting. The shipped results are
  essentially the linear solution with a grid-searched center.
* This explains why camcalib2's full joint bundle adjustment (all
  intrinsics incl. affine c/d/e + all extrinsics simultaneously) fits
  the same data better on every dataset checked - drastically so on
  decentered units (ME2P-1840 GBF25090143: rms 1.10 vs 1.30).
* The OCam path uses only the 2D board coordinates (the measured
  flatness z is dropped); camcalib2 uses the full 3D points.
* Both solvers require a CodeMeter dongle license at runtime.

## Export (Exporter.cpp)

* `dist_coeffs` order k1 k2 k3 p1 p2; OCam poly a0 a2 a3 a4 (a1=0).
* `c_xy` is the raw OpenCV principal point - there is **no** +0.5
  convention in the export; offsets against camcalib2 stem from the
  integer-truncated input data.
* Quirk: `f_xy` is reconstructed via `cv::calibrationMatrixValues`
  so the exported fx is actually fx^2/fy (negligible for fx~fy; not
  replicated).
* `rad` = `estimateConfidenceRadius(project, 5)`: outermost observation
  with reprojection error < 5 px, distance from the principal point,
  normalized by the half diagonal (camcalib2 matches this).

## Patterns (calib2 plates)

* Print geometry: central dot radius R, 8 ring dots of 0.19 R at ring
  radius 1.8 R, white clearance disc 2.7 R, 14 angular slots.
* All 216 markers of the print master `Marken_500x300.svg` decode
  correctly with camcalib2's reverse-engineered codebook; the 8 board
  definitions (192/196/204/216 markers) are shipped as builtin
  resources and are each rotation-unique.
* Boards can share the same id layout across physically different,
  individually measured prints - identification requires geometric
  disambiguation (implemented via homography residual).
