# Coded Dot Marker Format ("CCMarker")

Reverse-engineered from the reference calibration data
(`config.xml`: 196 markers with ids and per-image pixel positions across
33 images). The linear model below reproduces **every** reference id
exactly (error 0 over all markers).

**Verified against the original print master**: all 216 markers of
`Marken_500x300.svg` (codes 32..906, boards Codemarker A/B/IFF) decode
correctly with this model, and the id sets of all 8 board definitions
shipped with the original software lie inside the 462-code space.
The original software itself does not decode the markers - it delegates
to a Fraunhofer IFF plugin (`PluginDetectMarkerIFF*.xalg`, pipeline:
Gauss -> Canny -> ellipse fit "Teutsch" -> ring analysis).

## Print geometry (from the SVG master)

* central dot radius R,
* 8 ring dots of radius 0.19 R at ring radius 1.8 R,
* white clearance disc of radius 2.7 R,
* 14 angular slots (25.714 deg pitch).

## Geometry

Each marker consists of:

* one large central dot (the measurement point - its sub-pixel centroid
  is the calibration observation),
* a ring of exactly **8 small dots** at a single radius of ~1.75x the
  central dot radius,
* the ring positions are quantized to **14 angular slots**
  (pitch 360/14 = 25.714 deg).

## Slot semantics (canonical orientation)

| Slot | Role | Weight (if empty) |
|-----:|------|------------------:|
| 0 | data | 0 |
| 1 | **sync - always dot** | - |
| 2 | data | -64 |
| 3 | data | -192 |
| 4 | data | -448 |
| 5 | **quiet zone - always empty** | - |
| 6 | data | +64 |
| 7 | data | +63 |
| 8 | data | +62 |
| 9 | data | +60 |
| 10 | **sync - always dot** | - |
| 11 | data | +56 |
| 12 | data | +48 |
| 13 | data | +32 |

Of the 11 data slots exactly 6 carry a dot and 5 are empty
(constant-weight code: 6 + 2 sync = 8 dots total).

## Id formula

```
id = 704 + sum(weight[slot] for every EMPTY data slot)
```

* Injective over all C(11,5) = 462 possible patterns (ids 32..1009).
* The weights follow a power-of-two structure:
  right side 64 - {0,1,2,4,8,16,32}, left side -(2^k - 1) * 64.

## Rotation handling

The full 462-code space is **not** rotation-unique (3423 rotational
collisions). Boards therefore use a rotation-unique subset: the
reference board's 196 ids produce zero collisions over all 14 rotations
of all codes. The decoder builds a lookup table of all rotations of the
board's marker set and refuses marker sets that are not rotation-unique.

## Board definition

`camcalib2/patterns/resources/vioso_board_196.json` contains the
reference board: marker id -> measured 3D coordinate (mm), including the
measured flatness deviation of the print (z: 0..0.5 mm), which the
calibration uses as non-planar object points.

## Detection notes (validated against reference data)

* Detection rate: 94.8 % of the reference observations over 33 images
  (6.4 MP), plus extra detections the legacy software missed
  (homography-verified as genuine).
* Sub-pixel accuracy: ~0.12-0.15 px (1 sigma) against the reference
  reprojections.
* The legacy software's pixel coordinates are offset by ~(+0.5, +0.5)
  against true sub-pixel centers. Root cause (confirmed in its source,
  PatternDetector.cpp): the IFF detector's sub-pixel ellipse centers are
  stored into `cv::Point2i`, i.e. truncated to integers - the offset is
  a quantization artifact, not a coordinate convention. Its residual
  error (~0.4-0.5 px) is dominated by this quantization; camcalib2 keeps
  full sub-pixel precision and reaches ~0.26 px on the same images.
  `write_vendor_xml(..., legacy_pixel_origin=True)` can mimic the offset.
* The legacy checkerboard path uses `cv::findChessboardCorners` without
  `cornerSubPix` refinement; camcalib2 uses the sub-pixel SB detector.
* The legacy `dist_coeffs` XML order is `k1 k2 k3 p1 p2`
  (OpenCV: `k1 k2 p1 p2 k3`).
* `rad` (confidence radius) = largest distance of an observation with
  reprojection error < 5 px from the principal point, / half diagonal.
