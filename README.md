# camcalib2

Video-basierte Bestimmung von Kameraintrinsiken für Industriekameras
(Daheng, Basler, generisch über GigE/GenICam) mit perspektivischen und
Fisheye-Objektiven.

Statt Einzelbilder aufzunehmen wird die Kamera im Video-Modus um das
Testpattern bewegt. Marker werden in Echtzeit detektiert, der Sensor
wird Zelle für Zelle "abgetastet" und die Kalibrierung läuft inkrementell
im Hintergrund mit — mit Live-Feedback wie beim Einrichten von FaceTime.

## Features

* **Coded-Dot-Marker-Boards** (Format der Bestandssoftware, vollständig
  reverse-engineered und gegen die Original-Druckvorlage verifiziert,
  siehe [docs/MARKER_CODE.md](docs/MARKER_CODE.md)):
  rotationsinvariante Dekodierung, Subpixel-Zentren (~0.15 px),
  nutzt die vermessenen 3D-Koordinaten inkl. Board-Unebenheit.
  Alle 8 Board-Definitionen der Bestandssoftware sind eingebaut
  (`--target dots:<name>`), inklusive automatischer Board-Erkennung
  (`--target dots:auto`, unterscheidet auch baugleiche Drucke anhand
  ihrer Vermessung).
* **Checkerboard** für Fisheye: Teilgitter-Erkennung
  (`findChessboardCornersSB` + Meta), Board darf den Bildkreis verlassen.
* **Kalibriermodelle**: OpenCV Pinhole (5 Koeffizienten) und
  Kannala-Brandt Fisheye (`cv2.fisheye`), jeweils mit robuster
  Ausreißerbehandlung (vergiftete Views werden erkannt und entfernt).
* **Live-Session**: Keyframe-Auswahl (Schärfe/Bewegung/Neuabdeckung),
  Coverage-Grid + Winkel-Histogramm, inkrementelle Rekalibrierung im
  Hintergrund, Konvergenzerkennung.
* **Qt-UI** (PySide6): Live-Bild mit Coverage-Schleier, Fortschrittsring,
  Hinweistexten und Live-Intrinsiken.
* **Export**: Legacy-XML (`<camera-calibration>`, kompatibel zur
  Bestandssoftware inkl. deren Konventionen), OpenCV-YAML und
  `result.jpg`-Fehlerkarte im gewohnten Stil.
* **Kameras**: generisch über GenTL-Producer (harvesters) — Daheng Galaxy
  und Basler pylon werden automatisch gefunden; außerdem Videodatei und
  Bilderordner als Quellen (Simulation/Batch).

## Installation

```bash
pip install -e .          # Kern (numpy, opencv)
pip install -e .[ui]      # + PySide6 GUI
pip install -e .[genicam] # + GenICam/GigE Kameras (harvesters)
```

Unter Windows legt eine Paketinstallation zusaetzlich einen nativen
GUI-Startpunkt `camcalib2-gui` an.

## Windows-Paket fuer anderen Rechner

Fuer eine portable GUI-Verteilung ohne lokale Python-Installation:

```powershell
.\tools\build_windows_gui.ps1
```

Falls PowerShell lokale Skripte blockiert:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\build_windows_gui.ps1
```

Danach liegen die Dateien hier:

* `dist/camcalib2-gui/` mit `camcalib2-gui.exe`
* `release/camcalib2-gui-windows.zip` zum Kopieren auf den Zielrechner

Optional:

```powershell
.\tools\build_windows_gui.ps1 -OneFile
.\tools\build_windows_gui.ps1 -IncludeGenICam
```

Hinweise:

* `-OneFile` erzeugt eine einzelne EXE statt eines Verzeichnisses.
* `-IncludeGenICam` packt die optionale `harvesters`-Unterstuetzung mit ein.
* Fuer echte GenICam-Kameras werden auf dem Zielrechner weiterhin die
  passenden Hersteller-SDKs bzw. `.cti`-Producer benoetigt.

## Benutzung

Live mit Kamera (GUI):

```bash
camcalib2 --camera --serial FBK25040148 --target dots --model pinhole \
          --camera-id FBK25040148 --pixel-size 0.0024
```

Simulation aus Bilderordner:

```bash
camcalib2 --images ./aufnahmen --target dots --model pinhole
```

Batch (klassischer Einzelbild-Workflow, ohne GUI):

```bash
camcalib2-calibrate --images ./aufnahmen --target dots --model pinhole \
    --camera-id FBK25040148 --pixel-size 0.0024 --all-frames --out ./Results

camcalib2-calibrate --images ./fisheye_aufnahmen --target checker:12x12:40 \
    --model fisheye --camera-id FBF25110670 --all-frames --out ./Results
```

Targets: `dots` (eingebautes 196-Marker-Board), `dots:<board.json>`
(eigenes Board), `checker:<cols>x<rows>:<square_mm>`.

## Validierung gegen Referenzdaten

Perspektivkamera MER2-630 (33 Bilder, Coded-Dot-Board):

| | Bestandssoftware | camcalib2 |
|---|---|---|
| fx / fy | 1515.68 / 1515.54 | 1514.67 / 1514.83 |
| cx / cy | 1501.82 / 1062.08 | 1502.41 / 1064.59 (Legacy-Konvention) |
| RMS | 0.507 px | **0.276 px** |

Fisheye MER2-503 (26 Bilder, Checkerboard): RMS **0.335 px**,
fx/fy 405.9/406.1, Bildkreis korrekt erkannt.

## Architektur

```
src/camcalib2/
├── patterns/    # Marker-Codierung, Board-Definitionen (+ JSON-Ressource)
├── detection/   # Dot-Marker-Detektor, Checkerboard-Detektor
├── calibration/ # Pinhole-/Fisheye-Solver mit Robustheit
├── session/     # Coverage, Keyframes, Live-Controller
├── capture/     # FrameSource: GenICam, Video, Bilderordner
├── io/          # Export: Legacy-XML, OpenCV-YAML, result.jpg
├── ui/          # PySide6 GUI
└── cli.py       # Batch-Kalibrierung
```

## Tests

```bash
pip install -e .[dev]
pytest
```
