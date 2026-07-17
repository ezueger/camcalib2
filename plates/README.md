# Kalibrier-Testpatterns (Originalbestand)

## definitions/

Board-Definitionen der Bestandssoftware (Format: `id x_mm y_mm z_mm`
pro Zeile, individuell vermessene Drucke inkl. Ebenheitsabweichung in z):

| Datei | Marker | Codebereich |
|---|---|---|
| Kalibriertafel_Codemarker_A/B/IFF | 216 | 32..906 |
| Kalibrierfeld_2022-01/02 | 204 | 32..858 |
| Kalibriertafel_SN2025-001/002 | 196 | 32..683 |
| CalTafel_Erik_Marker | 192 | 32..678 |

Diese Dateien sind als JSON-Ressourcen in `camcalib2.patterns` eingebaut
(`MarkerBoard.builtin_names()`), laden per
`MarkerBoard.from_txt(pfad)` oder `--target dots:<name>`.

**Achtung:** Boards mit identischem Marker-Layout (z. B. SN2025-001 vs.
-002) sind verschiedene physische Drucke mit eigener Vermessung - die
automatische Board-Erkennung (`dots:auto`) unterscheidet sie anhand der
Geometrie.

## print/

Original-Druckvorlagen (PDF):

* `Marken_500x300.pdf` - Codemarker-Tafel 500x300 mm
* `codemarker 1200mm x 1450mm.pdf` - grosse Codemarker-Tafel
* `Marker_layout_kleine_Tafel_final.pdf` - kleine Codemarker-Tafel
* `checkerboard A0 board 15x22.pdf`, `checkerboard 1200mm x 14500mm 22x27.pdf`,
  `10 x 8.pdf` - Checkerboard-Vorlagen

Marker-Druckgeometrie (verifiziert gegen die SVG-Master): Zentrumspunkt
Radius R, 8 Ringpunkte mit Radius 0.19R auf Ringradius 1.8R, weisse
Freistellscheibe 2.7R, 14 Winkel-Slots. Codierung siehe
`docs/MARKER_CODE.md`.
