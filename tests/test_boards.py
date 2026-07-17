"""All shipped board definitions must load and be rotation-unique.

The board resources were converted from the original software's
CalibrationPatterns/*.txt; the print master (Marken_500x300.svg) was
decoded 216/216 with this codebook, verifying the encoding end-to-end.
"""

import pytest

from camcalib2.patterns.board import MarkerBoard

EXPECTED = {
    "caltafel_erik_marker": 192,
    "kalibrierfeld_2022_01": 204,
    "kalibrierfeld_2022_02": 204,
    "kalibriertafel_codemarker_a": 216,
    "kalibriertafel_codemarker_b": 216,
    "kalibriertafel_codemarker_iff": 216,
    "kalibriertafel_sn2025_001": 196,
    "kalibriertafel_sn2025_002": 196,
    "vioso_board_196": 196,
}


def test_builtin_names():
    names = MarkerBoard.builtin_names()
    assert set(EXPECTED) <= set(names)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_builtin_board_loads_rotation_unique(name):
    board = MarkerBoard.builtin(name)  # Codebook() raises if ambiguous
    assert len(board.markers) == EXPECTED[name]
    obj = board.object_points(sorted(board.markers)[:5])
    assert obj.shape == (5, 3)
