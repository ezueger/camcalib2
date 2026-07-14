import pytest

from camcalib2.patterns.dotcode import (ALL_IDS, N_RING_DOTS, N_SLOTS,
                                        Codebook, encode)
from camcalib2.patterns.board import MarkerBoard


def test_codespace_size():
    assert len(ALL_IDS) == 462  # C(11,5) injective codes


def test_reference_id_range():
    assert min(ALL_IDS) == 32
    assert max(ALL_IDS) == 1009


def test_encode_constant_weight():
    for marker_id in ALL_IDS:
        bits = encode(marker_id)
        assert len(bits) == N_SLOTS
        assert sum(bits) == N_RING_DOTS
        assert bits[1] == 1 and bits[10] == 1  # sync dots
        assert bits[5] == 0  # quiet slot


def test_encode_reference_examples():
    # spot checks against reverse-engineered reference data
    assert encode(32) == (0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0)
    assert encode(63) == (0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1)
    assert encode(127) == (1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1)
    assert encode(683) == (0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1)


def test_invalid_id_raises():
    with pytest.raises(ValueError):
        encode(31)


def test_builtin_board_codebook_rotation_invariant():
    board = MarkerBoard.builtin()
    assert len(board.markers) == 196
    cb = board.codebook
    for marker_id in board.markers:
        bits = encode(marker_id)
        for rot in range(N_SLOTS):
            rotated = tuple(bits[(i + rot) % N_SLOTS] for i in range(N_SLOTS))
            decoded = cb.decode(rotated)
            assert decoded is not None
            assert decoded[0] == marker_id


def test_full_codespace_is_ambiguous():
    # the complete 462-code space is NOT rotation-unique; the codebook
    # must refuse it
    with pytest.raises(ValueError):
        Codebook(ALL_IDS)


def test_decode_unknown_pattern():
    board = MarkerBoard.builtin()
    assert board.codebook.decode((0,) * N_SLOTS) is None
