"""Encoding/decoding of the circular coded dot markers ("CCMarker").

Each marker consists of a large central dot and a ring of 8 small dots
placed on 14 equally spaced angular slots (25.714 deg pitch) at a single
ring radius (~1.75x the central dot radius).

Slot layout (canonical orientation):

* slot 1 and slot 10 always carry a dot (sync dots),
* slot 5 is always empty (quiet zone),
* the remaining 11 slots are data slots of which exactly 6 carry a dot
  (constant-weight code: 8 dots total, 6 empty slots total).

The marker id is a linear function of the *empty* data slots:

    id = 704 + sum(WEIGHTS[slot] for empty data slot)

with the weight table below.  This encoding was reverse-engineered from
the reference calibration data (config.xml: 196 markers, ids 32..683)
and reproduces every reference id exactly; it is injective over all
C(11,5) = 462 possible patterns.

The full 462-code space is not rotation-unique, but marker sets actually
printed on boards are chosen rotation-unique; the codebook therefore is
built per marker-id set and validated for rotational uniqueness.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

N_SLOTS = 14
SLOT_PITCH_DEG = 360.0 / N_SLOTS
SYNC_DOT_SLOTS = (1, 10)  # always occupied
QUIET_SLOT = 5  # never occupied
BASE = 704

#: weight of each *empty* data slot in the id sum
WEIGHTS = {
    0: 0,
    2: -64,
    3: -192,
    4: -448,
    6: 64,
    7: 63,
    8: 62,
    9: 60,
    11: 56,
    12: 48,
    13: 32,
}

DATA_SLOTS = tuple(sorted(WEIGHTS))
N_EMPTY_DATA_SLOTS = 5
N_RING_DOTS = 8  # 2 sync + 6 occupied data slots


def _pattern_from_empty(empty_slots: Iterable[int]) -> tuple[int, ...]:
    """14-slot occupancy bits from the set of empty data slots."""
    empty = set(empty_slots)
    bits = []
    for s in range(N_SLOTS):
        if s in SYNC_DOT_SLOTS:
            bits.append(1)
        elif s == QUIET_SLOT or s in empty:
            bits.append(0)
        else:
            bits.append(1)
    return tuple(bits)


def _build_id_table() -> dict[int, tuple[int, ...]]:
    table: dict[int, tuple[int, ...]] = {}
    for empty in combinations(DATA_SLOTS, N_EMPTY_DATA_SLOTS):
        marker_id = BASE + sum(WEIGHTS[s] for s in empty)
        table[marker_id] = _pattern_from_empty(empty)
    return table


#: marker id -> canonical 14-bit occupancy pattern (all 462 valid codes)
ID_TO_PATTERN: dict[int, tuple[int, ...]] = _build_id_table()

ALL_IDS = frozenset(ID_TO_PATTERN)


def encode(marker_id: int) -> tuple[int, ...]:
    """Canonical 14-slot occupancy pattern for a marker id."""
    try:
        return ID_TO_PATTERN[marker_id]
    except KeyError:
        raise ValueError(f"{marker_id} is not a valid marker id") from None


class Codebook:
    """Rotation-invariant decoder for a specific set of marker ids.

    The board's marker ids must form a rotation-unique subset of the
    code space (checked on construction).
    """

    def __init__(self, marker_ids: Iterable[int]):
        self.marker_ids = frozenset(marker_ids)
        unknown = self.marker_ids - ALL_IDS
        if unknown:
            raise ValueError(f"invalid marker ids: {sorted(unknown)}")
        self._lut: dict[tuple[int, ...], tuple[int, int]] = {}
        for marker_id in self.marker_ids:
            bits = ID_TO_PATTERN[marker_id]
            for rot in range(N_SLOTS):
                key = tuple(bits[(i + rot) % N_SLOTS] for i in range(N_SLOTS))
                prev = self._lut.get(key)
                if prev is not None and prev[0] != marker_id:
                    raise ValueError(
                        "marker set is not rotation-unique: "
                        f"ids {prev[0]} and {marker_id} share a rotated pattern"
                    )
                self._lut[key] = (marker_id, rot)

    def decode(self, bits: Iterable[int]) -> tuple[int, int] | None:
        """Decode a 14-slot occupancy pattern.

        Returns ``(marker_id, rotation)`` where ``rotation`` is the number
        of slots the observed pattern is rotated against the canonical
        orientation, or ``None`` if the pattern is not in the codebook.
        """
        key = tuple(int(b) for b in bits)
        if len(key) != N_SLOTS:
            raise ValueError(f"pattern must have {N_SLOTS} bits")
        return self._lut.get(key)
