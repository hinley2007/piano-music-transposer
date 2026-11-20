"""Reading a human performance the way a musician would write it down.

Pianists often *roll* a chord: the notes are played quickly one after another
from the bottom up, usually so that the top arrives on the beat. Taken
literally that is three or four very fast notes just before the beat. Printed
music writes one chord with an arpeggio sign, so rolled chords are found here
and treated as a single chord.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from music_types import NoteEvent

ROLL_MAX_GAP = 0.11  # seconds between successive notes of a roll
ROLL_MIN_GAP = 0.025  # closer than this is simply "together"
ROLL_MIN_INTERVAL = 3  # semitones: rolls move in chord tones, not in steps
ROLL_MAX_INTERVAL = 12
ROLL_MAX_SECONDS = 0.45
ROLL_MIN_NOTES = 3
ROLL_MAX_NOTES = 8
TOGETHER = 0.04  # notes this close to the roll's end arrive "with" it


def find_rolled_chords(notes: Sequence[NoteEvent]) -> list[list[int]]:
    """Return groups of note indices that form upward-rolled chords.

    A roll is a quick rising chain of chord tones that stands on its own: no
    other note in the same register continues the motion right before or
    after it (otherwise it is simply a fast arpeggio passage, which is left
    as written).
    """
    count = len(notes)
    if count < ROLL_MIN_NOTES:
        return []
    order = sorted(range(count), key=lambda i: (notes[i].start, notes[i].pitch))
    starts = np.asarray([notes[i].start for i in order])
    pitches = np.asarray([notes[i].pitch for i in order])
    used = np.zeros(count, dtype=bool)
    rolls: list[list[int]] = []

    for position in range(count):
        if used[position]:
            continue
        chain = [position]
        current = position
        while len(chain) < ROLL_MAX_NOTES:
            low = np.searchsorted(starts, starts[current] + ROLL_MIN_GAP, side="left")
            high = np.searchsorted(starts, starts[current] + ROLL_MAX_GAP, side="right")
            following = None
            for candidate in range(low, high):
                if used[candidate] or candidate in chain:
                    continue
                step = pitches[candidate] - pitches[current]
                if ROLL_MIN_INTERVAL <= step <= ROLL_MAX_INTERVAL:
                    following = candidate
                    break  # earliest in time
            if following is None:
                break
            chain.append(following)
            current = following
        if len(chain) < ROLL_MIN_NOTES:
            continue
        total = starts[chain[-1]] - starts[chain[0]]
        if total > ROLL_MAX_SECONDS:
            continue
        mean_gap = total / (len(chain) - 1)
        if not _stands_alone(chain, starts, pitches, mean_gap):
            continue
        for member in chain:
            used[member] = True
        rolls.append([order[member] for member in chain])
    return rolls


def _stands_alone(chain: list[int], starts: np.ndarray, pitches: np.ndarray, mean_gap: float) -> bool:
    """True if no nearby note in the same register continues the fast motion."""
    members = set(chain)
    band_low = pitches[chain[0]] - 12
    band_high = pitches[chain[-1]] + 12
    reach = 1.6 * mean_gap
    first, last = starts[chain[0]], starts[chain[-1]]
    # Before the roll.
    low = np.searchsorted(starts, first - reach, side="left")
    high = np.searchsorted(starts, first - TOGETHER, side="left")
    for index in range(low, high):
        if index not in members and band_low <= pitches[index] <= band_high:
            return False
    # After the roll (notes arriving together with its last note do not count).
    low = np.searchsorted(starts, last + TOGETHER, side="right")
    high = np.searchsorted(starts, last + reach, side="right")
    for index in range(low, high):
        if index not in members and band_low <= pitches[index] <= band_high:
            return False
    return True


def _beat_distance(time: float, beat_times: np.ndarray) -> float:
    """Distance from a time to the nearest beat, as a fraction of the local beat."""
    if beat_times.size < 2:
        return 0.0
    index = int(np.clip(np.searchsorted(beat_times, time), 1, beat_times.size - 1))
    interval = beat_times[index] - beat_times[index - 1]
    nearest = min(abs(time - beat_times[index - 1]), abs(time - beat_times[index]))
    return float(nearest / max(interval, 1e-6))


def collapse_rolled_chords(
    notes: Sequence[NoteEvent], beat_times: Sequence[float]
) -> list[NoteEvent]:
    """Give every note of a rolled chord the same start time and mark it ``rolled``.

    Where does the chord belong: where the roll starts or where it ends? The
    other notes tell. If a note outside the roll arrives with its top (a
    melody note the roll leads up to), the chord goes there. If one arrives
    only with its bottom (melody and bass struck together, the chord spread
    afterwards), the chord goes to the start. Only when the roll stands
    alone is the beat grid consulted.
    """
    result = list(notes)
    beats = np.asarray(beat_times, dtype=float)
    all_starts = np.asarray(sorted(n.start for n in notes))

    def has_company(time: float, members: int) -> bool:
        """Is some note outside the roll struck together with this moment?"""
        low = np.searchsorted(all_starts, time - 0.03, side="left")
        high = np.searchsorted(all_starts, time + 0.03, side="right")
        return (high - low) > members

    for roll in find_rolled_chords(notes):
        first = notes[roll[0]].start
        last = notes[roll[-1]].start
        with_first = has_company(first, sum(1 for i in roll if abs(notes[i].start - first) <= 0.03))
        with_last = has_company(last, sum(1 for i in roll if abs(notes[i].start - last) <= 0.03))
        if with_last:
            anchor = last
        elif with_first:
            anchor = first
        else:
            nearer_last = _beat_distance(last, beats) <= _beat_distance(first, beats) + 0.05
            anchor = last if nearer_last else first
        for index in roll:
            item = notes[index]
            length = max(item.end - item.start, 0.05)
            result[index] = replace(
                item, start=anchor, end=max(item.end, anchor + length), rolled=True
            )
    return result
