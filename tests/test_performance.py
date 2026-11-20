"""Tests for reading performance habits (rolled chords)."""

from __future__ import annotations

import numpy as np

from music_types import NoteEvent
from performance import collapse_rolled_chords, find_rolled_chords

BEATS = np.arange(0, 10, 0.5)


def test_finds_an_isolated_upward_roll() -> None:
    notes = [
        NoteEvent(1.00, 1.05, 56),
        NoteEvent(1.08, 1.13, 63),
        NoteEvent(1.15, 1.20, 71),
        NoteEvent(2.00, 2.30, 60),
    ]
    assert find_rolled_chords(notes) == [[0, 1, 2]]


def test_fast_arpeggio_passage_is_not_a_roll() -> None:
    # Even broken chords going up and down: the motion continues, so it is a
    # written-out passage, not one rolled chord.
    pattern = [48, 55, 64, 72, 64, 55]
    notes = [NoteEvent(i * 0.07, i * 0.07 + 0.05, pattern[i % 6]) for i in range(24)]
    assert find_rolled_chords(notes) == []


def test_stepwise_run_is_not_a_roll() -> None:
    notes = [NoteEvent(1 + i * 0.07, 1.05 + i * 0.07, 60 + i) for i in range(4)]
    assert find_rolled_chords(notes) == []


def test_roll_lands_with_the_melody_note_it_leads_to() -> None:
    # Bass roll before the beat; the melody note arrives with its top.
    notes = [
        NoteEvent(0.86, 0.90, 56),
        NoteEvent(0.93, 0.97, 63),
        NoteEvent(1.00, 1.04, 71),
        NoteEvent(1.01, 1.20, 83),  # melody, struck with the top of the roll
    ]
    collapsed = collapse_rolled_chords(notes, BEATS)
    chord = [n for n in collapsed if n.rolled]
    assert [n.pitch for n in chord] == [56, 63, 71]
    assert all(abs(n.start - 1.00) < 1e-9 for n in chord)
    assert collapsed[3].start == 1.01  # the melody note is not moved


def test_roll_stays_at_its_bass_when_the_melody_is_struck_with_the_bass() -> None:
    notes = [
        NoteEvent(1.00, 1.05, 48),
        NoteEvent(1.00, 1.50, 79),  # melody on the beat with the bass
        NoteEvent(1.08, 1.13, 55),
        NoteEvent(1.16, 1.21, 64),
    ]
    collapsed = collapse_rolled_chords(notes, BEATS)
    chord = sorted((n for n in collapsed if n.rolled), key=lambda n: n.pitch)
    assert [n.pitch for n in chord] == [48, 55, 64]
    assert all(abs(n.start - 1.00) < 1e-9 for n in chord)
