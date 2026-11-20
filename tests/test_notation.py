"""Tests for hand splitting, staff events and score building."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from exporters import score_to_musicxml_bytes
from keys import KeyEstimate
from music_types import QuantizedNote
from notation import (
    PartSpec,
    StaffEvent,
    StaffSpec,
    bar_clefs,
    build_score,
    build_staff_events,
    split_hands,
    split_readable,
)
from rhythm import BeatGrid

F = Fraction


def q(start, end, pitch, velocity=80) -> QuantizedNote:
    return QuantizedNote(F(start), F(end), pitch, velocity, raw_start=float(start), raw_end=float(end))


def test_low_and_high_notes_go_to_different_hands() -> None:
    notes = [q(0, 1, 40), q(0, 1, 76), q(1, 2, 43), q(1, 2, 79)]
    placed = {(n.pitch, n.staff) for n in split_hands(notes)}
    assert placed == {(40, 1), (76, 0), (43, 1), (79, 0)}


def test_accompaniment_pattern_stays_in_the_left_hand() -> None:
    # Melody C5 held for two beats over an Alberti bass C4-G4-E4-G4.
    notes = [q(0, 2, 72)]
    for index, pitch in enumerate([60, 67, 64, 67, 60, 67, 64, 67]):
        notes.append(q(F(index, 2), F(index + 1, 2), pitch))
    notes.append(q(2, 3, 76))
    placed = split_hands(notes)
    left = sorted(n.pitch for n in placed if n.staff == 1)
    assert left == [60, 60, 64, 64, 67, 67, 67, 67]


def test_short_chords_stay_together_under_a_sustained_melody() -> None:
    # The same recording measured in quarter-note or eighth-note beats.
    for beats_per_quarter in (1, 2):
        notes = []
        for index, melody in enumerate((64, 65, 67, 64)):
            start = index * beats_per_quarter
            release = start + F(beats_per_quarter, 2)
            notes.extend(q(start, release, p) for p in (55, 59))
            notes.append(q(start, start + beats_per_quarter, melody))
        placed = split_hands(notes, beat_seconds=0.5 / beats_per_quarter)
        assert all(n.staff == (1 if n.pitch in (55, 59) else 0) for n in placed)


def test_melody_between_bell_notes_stays_in_the_right_hand() -> None:
    """La Campanella's figure: melody note, repeated high note, melody note...
    The melody dips into the left hand's register (below its chords' top)
    but belongs to the hand that plays the bell."""
    notes = []
    melody = [75, 76, 75, 73, 71, 71, 70, 68, 67, 68, 70, 71]
    for index, pitch_value in enumerate(melody):
        start = F(index, 2)
        notes.append(q(start, start + F(1, 8), pitch_value))
        notes.append(q(start + F(1, 4), start + F(3, 8), 87))  # the bell, D#6
        if index % 3 == 0:  # left-hand chord under every third melody note
            notes += [q(start, start + F(1, 8), p) for p in (59, 67, 70)]
    placed = split_hands(notes, beat_seconds=0.58)
    bell_hands = {n.staff for n in placed if n.pitch == 87}
    assert bell_hands == {0}
    melody_starts = {F(i, 2) for i in range(len(melody))}
    melody_in_right = [
        n.staff == 0 for n in placed if n.start in melody_starts and n.pitch in melody and n.pitch > 70
    ]
    assert sum(melody_in_right) >= len(melody_in_right) - 1
    assert all(n.staff == 1 for n in placed if n.pitch <= 59)


def test_repeated_left_hand_chords_do_not_pull_the_melody_over() -> None:
    """Ragtime: the left hand repeats a chord while the right hand plays between."""
    notes = []
    for index in range(8):
        start = F(index, 2)
        notes += [q(start, start + F(1, 4), p) for p in (51, 56, 60)]
        notes.append(q(start + F(1, 4), start + F(1, 2), 80 + index % 3))
    placed = split_hands(notes, beat_seconds=0.6)
    assert all(n.staff == 1 for n in placed if n.pitch <= 60)
    assert all(n.staff == 0 for n in placed if n.pitch >= 80)


def test_no_note_is_dropped_from_a_big_chord() -> None:
    notes = [q(0, 1, p) for p in (36, 43, 48, 52, 55, 60, 64, 67, 72, 76, 79, 84)]
    placed = split_hands(notes)
    assert len(placed) == 12
    assert {n.staff for n in placed} == {0, 1}


def test_detached_piano_notes_fill_their_beat_and_get_a_dot() -> None:
    # Two staccato chords a beat apart, played for a tenth of a beat each.
    notes = [
        QuantizedNote(F(0), F(1, 4), 60, 80, raw_start=0.0, raw_end=0.1),
        QuantizedNote(F(0), F(1, 4), 64, 80, raw_start=0.0, raw_end=0.1),
        QuantizedNote(F(1), F(5, 4), 62, 80, raw_start=1.0, raw_end=1.1),
    ]
    events = build_staff_events(notes, keyboard=True)
    assert [(e.start, e.end) for e in events] == [(F(0), F(1)), (F(1), F(2))]
    assert all(e.staccato for e in events)
    # Without the keyboard rule the first chord is a short note followed by rests.
    plain = build_staff_events(notes)
    assert plain[0].end == F(1, 4) and not plain[0].staccato


def test_rolled_notes_make_an_arpeggiated_chord() -> None:
    notes = [QuantizedNote(F(0), F(1), p, 80, raw_end=1.0, rolled=True) for p in (48, 55, 64)]
    events = build_staff_events(notes, keyboard=True)
    assert events[0].rolled
    grid = BeatGrid(np.arange(16) * 0.5, beats_per_bar=4)
    part = PartSpec("Piano", [StaffSpec(events, "bass")])
    xml = score_to_musicxml_bytes(
        build_score([part], grid, KeyEstimate(0, "major", 1.0), title="Roll", bpm=120)
    ).decode("utf-8")
    assert "<arpeggiate" in xml


def test_six_eight_groups_notes_by_dotted_quarters() -> None:
    from notation import COMPOUND_PULSE, notation_pulse

    assert notation_pulse(6, 8) == COMPOUND_PULSE
    assert notation_pulse(3, 4) == F(1)
    # A rest covering eighths 2-6 of a 6/8 bar: quarter, then a dotted quarter.
    assert split_readable(F(1, 2), F(3), COMPOUND_PULSE, F(3)) == [
        (F(1, 2), F(3, 2)),
        (F(3, 2), F(3)),
    ]
    # A sixteenth inside an eighth is completed to the eighth first.
    assert split_readable(F(1, 4), F(3, 2), COMPOUND_PULSE, F(3)) == [
        (F(1, 4), F(1, 2)),
        (F(1, 2), F(3, 2)),
    ]


def test_six_eight_score_with_eighth_note_beats() -> None:
    """The grid counts eighths; the score shows 6/8 with a dotted-quarter tempo."""
    grid = BeatGrid(np.arange(40) * 0.3, beats_per_bar=6, beat_unit=F(1, 2))
    events = [StaffEvent(F(i), F(i + 1), [72 + i % 3]) for i in range(12)]
    part = PartSpec("Piano", [StaffSpec(events, "treble")])
    xml = score_to_musicxml_bytes(
        build_score([part], grid, KeyEstimate(8, "minor", 1.0), title="6/8", bpm=grid.bpm)
    ).decode("utf-8")
    assert "<beats>6</beats>" in xml and "<beat-type>8</beat-type>" in xml
    assert "<fifths>5</fifths>" in xml  # G sharp minor
    assert "<per-minute>67</per-minute>" in xml  # 200 eighths a minute = 67 dotted quarters
    assert xml.count("<measure ") == 2


def test_very_high_bars_get_an_ottava_line() -> None:
    grid = BeatGrid(np.arange(40) * 0.5, beats_per_bar=4)
    high = [StaffEvent(F(i), F(i + 1), [99 if i % 2 else 87]) for i in range(8)]
    low = [StaffEvent(F(8 + i), F(9 + i), [64]) for i in range(4)]
    part = PartSpec("Piano", [StaffSpec(high + low, "treble")])
    score = build_score([part], grid, KeyEstimate(0, "major", 1.0), title="8va", bpm=120)
    xml = score_to_musicxml_bytes(score).decode("utf-8")
    assert xml.count("<octave-shift") == 2  # one start, one stop
    # Notes keep their sounding pitch (D#7 is still octave 7).
    assert "<octave>7</octave>" in xml


def test_simultaneous_notes_become_one_chord_without_overlaps() -> None:
    notes = [q(0, 2, 60), q(0, 2, 64), q(1, 3, 67)]
    events = build_staff_events(notes)
    assert [e.pitches for e in events] == [[60, 64], [67]]
    assert events[0].end == 1  # cut where the next chord starts


def test_short_gaps_are_closed() -> None:
    notes = [q(0, F(3, 4), 60), q(1, 2, 62)]
    events = build_staff_events(notes)
    assert events[0].end == 1


def test_rests_are_split_on_beats() -> None:
    pieces = split_readable(F(1, 2), F(4), F(1), F(4))
    assert pieces == [(F(1, 2), F(1)), (F(1), F(2)), (F(2), F(4))]


def test_clef_switches_when_the_left_hand_stays_high() -> None:
    events = [StaffEvent(F(bar * 4), F(bar * 4 + 4), [72, 76]) for bar in range(2, 4)]
    events = [StaffEvent(F(0), F(4), [43])] + events
    clefs = bar_clefs(events, "bass", 4, lambda b: F(b), F(4))
    assert clefs[0] == "bass" and clefs[2] == "treble"


def test_clef_follows_the_ledger_lines() -> None:
    from notation import ledger_lines

    assert ledger_lines(60, "treble") == 1 and ledger_lines(60, "bass") == 1  # middle C
    assert ledger_lines(64, "treble") == 0 and ledger_lines(43, "bass") == 0
    assert ledger_lines(36, "treble") > 5  # low C needs the bass clef
    # A left hand that starts with high octaves and then drops to bass octaves.
    events = [StaffEvent(F(0), F(4), [63, 75]), StaffEvent(F(4), F(8), [63, 75])]
    events += [StaffEvent(F(8 + 4 * i), F(12 + 4 * i), [32, 44, 56]) for i in range(3)]
    clefs = bar_clefs(events, "bass", 5, lambda b: F(b), F(4))
    assert clefs == ["treble", "treble", "bass", "bass", "bass"]


def test_builds_musicxml_with_triplets_and_ties() -> None:
    grid = BeatGrid(np.arange(32) * 0.5, beats_per_bar=4)
    right = [
        StaffEvent(F(0), F(1, 3), [72]),
        StaffEvent(F(1, 3), F(2, 3), [74]),
        StaffEvent(F(2, 3), F(1), [76]),
        StaffEvent(F(3), F(5), [77]),  # crosses the bar line -> tie
    ]
    left = [StaffEvent(F(0), F(4), [48, 55])]
    part = PartSpec("Piano", [StaffSpec(right, "treble"), StaffSpec(left, "bass")])
    score = build_score([part], grid, KeyEstimate(0, "major", 1.0), title="Test", bpm=120)
    xml = score_to_musicxml_bytes(score).decode("utf-8")
    assert "<time-modification>" in xml  # triplet
    assert '<tie type="start"' in xml
    assert "<staves>2</staves>" in xml  # grand staff in one part
    assert xml.count("<measure ") == 2
