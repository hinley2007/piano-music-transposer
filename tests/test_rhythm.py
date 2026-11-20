"""Tests for beat grids, quantization and beat-level choices."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from music_types import NoteEvent
from rhythm import (
    BeatGrid,
    build_beat_grid,
    choose_beat_level,
    estimate_onset_lag,
    is_compound,
    quantize_notes,
)


def steady_grid(bpm: float = 120.0, beats: int = 64, beats_per_bar: int = 4) -> BeatGrid:
    interval = 60.0 / bpm
    return BeatGrid(np.arange(beats) * interval, beats_per_bar=beats_per_bar)


def test_beat_positions_round_trip() -> None:
    grid = steady_grid()
    seconds = np.array([0.0, 0.25, 1.1, 7.3])
    assert np.allclose(grid.to_seconds(grid.to_beats(seconds)), seconds)


def test_follows_tempo_changes() -> None:
    # Beats speed up from 0.6 s to 0.4 s apart; positions follow the beats.
    beats = np.cumsum([0.0] + [0.6] * 8 + [0.4] * 8)
    grid = BeatGrid(beats)
    assert grid.to_beats([beats[3]])[0] == 3
    assert abs(grid.to_beats([beats[12] + 0.2])[0] - 12.5) < 1e-9


def test_sloppy_eighths_snap_to_eighths() -> None:
    grid = steady_grid()  # 0.5 s per beat
    starts = [0.02, 0.27, 0.49, 0.76, 1.01, 1.24]
    notes = [NoteEvent(s, s + 0.2, 60 + i) for i, s in enumerate(starts)]
    result = quantize_notes(notes, grid)
    assert [n.start for n in result.notes] == [
        Fraction(0), Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(2), Fraction(5, 2)
    ]


def test_detects_triplets() -> None:
    grid = steady_grid()
    starts = [0.0, 0.167, 0.333, 0.5, 0.667, 0.833]
    notes = [NoteEvent(s, s + 0.15, 67) for s in starts]
    result = quantize_notes(notes, grid)
    assert [n.start for n in result.notes] == [Fraction(k, 3) for k in range(6)]


def test_even_sextuplets_do_not_become_uneven_32nds() -> None:
    grid = steady_grid(bpm=100)
    beat = 0.6
    starts = [beat * k / 6 for k in range(6)]
    notes = [NoteEvent(s, s + 0.08, 72 + k) for k, s in enumerate(starts)]
    result = quantize_notes(notes, grid)
    assert [n.start for n in result.notes] == [Fraction(k, 6) for k in range(6)]


def test_chord_notes_share_one_start() -> None:
    grid = steady_grid()
    notes = [NoteEvent(1.0, 1.4, 60), NoteEvent(1.015, 1.4, 64), NoteEvent(0.995, 1.4, 67)]
    result = quantize_notes(notes, grid)
    assert {n.start for n in result.notes} == {Fraction(2)}


def test_grid_is_extended_and_gaps_filled() -> None:
    # One missing beat at 2.0 s and nothing tracked before 0.5 s.
    beats = [0.5, 1.0, 1.5, 2.5, 3.0, 3.5, 4.0, 4.5]
    grid = build_beat_grid(beats, [], duration=6.0)
    assert grid.beat_times[0] <= 0.0
    assert grid.beat_times[-1] >= 6.0
    assert np.any(np.isclose(grid.beat_times, 2.0))


@pytest.mark.parametrize("strength", [None, []])
def test_recurring_bass_chords_recover_waltz_meter_without_downbeats(strength) -> None:
    beats = np.arange(48) * 0.55
    notes = [NoteEvent(t, t + 0.2, 72) for t in beats]
    # The recording begins on a pickup, one beat before the first bass chord.
    for t in beats[1::3]:
        notes.extend(NoteEvent(t, t + 0.4, p) for p in (40, 47, 52))
    grid = build_beat_grid(beats, [], beats[-1], notes=notes, downbeat_strength=strength)
    assert grid.time_signature == (3, 4)
    assert grid.to_beats(beats[1])[0] % 3 == grid.downbeat_phase

    # Explicit bar marks still take precedence over the accompaniment alone.
    marked = build_beat_grid(beats, beats[::4], beats[-1], notes=notes, downbeat_strength=strength)
    assert marked.time_signature == (4, 4)


@pytest.mark.parametrize("count", [4, 48])
def test_missing_downbeats_keep_default_meter_without_recurring_accents(count) -> None:
    beats = np.arange(48) * 0.7
    notes = [NoteEvent(t, t + 0.4, p) for t in beats[:count] for p in (48, 55, 64)]
    grid = build_beat_grid(beats, [], beats[-1], notes=notes, downbeat_strength=[])
    assert grid.time_signature == (4, 4)


def test_waltz_meter_from_note_lengths_after_a_silent_intro() -> None:
    beats = np.arange(70) * 0.5
    # A single repeated pitch carries the meter through articulation alone.
    # The strength of each bar's first beat varies across a five-bar phrase.
    notes = []
    for index, t in enumerate(beats[5:62]):
        duration = 0.3 + 0.04 * ((index // 3) % 5) if index % 3 == 0 else 0.15
        notes.append(NoteEvent(t, t + duration, 72))
    grid = build_beat_grid(beats, [], beats[-1], notes=notes)
    assert grid.time_signature == (3, 4)
    assert grid.bar_start_beat(5) == 5


def test_half_bar_accents_without_downbeats_keep_common_time() -> None:
    beats = np.arange(48) * 0.5
    notes = [NoteEvent(t, t + 0.2, 72) for t in beats]
    notes += [NoteEvent(t, t + 0.6, 43) for t in beats[::2]]
    assert build_beat_grid(beats, [], beats[-1], notes=notes).time_signature == (4, 4)


def test_slow_half_note_beat_is_doubled_when_notes_are_fast() -> None:
    grid = steady_grid(bpm=66, beats_per_bar=2)  # tracker followed half notes
    interval = 60.0 / 66 / 4  # notes every quarter of a tracked beat
    notes = [NoteEvent(k * interval, k * interval + 0.1, 60 + k % 5) for k in range(64)]
    chosen = choose_beat_level(grid, notes)
    assert round(chosen.bpm) == 132
    assert chosen.beats_per_bar == 4


def test_mostly_triplet_beats_mean_compound_meter() -> None:
    assert is_compound({beat: 3 for beat in range(12)})
    assert not is_compound({beat: 2 for beat in range(12)})


def six_eight_notes(bars: int = 8, eighth: float = 0.3, first_beat: int = 0) -> list[NoteEvent]:
    """A melody on every eighth with a chord on eighths 1 and 4 of each 6/8 bar."""
    notes = []
    for index in range(bars * 6):
        start = (first_beat + index) * eighth
        notes.append(NoteEvent(start, start + 0.05, 76 + index % 5, 70))
        if index % 6 == 0:
            notes += [NoteEvent(start, start + 0.05, p, 70) for p in (44, 56, 63)]
        elif index % 6 == 3:
            notes += [NoteEvent(start, start + 0.05, p, 70) for p in (59, 63, 68)]
    return notes


def test_six_fast_beats_with_accents_in_threes_are_six_eight() -> None:
    from rhythm import six_beat_grouping

    grid = BeatGrid(np.arange(80) * 0.3, beats_per_bar=6)
    group, phase = six_beat_grouping(six_eight_notes(), grid)
    assert (group, phase) == (3, 0)
    chosen = choose_beat_level(grid, six_eight_notes())
    assert chosen.time_signature == (6, 8)


def test_six_fast_beats_with_accents_in_twos_are_three_four() -> None:
    from rhythm import six_beat_grouping

    notes = []
    for index in range(48):  # bass on quarter 1, chords on quarters 2 and 3
        start = index * 0.3
        if index % 6 == 0:
            notes.append(NoteEvent(start, start + 0.2, 40, 80))
        elif index % 2 == 0:
            notes += [NoteEvent(start, start + 0.2, p, 60) for p in (60, 64, 67)]
        notes.append(NoteEvent(start, start + 0.1, 79 + index % 3, 70))
    grid = BeatGrid(np.arange(60) * 0.3, beats_per_bar=6)
    assert six_beat_grouping(notes, grid)[0] == 2
    assert choose_beat_level(grid, notes).time_signature == (3, 4)


def test_tracker_bar_marks_six_apart_vote_for_six_beats() -> None:
    from rhythm import downbeat_gap_shares

    beats = np.arange(120) * 0.3
    marks = np.concatenate([beats[0:60:6], beats[63:120:6]])  # the count slips by 3 once
    shares = downbeat_gap_shares(beats, marks)
    assert max(shares, key=shares.get) == 6


def test_accent_repetition_finds_the_bar_length_despite_a_slip() -> None:
    from rhythm import accent_per_beat, accent_repetition

    beats = np.arange(120) * 0.3
    notes = six_eight_notes(bars=8) + six_eight_notes(bars=8, first_beat=50)
    repetition = accent_repetition(accent_per_beat(beats, notes))
    assert repetition[3] > 0.5 and repetition[6] > 0.5
    assert repetition[2] < 0 and repetition[4] < 0


def test_bar_lines_are_reseated_after_a_pause() -> None:
    from rhythm import align_bar_lines

    # Eight bars, a pause of 10 beats (not a whole bar), eight more bars.
    notes = six_eight_notes(bars=8) + six_eight_notes(bars=8, first_beat=58)
    beats = np.arange(110) * 0.3
    marks = np.concatenate([beats[0:48:6], beats[58:106:6]])
    grid = BeatGrid(beats, beats_per_bar=6, downbeat_times=marks)
    aligned = align_bar_lines(grid, notes)
    positions = aligned.to_beats([0.0, 58 * 0.3, 64 * 0.3])
    for position in positions:  # every section's chords start a bar
        assert (round(float(position)) - aligned.downbeat_phase) % 6 == 0


def test_beats_are_moved_onto_the_notes() -> None:
    from rhythm import lock_beats_to_notes

    # A low note on every beat (0.29 s apart) with a high note half-way between.
    # The tracker's beats are 40 ms early and wobble by 15 ms.
    notes = [NoteEvent(1.0 + k * 0.29, 1.03 + k * 0.29, 72) for k in range(40)]
    notes += [NoteEvent(1.145 + k * 0.29, 1.17 + k * 0.29, 96) for k in range(40)]
    beats = 0.96 + np.arange(38) * 0.29 + 0.015 * (-1.0) ** np.arange(38)
    locked = lock_beats_to_notes(beats, notes)
    onsets = np.array(sorted(n.start for n in notes))
    distance = np.array([np.min(np.abs(onsets - b)) for b in locked[1:-1]])
    assert float(np.mean(distance < 0.005)) > 0.9
    spacing = np.diff(locked[1:-1])
    assert float(np.std(spacing)) < 0.03  # still evenly spaced: no hopping


def test_estimates_a_constant_onset_lag() -> None:
    grid = steady_grid()  # eighth notes every 0.25 s
    notes = [NoteEvent(k * 0.25 + 0.08, k * 0.25 + 0.3, 60) for k in range(32)]
    assert abs(estimate_onset_lag(notes, grid) - 0.08) < 0.015
