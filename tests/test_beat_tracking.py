"""Tests for the beat-refinement helpers (no audio or models needed)."""

from __future__ import annotations

import numpy as np

from beat_tracking import (
    FPS,
    _figure_size,
    _fit_note_pulse,
    _note_pulse,
    _period_from_tracker,
    _phase_support,
    meter_scores,
    note_contour,
    steady_beats,
)
from music_types import NoteEvent


def bell_figure(count: int = 60, gap: float = 0.14) -> list[NoteEvent]:
    """Melody note, high 'bell', melody note, bell... (a two-note figure)."""
    notes = []
    for index in range(count):
        pitch = 99 if index % 2 else 80 + (index // 2) % 5
        notes.append(NoteEvent(index * gap, index * gap + 0.03, pitch))
    return notes


def triplet_figure(count: int = 60, gap: float = 0.14) -> list[NoteEvent]:
    """Low, middle, high, low, middle, high... (a three-note figure)."""
    return [NoteEvent(i * gap, i * gap + 0.03, (60, 67, 76)[i % 3]) for i in range(count)]


def test_two_note_figures_are_recognised() -> None:
    assert _figure_size(note_contour(bell_figure()), 2.0, 2.84) == 2


def test_three_note_figures_are_recognised() -> None:
    assert _figure_size(note_contour(triplet_figure()), 2.0, 2.84) == 3


def test_slow_tracker_beat_over_pairs_is_split_in_three() -> None:
    """The La Campanella case: the tracker follows dotted quarters (0.84 s, six
    notes each) while the piece's pulse elsewhere is 0.35 s. Six notes in
    pairs are three beats of two, not two beats of three."""
    raw_beats = np.arange(0, 8.4, 0.84)
    period = _period_from_tracker(raw_beats, 0.35, 420, note_contour(bell_figure()))
    assert abs(float(np.median(period)) - 0.28) < 0.01


def test_slow_tracker_beat_over_threes_is_split_in_two() -> None:
    raw_beats = np.arange(0, 8.4, 0.84)
    period = _period_from_tracker(raw_beats, 0.35, 420, note_contour(triplet_figure()))
    assert abs(float(np.median(period)) - 0.42) < 0.01


def test_tempo_is_corrected_against_the_notes_own_pulse() -> None:
    times = np.arange(0, 6, 0.145)
    pulse = _note_pulse(times, 3.0)
    assert abs(pulse - 0.145) < 0.002
    # The tracker says 0.34 s per beat; two notes of 0.145 s make 0.29 s.
    period = np.full(6 * FPS, 0.34)
    locked, divisions = _fit_note_pulse(period, times)
    assert abs(locked[3 * FPS] - 0.29) < 0.005
    assert divisions[3 * FPS] == 2
    # No steady pulse (a slow-down): the tracker's tempo is left alone.
    slowing = np.cumsum(np.linspace(0.15, 0.6, 20))
    assert _note_pulse(slowing, float(slowing[10])) is None
    unchanged, divisions = _fit_note_pulse(period, np.array([]))
    assert np.allclose(unchanged, period) and not divisions.any()


def test_syncopated_opening_follows_the_beat_through_tempo_changes() -> None:
    intervals = np.linspace(0.58, 0.48, 32)
    beats = 0.16 + np.r_[0, np.cumsum(intervals)]
    offbeats = beats[:-1] + intervals / 2
    # The player skips occasional sixteenths, but keeps the same subdivision.
    onsets = (beats[:-1, None] + intervals[:, None] * np.arange(4) / 4).ravel()
    onsets = onsets[np.arange(onsets.size) % 7 != 1]
    contour = (onsets, np.full(onsets.size, 60))
    activation = np.zeros(int((beats[-1] + 0.4) * FPS))
    syncopated = np.arange(intervals.size) < 10
    activation[np.rint(beats[:-1] * FPS).astype(int)] = np.where(syncopated, 0.14, 0.95)
    activation[np.rint(offbeats * FPS).astype(int)] = np.where(syncopated, 0.9, 0.08)
    raw = np.where(syncopated, offbeats, beats[:-1])

    recovered = steady_beats(activation, raw, contour=contour)
    played = recovered[(recovered >= beats[0] - 0.04) & (recovered < beats[-1] - 0.04)]
    assert np.allclose(played, beats[:-1], atol=1 / FPS)


def test_phase_evidence_does_not_carry_across_a_rest() -> None:
    frames = np.arange(640)
    # Both passages divide the beat in two, but start on opposite subdivisions.
    onsets = np.r_[frames[0:240:10], frames[320:600:10]] / FPS
    activation = np.zeros(frames.size)
    activation[0:240:20] = 1
    activation[330:600:20] = 1
    support = _phase_support(activation, onsets, np.full(frames.size, 0.4), np.full(frames.size, 2))
    assert not support.any()


def test_bar_mark_spacing_decides_the_meter() -> None:
    strength = np.zeros(120)
    best = meter_scores(strength, 0.3, None, {2: 0.1, 3: 0.3, 4: 0.1, 6: 0.8}, {3: 0.6, 6: 0.5})[0]
    assert best[1] == 6
