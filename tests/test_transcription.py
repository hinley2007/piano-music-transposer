"""Tests for note segmentation, cleanup and chord naming (no ML models needed)."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from chords import estimate_chords
from keys import KeyEstimate
from music_types import NoteEvent
from rhythm import BeatGrid
from transcription import HOP_SECONDS, clean_piano_notes, segment_notes


def hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def track(segments: list[tuple[float, float, float | None]], vibrato: float = 0.0):
    """Build (f0, confidence, loudness) frames from (start, end, midi or None) pieces."""
    total = int(max(end for _, end, _ in segments) / HOP_SECONDS) + 10
    f0 = np.zeros(total)
    confidence = np.zeros(total)
    loudness = np.full(total, -80.0)
    for start, end, pitch in segments:
        first, last = int(start / HOP_SECONDS), int(end / HOP_SECONDS)
        if pitch is None:
            continue
        frames = np.arange(first, last)
        wobble = vibrato * np.sin(2 * np.pi * 5.5 * frames * HOP_SECONDS)
        f0[first:last] = [hz(pitch + w) for w in wobble]
        confidence[first:last] = 0.9
        loudness[first:last] = -20.0
    return f0, confidence, loudness


def test_steady_notes_with_vibrato_stay_single_notes() -> None:
    f0, conf, loud = track([(0.0, 0.5, 67), (0.5, 1.0, 69), (1.0, 2.0, 71)], vibrato=0.4)
    notes = segment_notes(f0, conf, loud, part="melody", min_pitch=40, max_pitch=90)
    assert [n.pitch for n in notes] == [67, 69, 71]
    assert abs(notes[1].start - 0.5) < 0.06


def test_silence_separates_notes() -> None:
    f0, conf, loud = track([(0.0, 0.4, 60), (0.4, 0.6, None), (0.6, 1.0, 60)])
    notes = segment_notes(f0, conf, loud, part="melody", min_pitch=40, max_pitch=90)
    assert len(notes) == 2


def test_quiet_ghost_notes_are_removed_but_short_notes_are_kept() -> None:
    notes = [NoteEvent(0, 1, 60, 90), NoteEvent(0, 1, 72, 8), NoteEvent(1, 1.01, 64, 90)]
    kept = clean_piano_notes(notes, relative_velocity=0.12)
    # The faint note goes; the 10 ms staccato note is a real note and stays.
    assert [n.pitch for n in kept] == [60, 64]


def test_staccato_piece_keeps_every_note_at_default_sensitivity() -> None:
    """Regression: fast staccato playing has notes of ~30 ms; none may be dropped."""
    from config import DEFAULT_SENSITIVITY, SENSITIVITY_PRESETS

    notes = [NoteEvent(i * 0.14, i * 0.14 + 0.03, 60 + i % 12, 60 + i % 30) for i in range(200)]
    kept = clean_piano_notes(notes, **SENSITIVITY_PRESETS[DEFAULT_SENSITIVITY])
    assert len(kept) == len(notes)


def test_chord_names_follow_the_harmony() -> None:
    grid = BeatGrid(np.arange(40) * 0.5, beats_per_bar=4)
    progression = [(0, (0, 4, 7)), (7, (7, 11, 2)), (9, (9, 0, 4)), (5, (5, 9, 0))]  # C G Am F
    times = np.arange(0, 8.0, 0.05)
    chroma = np.zeros((12, times.size))
    for index, time in enumerate(times):
        _, tones = progression[int(time // 2.0) % 4]
        for tone in tones:
            chroma[tone, index] = 1.0
    labels = estimate_chords(
        chroma, times, grid, first_beat=0, last_beat=16, key_estimate=KeyEstimate(0, "major", 1.0)
    )
    names = [(label.root, label.quality, label.start) for label in labels]
    assert names == [(0, "maj", Fraction(0)), (7, "maj", Fraction(4)), (9, "min", Fraction(8)), (5, "maj", Fraction(12))]
