"""Tests for music cleanup and transposition."""

from __future__ import annotations

import pytest
from music21 import key, note, stream

from music_processing import (
    quantize_score,
    transpose_by_semitones,
    transpose_to_target_key,
)


def make_test_melody() -> stream.Score:
    """Create a small original C-major pitch collection for tests."""
    score = stream.Score()
    part = stream.Part()
    for offset, pitch_name in zip((0.13, 1.12, 2.11, 3.13), ("C4", "D4", "E4", "G4")):
        melody_note = note.Note(pitch_name, quarterLength=0.83)
        part.insert(offset, melody_note)
    score.insert(0, part)
    return score


def test_quantizes_note_offsets() -> None:
    quantized = quantize_score(make_test_melody(), "Eighth notes")
    offsets = [float(item.offset) for item in quantized.parts[0].notes]
    assert offsets == [0.0, 1.0, 2.0, 3.0]


def test_quantizes_note_durations() -> None:
    quantized = quantize_score(make_test_melody(), "Eighth notes")
    durations = [
        float(item.duration.quarterLength) for item in quantized.parts[0].notes
    ]
    assert durations == [1.0, 1.0, 1.0, 1.0]


def test_transposes_simple_melody_by_semitones() -> None:
    transposed = transpose_by_semitones(make_test_melody(), 2)
    pitches = [item.nameWithOctave for item in transposed.recurse().notes]
    assert pitches == ["D4", "E4", "F#4", "A4"]


def test_transposes_c_major_melody_to_g_major() -> None:
    transposed, target = transpose_to_target_key(
        make_test_melody(),
        "G major",
        original_key=key.Key("C", "major"),
    )
    pitches = [item.nameWithOctave for item in transposed.recurse().notes]
    assert pitches == ["G4", "A4", "B4", "D5"]
    assert target.name == "G major"


def test_rejects_major_to_minor_as_simple_transposition() -> None:
    with pytest.raises(Exception, match="cannot change major music into minor"):
        transpose_to_target_key(
            make_test_melody(),
            "A minor",
            original_key=key.Key("C", "major"),
        )

