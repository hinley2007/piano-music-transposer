"""Tests for key detection and key-aware spelling."""

from __future__ import annotations

from keys import (
    KeyEstimate,
    estimate_key,
    key_signature_sharps,
    pitch_class_histogram,
    spell_pitch,
)


def test_detects_c_major_from_scale_and_triad_weight() -> None:
    pitches = [60, 62, 64, 65, 67, 69, 71, 72, 60, 64, 67, 60, 67]
    found = estimate_key(pitch_class_histogram(pitches))
    assert (found.tonic, found.mode) == (0, "major")


def test_detects_a_minor() -> None:
    pitches = [57, 59, 60, 62, 64, 65, 68, 69, 57, 60, 64, 57, 64, 69, 68]
    found = estimate_key(pitch_class_histogram(pitches))
    assert (found.tonic, found.mode) == (9, "minor")


def test_a_hammered_fifth_does_not_pass_for_the_home_note() -> None:
    """La Campanella is in G# minor, but its 'bell' is D# (the fifth), played
    more than twice as often as any other note."""
    counts = {
        "C": 92, "C#": 356, "D": 184, "D#": 1256, "E": 364, "F": 133,
        "F#": 317, "G": 244, "G#": 520, "A": 156, "A#": 393, "B": 398,
    }
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    histogram = [counts[name] for name in names]
    found = estimate_key(histogram, final_bass=8)
    assert (found.tonic, found.mode) == (8, "minor")
    assert key_signature_sharps(found) == 5
    # Even without knowing the final bass note.
    assert estimate_key(histogram).tonic == 8


def test_flat_keys_use_flats() -> None:
    f_major = KeyEstimate(5, "major", 1.0)
    assert spell_pitch(70, f_major).name == "B-4"


def test_sharp_keys_use_sharps() -> None:
    e_major = KeyEstimate(4, "major", 1.0)
    assert spell_pitch(68, e_major).name == "G#4"
    assert spell_pitch(63, e_major).name == "D#4"


def test_leading_tone_in_minor_is_sharpened() -> None:
    a_minor = KeyEstimate(9, "minor", 1.0)
    assert spell_pitch(68, a_minor).name == "G#4"


def test_chromatic_note_follows_its_direction() -> None:
    c_major = KeyEstimate(0, "major", 1.0)
    assert spell_pitch(68, c_major, next_pitch=69).name == "G#4"  # rising
    assert spell_pitch(68, c_major, next_pitch=67).name == "A-4"  # falling


def test_spelling_keeps_the_sounding_pitch_across_octave_edges() -> None:
    f_sharp_major = KeyEstimate(6, "major", 1.0)
    spelled = spell_pitch(65, f_sharp_major)  # F natural sounds = E# in F# major
    assert spelled.name == "E#4"


def test_key_signatures() -> None:
    assert key_signature_sharps(KeyEstimate(7, "major", 1.0)) == 1
    assert key_signature_sharps(KeyEstimate(3, "minor", 1.0)) == -6  # E-flat minor
    assert key_signature_sharps(KeyEstimate(8, "major", 1.0)) == -4  # A-flat major
