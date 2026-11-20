"""Tests for upload validation."""

from __future__ import annotations

import pytest

from audio_utils import AudioValidationError, sanitize_filename, validate_audio_file


def test_rejects_unsupported_file_extension() -> None:
    with pytest.raises(AudioValidationError, match="Unsupported file type"):
        validate_audio_file("notes.txt", 100)


def test_rejects_empty_upload() -> None:
    with pytest.raises(AudioValidationError, match="empty"):
        validate_audio_file("recording.wav", 0)


def test_sanitizes_uploaded_filename() -> None:
    assert sanitize_filename("../../My demo (take 1).WAV") == "My_demo_take_1.wav"


def test_accepts_common_audio_formats() -> None:
    assert validate_audio_file("song.FLAC", 100) == "song.flac"
    assert validate_audio_file("voice memo.m4a", 100) == "voice_memo.m4a"


def test_retuning_moves_a_sharp_tone_to_concert_pitch() -> None:
    import numpy as np

    from audio_utils import retune

    rate = 16000
    t = np.arange(rate) / rate
    sharp_a = np.sin(2 * np.pi * 440 * 2 ** (0.3 / 12) * t).astype(np.float32)  # +30 cents
    fixed = retune(sharp_a, rate, 0.3)
    spectrum = np.abs(np.fft.rfft(fixed * np.hanning(fixed.size)))
    peak_hz = np.argmax(spectrum) * rate / fixed.size
    assert abs(peak_hz - 440) < 2

