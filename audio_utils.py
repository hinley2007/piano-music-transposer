"""Audio validation, decoding, and tuning helpers."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from config import (
    ALLOWED_AUDIO_EXTENSIONS,
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
)

LOGGER = logging.getLogger(__name__)


class AudioValidationError(ValueError):
    """Raised when an uploaded audio file is not acceptable."""


class AudioConversionError(RuntimeError):
    """Raised when an audio file cannot be decoded."""


def sanitize_filename(filename: str) -> str:
    """Return a safe basename while preserving a supported file extension."""
    basename = Path(filename).name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(basename).stem).strip("._")
    extension = Path(basename).suffix.lower()
    return f"{stem or 'uploaded_audio'}{extension}"


def validate_audio_file(
    filename: str,
    size_bytes: int,
    *,
    max_size_bytes: int = MAX_UPLOAD_SIZE_BYTES,
) -> str:
    """Validate an upload's name and size and return a sanitized filename."""
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        supported = ", ".join(sorted(ALLOWED_AUDIO_EXTENSIONS))
        raise AudioValidationError(
            f"Unsupported file type '{extension or 'none'}'. Use {supported}."
        )
    if size_bytes <= 0:
        raise AudioValidationError("The uploaded file is empty.")
    if size_bytes > max_size_bytes:
        raise AudioValidationError(
            f"The uploaded file exceeds the {MAX_UPLOAD_SIZE_MB} MB limit."
        )
    return sanitize_filename(filename)


def find_ffmpeg() -> str:
    """Return the FFmpeg executable or raise a friendly error."""
    located = shutil.which("ffmpeg")
    if not located:
        raise AudioConversionError(
            "FFmpeg was not found. Install FFmpeg (for example with "
            "'winget install Gyan.FFmpeg') and make sure it is on PATH."
        )
    return located


def decode_audio(
    path: Path,
    sample_rate: int,
    *,
    mono: bool = False,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> np.ndarray:
    """Decode any FFmpeg-readable file to float32 samples.

    Returns an array shaped (samples,) for mono or (samples, 2) for stereo.
    """
    command = [find_ffmpeg(), "-nostdin", "-v", "error"]
    if start_seconds:
        command += ["-ss", f"{max(0.0, start_seconds):.3f}"]
    command += ["-i", str(path)]
    if end_seconds is not None:
        length = end_seconds - (start_seconds or 0.0)
        if length <= 0:
            raise AudioValidationError("The selected time range is empty.")
        command += ["-t", f"{length:.3f}"]
    channels = 1 if mono else 2
    command += [
        "-vn",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, check=False)
    except OSError as exc:
        raise AudioConversionError("FFmpeg could not be started.") from exc
    if completed.returncode != 0:
        LOGGER.error("FFmpeg failed: %s", completed.stderr.decode(errors="replace"))
        raise AudioConversionError(
            "The audio file could not be decoded. It may be corrupted or use an "
            "unsupported encoding."
        )
    samples = np.frombuffer(completed.stdout, dtype=np.float32)
    if samples.size == 0:
        raise AudioConversionError("The audio file contains no samples.")
    if mono:
        return samples.copy()
    return samples.reshape(-1, 2).copy()


def get_duration_seconds(path: Path) -> float:
    """Length of an audio file in seconds (uses ffprobe, falls back to decoding)."""
    probe = shutil.which("ffprobe")
    if probe:
        completed = subprocess.run(
            [
                probe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            return float(completed.stdout.strip())
        except ValueError:
            pass
    samples = decode_audio(Path(path), 8000, mono=True)
    return samples.size / 8000.0


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Average stereo audio to mono."""
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1)


def estimate_tuning_semitones(mono_audio: np.ndarray, sample_rate: int) -> float:
    """Estimate how far the recording is from A=440 Hz, in semitones (-0.5..0.5)."""
    import librosa

    # A few minutes is plenty and keeps this step fast.
    excerpt = mono_audio[: sample_rate * 180]
    if excerpt.size < sample_rate:
        return 0.0
    try:
        # With 12 bins per octave, librosa's answer is already in semitones.
        tuning = float(
            librosa.estimate_tuning(y=excerpt, sr=sample_rate, bins_per_octave=12)
        )
    except Exception:  # pragma: no cover - librosa edge cases
        LOGGER.warning("Tuning estimation failed", exc_info=True)
        return 0.0
    return float(np.clip(tuning, -0.5, 0.5))


def retune(audio: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
    """Shift pitch by resampling (this also changes speed by the same tiny ratio).

    A recording that is ``semitones`` sharp is moved down to concert pitch.
    Timestamps measured on the result must be divided by ``speed_ratio(semitones)``.
    """
    if abs(semitones) < 1e-3:
        return audio
    import soxr

    ratio = speed_ratio(semitones)
    # Pretend the audio was recorded at a lower/higher rate, then resample back.
    return soxr.resample(audio, sample_rate / ratio, sample_rate).astype(np.float32)


def speed_ratio(semitones: float) -> float:
    """Return how much longer audio becomes after :func:`retune`."""
    return float(2.0 ** (semitones / 12.0))
