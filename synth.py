"""A tiny offline piano-like synthesizer to listen to a transcription."""

from __future__ import annotations

import io
import wave
from typing import Iterable

import numpy as np

SAMPLE_RATE = 22050


def render_notes(
    notes: Iterable[tuple[float, float, int, int]],
    *,
    total_seconds: float | None = None,
) -> np.ndarray:
    """Synthesize (start, end, midi pitch, velocity) notes into mono audio."""
    notes = [n for n in notes if n[1] > n[0]]
    if not notes:
        return np.zeros(SAMPLE_RATE, dtype=np.float32)
    end_time = max(n[1] for n in notes) + 1.0
    if total_seconds:
        end_time = max(end_time, total_seconds)
    output = np.zeros(int(end_time * SAMPLE_RATE) + 1, dtype=np.float32)
    harmonics = np.array([1.0, 0.45, 0.25, 0.12, 0.06])
    for start, end, pitch, velocity in notes:
        frequency = 440.0 * 2 ** ((pitch - 69) / 12)
        length = min(end - start, 4.0) + 0.25  # short release tail
        count = int(length * SAMPLE_RATE)
        t = np.arange(count) / SAMPLE_RATE
        tone = np.zeros(count, dtype=np.float32)
        for index, weight in enumerate(harmonics, start=1):
            partial = frequency * index
            if partial > SAMPLE_RATE / 2.2:
                break
            # Higher partials fade faster, like a struck string.
            decay = np.exp(-t * (1.2 + 0.9 * index) * (frequency / 440.0) ** 0.3)
            tone += (weight * np.sin(2 * np.pi * partial * t) * decay).astype(np.float32)
        attack = np.minimum(1.0, t / 0.005)
        release = np.clip((length - t) / 0.25, 0.0, 1.0)
        tone *= (attack * release).astype(np.float32) * (0.15 + 0.85 * velocity / 127.0)
        begin = int(start * SAMPLE_RATE)
        stop = min(output.size, begin + count)
        if begin < output.size:
            output[begin:stop] += tone[: stop - begin]
    peak = float(np.max(np.abs(output)))
    if peak > 0:
        output *= 0.9 / peak
    return output


def to_wav_bytes(audio: np.ndarray) -> bytes:
    """Encode mono float audio as 16-bit WAV."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()
