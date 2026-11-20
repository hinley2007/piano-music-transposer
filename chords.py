"""Chord-name estimation from audio (chroma) aligned to the beat grid."""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence

import numpy as np

from keys import KeyEstimate
from music_types import ChordLabel, NoteEvent
from rhythm import BeatGrid

CHROMA_SAMPLE_RATE = 22050
CHROMA_HOP = 1024

CHORD_QUALITIES: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "7": (0, 4, 7, 10),
    "min7": (0, 3, 7, 10),
    "maj7": (0, 4, 7, 11),
    "dim": (0, 3, 6),
    "sus4": (0, 5, 7),
}
# Small penalties so plain triads win unless the extra note is clearly there.
QUALITY_PENALTY = {"maj": 0.0, "min": 0.0, "7": 0.05, "min7": 0.05, "maj7": 0.07, "dim": 0.12, "sus4": 0.1}
TONE_WEIGHTS = (1.0, 0.85, 0.8, 0.6)  # root, third, fifth, seventh

MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
NATURAL_MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)


def compute_chroma(mono_audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (chroma 12 x frames, frame times in seconds)."""
    import librosa

    if sample_rate != CHROMA_SAMPLE_RATE:
        import soxr

        mono_audio = soxr.resample(mono_audio, sample_rate, CHROMA_SAMPLE_RATE)
    chroma = librosa.feature.chroma_cqt(
        y=mono_audio.astype(np.float32),
        sr=CHROMA_SAMPLE_RATE,
        hop_length=CHROMA_HOP,
        bins_per_octave=36,
        norm=None,
    )
    times = librosa.frames_to_time(
        np.arange(chroma.shape[1]), sr=CHROMA_SAMPLE_RATE, hop_length=CHROMA_HOP
    )
    return chroma.astype(np.float32), times


def _templates() -> tuple[np.ndarray, list[tuple[int, str]]]:
    rows, labels = [], []
    for root in range(12):
        for quality, intervals in CHORD_QUALITIES.items():
            vector = np.zeros(12)
            for weight, interval in zip(TONE_WEIGHTS, intervals):
                vector[(root + interval) % 12] = weight
            rows.append(vector / np.linalg.norm(vector))
            labels.append((root, quality))
    return np.asarray(rows), labels


def _diatonic_bonus(root: int, quality: str, key_estimate: KeyEstimate | None) -> float:
    if key_estimate is None:
        return 0.0
    scale = MAJOR_SCALE if key_estimate.mode == "major" else NATURAL_MINOR_SCALE
    scale_set = {(key_estimate.tonic + step) % 12 for step in scale}
    tones = {(root + interval) % 12 for interval in CHORD_QUALITIES[quality][:3]}
    if tones <= scale_set:
        return 0.06
    if key_estimate.mode == "minor" and quality in ("maj", "7"):
        # Dominant chord of a minor key (raised leading tone).
        if root == (key_estimate.tonic + 7) % 12:
            return 0.06
    return 0.0


def estimate_chords(
    chroma: np.ndarray,
    chroma_times: np.ndarray,
    grid: BeatGrid,
    *,
    first_beat: int,
    last_beat: int,
    bass_notes: Sequence[NoteEvent] = (),
    key_estimate: KeyEstimate | None = None,
) -> list[ChordLabel]:
    """Label each beat with a chord, smoothed so chords change at sensible places."""
    if chroma.size == 0 or last_beat <= first_beat:
        return []
    beat_indices = list(range(first_beat, last_beat))
    starts = grid.to_seconds(np.asarray(beat_indices, dtype=float))
    ends = grid.to_seconds(np.asarray(beat_indices, dtype=float) + 1)
    beat_chroma = np.zeros((len(beat_indices), 12))
    for row, (start, end) in enumerate(zip(starts, ends)):
        mask = (chroma_times >= start) & (chroma_times < end)
        if np.any(mask):
            beat_chroma[row] = chroma[:, mask].mean(axis=1)
    energy = beat_chroma.sum(axis=1)
    loud = energy > 0.05 * (np.percentile(energy, 95) + 1e-9)
    norms = np.linalg.norm(beat_chroma, axis=1, keepdims=True) + 1e-9
    normalized = beat_chroma / norms

    bass_by_beat: list[int | None] = []
    for start, end in zip(starts, ends):
        best_pitch, best_overlap = None, 0.0
        for item in bass_notes:
            overlap = min(end, item.end) - max(start, item.start)
            if overlap > best_overlap:
                best_overlap, best_pitch = overlap, item.pitch
        bass_by_beat.append(None if best_pitch is None else best_pitch % 12)

    templates, labels = _templates()
    scores = normalized @ templates.T  # beats x chords
    for column, (root, quality) in enumerate(labels):
        scores[:, column] -= QUALITY_PENALTY[quality]
        scores[:, column] += _diatonic_bonus(root, quality, key_estimate)
        for row, bass in enumerate(bass_by_beat):
            if bass is None:
                continue
            if bass == root:
                scores[row, column] += 0.12
            elif bass in {(root + i) % 12 for i in CHORD_QUALITIES[quality]}:
                scores[row, column] += 0.03
    no_chord_score = 0.35
    n_chords = len(labels)
    emission = np.full((len(beat_indices), n_chords + 1), 0.0)
    emission[:, :n_chords] = -scores * 6.0
    emission[:, n_chords] = -no_chord_score * 6.0
    emission[~loud, :n_chords] += 6.0
    emission[~loud, n_chords] = -6.0

    # Viterbi: changing chord costs less on downbeats and half-bar points.
    cost = emission[0].copy()
    back = np.zeros(emission.shape, dtype=np.int32)
    bar = grid.beats_per_bar
    for row in range(1, len(beat_indices)):
        position = (beat_indices[row] - grid.downbeat_phase) % bar
        if position == 0:
            change = 1.0
        elif bar % 2 == 0 and position == bar // 2:
            change = 1.8
        else:
            change = 3.2
        best_previous = int(np.argmin(cost))
        stay = cost
        switch = cost[best_previous] + change
        use_switch = switch < stay
        back[row] = np.where(use_switch, best_previous, np.arange(n_chords + 1))
        cost = np.minimum(stay, switch) + emission[row]
    path = np.empty(len(beat_indices), dtype=np.int32)
    path[-1] = int(np.argmin(cost))
    for row in range(len(beat_indices) - 1, 0, -1):
        path[row - 1] = back[row, path[row]]

    chords: list[ChordLabel] = []
    row = 0
    while row < len(path):
        state = path[row]
        end_row = row
        while end_row + 1 < len(path) and path[end_row + 1] == state:
            end_row += 1
        if state != n_chords:
            root, quality = labels[state]
            chords.append(
                ChordLabel(
                    start=Fraction(beat_indices[row]),
                    end=Fraction(beat_indices[end_row] + 1),
                    root=root,
                    quality=quality,
                )
            )
        row = end_row + 1
    return chords
