"""MIDI cleanup, musical analysis, quantization, and transposition."""

from __future__ import annotations

import copy
import logging
import math
from pathlib import Path

from music21 import chord, converter, interval, key, meter, note, stream, tempo

from config import (
    DEFAULT_TEMPO_BPM,
    DEFAULT_TIME_SIGNATURE,
    KEY_CONFIDENCE_THRESHOLD,
    MIN_NOTE_DURATION_QUARTER_LENGTH,
    SUPPORTED_QUANTIZATION,
)

LOGGER = logging.getLogger(__name__)


class MidiProcessingError(RuntimeError):
    """Raised when a MIDI file cannot be processed."""


class NoUsableNotesError(MidiProcessingError):
    """Raised when a score contains no usable notes."""


class KeyDetectionError(MidiProcessingError):
    """Raised when a reliable source key is unavailable."""


class TargetKeyError(MidiProcessingError):
    """Raised when a target key cannot be interpreted or reached faithfully."""


def load_midi(midi_path: Path) -> stream.Score:
    """Load a MIDI file as a music21 Score and verify that it has notes."""
    midi_path = Path(midi_path)
    try:
        parsed = converter.parse(str(midi_path))
    except Exception as exc:
        LOGGER.exception("Could not parse MIDI file %s", midi_path)
        raise MidiProcessingError(
            "The generated MIDI file could not be parsed."
        ) from exc

    if isinstance(parsed, stream.Score):
        score = parsed
    else:
        score = stream.Score()
        part = stream.Part()
        for element in parsed.flatten().notesAndRests:
            part.insert(element.offset, copy.deepcopy(element))
        score.insert(0, part)

    if not list(score.recurse().notes):
        raise NoUsableNotesError(
            "The transcription did not contain any usable musical notes."
        )
    return score


def _round_to_grid(value: float, grid_size: float) -> float:
    """Round a non-negative quarter-length value to the nearest rhythmic grid."""
    return math.floor((value / grid_size) + 0.5) * grid_size


def quantize_score(
    score: stream.Score,
    quantization: str | float,
    *,
    minimum_duration: float = MIN_NOTE_DURATION_QUARTER_LENGTH,
) -> stream.Score:
    """Return a copy with note/chord offsets and durations snapped to a grid."""
    if isinstance(quantization, str):
        try:
            grid_size = float(SUPPORTED_QUANTIZATION[quantization])
        except KeyError as exc:
            raise ValueError(f"Unsupported quantization: {quantization}") from exc
    else:
        grid_size = float(quantization)

    if grid_size <= 0:
        raise ValueError("Quantization grid size must be greater than zero.")

    cleaned = copy.deepcopy(score)
    for element in list(cleaned.recurse().notes):
        duration = float(element.duration.quarterLength)
        if duration < minimum_duration:
            if element.activeSite is not None:
                element.activeSite.remove(element)
            continue

        element.offset = _round_to_grid(float(element.offset), grid_size)
        quantized_duration = _round_to_grid(duration, grid_size)
        element.duration.quarterLength = max(grid_size, quantized_duration)

    if not list(cleaned.recurse().notes):
        raise NoUsableNotesError(
            "Only extremely short detection artifacts were found; no usable notes remain."
        )
    return cleaned


def count_detected_notes(score: stream.Score) -> int:
    """Count individual pitches, including every pitch inside each chord."""
    total = 0
    for element in score.recurse().notes:
        if isinstance(element, chord.Chord):
            total += len(element.pitches)
        elif isinstance(element, note.Note):
            total += 1
    return total


def detect_key(score: stream.Score) -> key.Key:
    """Estimate the score's key with music21."""
    if not list(score.recurse().notes):
        raise KeyDetectionError("A key cannot be estimated because the score has no notes.")
    try:
        detected = score.analyze("key")
    except Exception as exc:
        LOGGER.exception("music21 key analysis failed")
        raise KeyDetectionError("The original key could not be estimated.") from exc
    if not isinstance(detected, key.Key):
        raise KeyDetectionError("The original key could not be estimated.")
    return detected


def key_detection_is_confident(
    detected_key: key.Key,
    *,
    threshold: float = KEY_CONFIDENCE_THRESHOLD,
) -> bool:
    """Return whether music21's key correlation clears a modest confidence floor."""
    coefficient = getattr(detected_key, "correlationCoefficient", None)
    # Explicitly supplied Key objects have no coefficient and are trusted.
    return coefficient is None or float(coefficient) >= threshold


def estimate_tempo(score: stream.Score) -> tuple[float, bool]:
    """Return BPM and whether it came from the MIDI rather than the fallback."""
    for mark in score.recurse().getElementsByClass(tempo.MetronomeMark):
        try:
            bpm = mark.getQuarterBPM()
        except Exception:
            bpm = None
        if bpm is not None and float(bpm) > 0:
            return float(bpm), True
    return DEFAULT_TEMPO_BPM, False


def ensure_score_defaults(score: stream.Score, bpm: float) -> stream.Score:
    """Add default time signature and tempo when the MIDI does not provide them."""
    prepared = copy.deepcopy(score)
    parts = list(prepared.parts)
    destination: stream.Stream = parts[0] if parts else prepared

    if not list(prepared.recurse().getElementsByClass(meter.TimeSignature)):
        destination.insert(0, meter.TimeSignature(DEFAULT_TIME_SIGNATURE))
    if not list(prepared.recurse().getElementsByClass(tempo.MetronomeMark)):
        destination.insert(0, tempo.MetronomeMark(number=bpm))
    return prepared


def add_key_signature(score: stream.Score, score_key: key.Key) -> stream.Score:
    """Return a copy with a key indication at the start of every musical part."""
    updated = copy.deepcopy(score)
    destinations: list[stream.Stream] = list(updated.parts) or [updated]

    for destination in destinations:
        for existing in list(
            destination.getElementsByClass((key.Key, key.KeySignature))
        ):
            destination.remove(existing)
        destination.insert(0, key.Key(score_key.tonic.name, score_key.mode))
    return updated


def transpose_by_semitones(score: stream.Score, semitones: int) -> stream.Score:
    """Transpose a score chromatically by -12 through +12 semitones."""
    if not -12 <= int(semitones) <= 12:
        raise ValueError("Semitone transposition must be between -12 and +12.")
    return score.transpose(int(semitones), inPlace=False)


def parse_target_key(target_key_name: str) -> key.Key:
    """Parse UI text such as 'F# minor' into a music21 Key."""
    pieces = target_key_name.strip().split()
    if len(pieces) != 2 or pieces[1].lower() not in {"major", "minor"}:
        raise TargetKeyError(
            f"'{target_key_name}' is not a valid target key. Use a name such as "
            "'G major' or 'F# minor'."
        )
    try:
        return key.Key(pieces[0], pieces[1].lower())
    except Exception as exc:
        raise TargetKeyError(
            f"'{target_key_name}' is not a key that music21 can interpret."
        ) from exc


def transpose_to_target_key(
    score: stream.Score,
    target_key_name: str,
    *,
    original_key: key.Key | None = None,
) -> tuple[stream.Score, key.Key]:
    """Move a score's tonic to a target key while preserving its major/minor mode."""
    source_key = original_key or detect_key(score)
    if not key_detection_is_confident(source_key):
        raise KeyDetectionError(
            "The original key estimate is not confident enough for target-key "
            "transposition. Choose Semitones mode instead."
        )

    target = parse_target_key(target_key_name)
    if source_key.mode != target.mode:
        raise TargetKeyError(
            f"The detected music is {source_key.mode}, but the target is {target.mode}. "
            "Chromatic transposition cannot change major music into minor (or vice "
            "versa) without rewriting notes. Choose a target with the same mode or "
            "use Semitones mode."
        )

    transposition_interval = interval.Interval(source_key.tonic, target.tonic)
    transposed = score.transpose(transposition_interval, inPlace=False)
    return add_key_signature(transposed, target), target
