"""Note detection: turn audio into a list of notes (times in seconds).

Three engines are used, each for what it does best:

* Transkun - a state-of-the-art piano transcription model (piano audio).
* CREPE pitch tracking + note segmentation - one melody line (voice, flute,
  violin, bass...). A Viterbi pass keeps notes steady through vibrato.
* Spotify Basic Pitch - a general model for other polyphonic instruments.
"""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from typing import Callable

import numpy as np

from music_types import NoteEvent, PedalEvent

LOGGER = logging.getLogger(__name__)

ProgressCallback = Callable[[float], None]

_MODEL_LOCK = threading.Lock()
_PIANO_MODEL = None


class TranscriptionError(RuntimeError):
    """Raised when notes cannot be detected."""


class NoNotesDetectedError(TranscriptionError):
    """Raised when the recording produced no usable notes."""


# ---------------------------------------------------------------------------
# Piano (Transkun)
# ---------------------------------------------------------------------------

PIANO_SAMPLE_RATE = 44100


def _load_piano_model():
    global _PIANO_MODEL
    with _MODEL_LOCK:
        if _PIANO_MODEL is not None:
            return _PIANO_MODEL
        try:
            from importlib import resources

            import moduleconf
            import torch
        except ImportError as exc:  # pragma: no cover - install problem
            raise TranscriptionError(
                "The piano model is not installed. Run "
                "'pip install -r requirements.txt'."
            ) from exc
        package = resources.files("transkun") / "pretrained"
        config_manager = moduleconf.parseFromFile(str(package / "2.0.conf"))
        model_class = config_manager["Model"].module.TransKun
        config = config_manager["Model"].config
        checkpoint = torch.load(
            str(package / "2.0.pt"), map_location="cpu", weights_only=False
        )
        model = model_class(conf=config)
        state = checkpoint.get("best_state_dict", checkpoint.get("state_dict"))
        model.load_state_dict(state, strict=False)
        model.eval()
        _PIANO_MODEL = model
        return model


def transcribe_piano(
    audio: np.ndarray,
    *,
    part: str = "piano",
    progress: ProgressCallback | None = None,
) -> tuple[list[NoteEvent], list[PedalEvent]]:
    """Transcribe piano audio (44.1 kHz, shape (samples, 2)) with Transkun."""
    import torch

    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    model = _load_piano_model()

    # Report progress by counting the model's internal 16-second windows.
    hop_seconds = model.segmentHopSizeInSecond
    pad_seconds = model.segmentSizeInSecond - hop_seconds
    total_seconds = audio.shape[0] / PIANO_SAMPLE_RATE + 2 * pad_seconds
    total_windows = max(1, math.ceil(total_seconds / hop_seconds))
    original = model.transcribeFrames
    counter = {"done": 0}

    def counting_transcribe_frames(*args, **kwargs):
        result = original(*args, **kwargs)
        counter["done"] += 1
        if progress:
            progress(min(1.0, counter["done"] / total_windows))
        return result

    try:
        model.transcribeFrames = counting_transcribe_frames
        with torch.inference_mode():
            raw_events = model.transcribe(torch.from_numpy(np.ascontiguousarray(audio)))
    except Exception as exc:
        LOGGER.exception("Piano transcription failed")
        raise TranscriptionError("The piano model could not process this audio.") from exc
    finally:
        model.transcribeFrames = original

    notes: list[NoteEvent] = []
    pedals: list[PedalEvent] = []
    for event in raw_events:
        if event.pitch > 0:
            notes.append(
                NoteEvent(
                    float(event.start),
                    float(event.end),
                    int(event.pitch),
                    int(np.clip(event.velocity, 1, 127)),
                    part,
                )
            )
        elif event.pitch == -64:
            pedals.append(PedalEvent(float(event.start), float(event.end)))
    notes.sort(key=lambda item: (item.start, item.pitch))
    return notes, pedals


def clean_piano_notes(
    notes: list[NoteEvent],
    *,
    relative_velocity: float = 0.0,
    min_seconds: float = 0.0,
) -> list[NoteEvent]:
    """Optionally drop "ghost" detections that are far quieter than the music.

    ``relative_velocity`` is a fraction of the recording's loud-note level
    (90th percentile), so quiet recordings are not thinned out.

    ``min_seconds`` must stay 0 for real piano recordings: staccato notes in
    fast music are often held for only 30 ms, and dropping "short" notes
    throws away a third of a virtuoso piece. It is only for piano-model output
    on non-piano sounds (song accompaniment), where short blips are errors.
    """
    if not notes:
        return []
    velocities = np.asarray([n.velocity for n in notes], dtype=float)
    floor = relative_velocity * float(np.percentile(velocities, 90))
    return [
        n
        for n in notes
        if n.velocity >= floor and (n.end - n.start) >= min_seconds
    ]


# ---------------------------------------------------------------------------
# Single melody line (CREPE + Viterbi note segmentation)
# ---------------------------------------------------------------------------

MELODY_SAMPLE_RATE = 16000
HOP_SECONDS = 0.01


def track_pitch(
    audio: np.ndarray,
    *,
    fmin: float,
    fmax: float,
    model: str = "full",
    decoder: str = "viterbi",
    progress: ProgressCallback | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (f0 in Hz, voicing confidence 0..1, loudness in dB) per 10 ms frame."""
    import torch
    import torchcrepe

    decode = {
        "viterbi": torchcrepe.decode.viterbi,
        "weighted_argmax": torchcrepe.decode.weighted_argmax,
    }[decoder]

    hop = int(MELODY_SAMPLE_RATE * HOP_SECONDS)
    total_frames = int(math.ceil(audio.size / hop))
    loudness = _frame_loudness(audio, hop, total_frames)
    f0 = np.zeros(total_frames)
    confidence = np.zeros(total_frames)

    # Only run the (slow) pitch network where there is sound. Separated vocal
    # and bass tracks are silent much of the time.
    active = loudness > max(float(loudness.max()) - 50.0, -75.0)
    margin = 10  # frames (100 ms) of context on each side
    regions = _active_regions(active, margin, max_frames=3000)
    work = sum(end - start for start, end in regions) or 1
    done = 0
    with torch.inference_mode():
        for start, end in regions:
            piece = audio[start * hop : end * hop]
            if piece.size < 4 * hop:
                continue
            signal = torch.from_numpy(np.ascontiguousarray(piece, dtype=np.float32))[None]
            pitch, periodicity = torchcrepe.predict(
                signal,
                MELODY_SAMPLE_RATE,
                hop_length=hop,
                fmin=fmin,
                fmax=fmax,
                model=model,
                decoder=decode,
                return_periodicity=True,
                batch_size=512,
                device="cpu",
            )
            count = min(end - start, pitch.shape[1])
            f0[start : start + count] = pitch[0, :count].numpy()
            confidence[start : start + count] = periodicity[0, :count].numpy()
            done += end - start
            if progress:
                progress(min(1.0, done / work))
    return f0, confidence, loudness


def _active_regions(active: np.ndarray, margin: int, max_frames: int) -> list[tuple[int, int]]:
    """Contiguous active frame ranges (with margins), split into bounded chunks."""
    if not np.any(active):
        return []
    padded = active.copy()
    indices = np.flatnonzero(active)
    for index in indices:
        padded[max(0, index - margin) : index + margin + 1] = True
    regions: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(padded):
        if value and start is None:
            start = index
        elif not value and start is not None:
            regions.append((start, index))
            start = None
    if start is not None:
        regions.append((start, padded.size))
    bounded: list[tuple[int, int]] = []
    for start, end in regions:
        while end - start > max_frames:
            bounded.append((start, start + max_frames))
            start += max_frames
        bounded.append((start, end))
    return bounded


def _frame_loudness(audio: np.ndarray, hop: int, frames: int) -> np.ndarray:
    window = hop * 4
    padded = np.pad(audio.astype(np.float64), (window // 2, window))
    values = np.empty(frames)
    for index in range(frames):
        segment = padded[index * hop : index * hop + window]
        values[index] = math.sqrt(float(np.mean(segment * segment)) + 1e-12)
    return 20 * np.log10(values + 1e-9)


def segment_notes(
    f0: np.ndarray,
    confidence: np.ndarray,
    loudness: np.ndarray,
    *,
    part: str,
    min_pitch: int,
    max_pitch: int,
    voicing_threshold: float = 0.45,
    min_note_seconds: float = 0.07,
    onset_times: np.ndarray | None = None,
    separated: bool = False,
) -> list[NoteEvent]:
    """Turn a pitch track into notes with a Viterbi pass over semitone states.

    Staying on the same note is free, changing note costs a penalty, so
    vibrato, scoops and small wobbles do not create extra notes.
    """
    frames = f0.size
    if frames == 0:
        return []
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = 69 + 12 * np.log2(np.where(f0 > 0, f0, np.nan) / 440.0)
    loud_floor = float(np.max(loudness)) - 45.0
    voiced = (confidence >= voicing_threshold) & np.isfinite(midi) & (loudness > loud_floor)

    states = np.arange(min_pitch, max_pitch + 1)
    n_states = states.size + 1  # last state = silence
    sigma = 0.45
    emission = np.empty((frames, n_states))
    safe_midi = np.where(voiced, midi, 0.0)
    diff = (safe_midi[:, None] - states[None, :]) / sigma
    weight = (0.5 + confidence)[:, None]
    note_cost = np.minimum(0.5 * diff * diff, 12.0) * weight
    emission[:, :-1] = np.where(voiced[:, None], note_cost, 1.2)
    emission[:, -1] = np.where(voiced, 3.0 * confidence + 1.0, 0.0)

    change_note = 5.0
    to_or_from_silence = 2.5
    cost = emission[0].copy()
    back = np.zeros((frames, n_states), dtype=np.int32)
    for index in range(1, frames):
        previous = cost
        best_note = int(np.argmin(previous[:-1]))
        best_note_cost = previous[best_note]
        new_cost = np.empty(n_states)
        # Note states: stay, come from another note, or come from silence.
        stay = previous[:-1]
        from_note = best_note_cost + change_note
        from_silence = previous[-1] + to_or_from_silence
        candidates = np.stack(
            [stay, np.full(n_states - 1, from_note), np.full(n_states - 1, from_silence)]
        )
        choice = np.argmin(candidates, axis=0)
        new_cost[:-1] = candidates[choice, np.arange(n_states - 1)]
        back[index, :-1] = np.where(
            choice == 0,
            np.arange(n_states - 1),
            np.where(choice == 1, best_note, n_states - 1),
        )
        # Silence: stay silent or end a note.
        if previous[-1] <= best_note_cost + to_or_from_silence:
            new_cost[-1] = previous[-1]
            back[index, -1] = n_states - 1
        else:
            new_cost[-1] = best_note_cost + to_or_from_silence
            back[index, -1] = best_note
        cost = new_cost + emission[index]

    path = np.empty(frames, dtype=np.int32)
    path[-1] = int(np.argmin(cost))
    for index in range(frames - 1, 0, -1):
        path[index - 1] = back[index, path[index]]

    notes: list[NoteEvent] = []
    levels: list[float] = []
    silence = n_states - 1
    index = 0
    while index < frames:
        state = path[index]
        end = index
        while end + 1 < frames and path[end + 1] == state:
            end += 1
        if state != silence:
            start_time = index * HOP_SECONDS
            end_time = (end + 1) * HOP_SECONDS
            values = midi[index : end + 1]
            values = values[np.isfinite(values)]
            pitch_value = int(states[state])
            if values.size:
                # Ignore the first/last 20% (scoops and releases).
                trim = int(values.size * 0.2)
                core = values[trim : values.size - trim] if values.size > 4 else values
                pitch_value = int(round(float(np.median(core))))
            level = float(np.max(loudness[index : end + 1]))
            notes.append(NoteEvent(start_time, end_time, pitch_value, 80, part))
            notes[-1].velocity = int(np.clip(64 + (level - loud_floor - 25) * 1.5, 30, 120))
            levels.append(level)
        index = end + 1

    _refine_boundaries(notes, midi, voiced)
    if levels:
        # Drop faint notes far below the line's normal level (e.g. other
        # instruments leaking into a separated vocal track).
        typical = float(np.percentile(levels, 75))
        notes = [n for n, level in zip(notes, levels) if level >= typical - 18.0]
    notes = _merge_short_notes(notes, min_note_seconds)
    onsets = onset_times if onset_times is not None else np.zeros(0)
    dip_times, dip_rises = _loudness_dips(loudness)
    onset_strength = np.full(onsets.size, 12.0)
    if separated:
        # Separated tracks wobble in loudness (tremolo, separation artefacts),
        # so only clear attacks count, and a repeated note needs an attack AND
        # a deep dip at the same moment (a sung consonant gives both).
        align = (onsets, onset_strength)
        deep = dip_times[dip_rises >= 8.0]
        split_times = np.array(
            [t for t in onsets if deep.size and np.min(np.abs(deep - t)) <= 0.04]
        )
    else:
        times = np.concatenate([onsets, dip_times])
        strengths = np.concatenate([onset_strength, dip_rises])
        order = np.argsort(times)
        align = (times[order], strengths[order])
        split_times = align[0][align[1] >= 6.0]
    notes = _align_starts_to_attacks(notes, align, min_note_seconds)
    return _split_repeated(notes, split_times, min_note_seconds)


def _refine_boundaries(notes: list[NoteEvent], midi: np.ndarray, voiced: np.ndarray) -> None:
    """Move note starts back to where the pitch actually changed.

    The note-tracking pass only switches notes once enough evidence has piled
    up, which places starts 50-150 ms late. Walking back while the raw pitch is
    already nearer the new note than the old one recovers the real moment.
    """
    max_frames = int(round(0.25 / HOP_SECONDS))
    for index, current in enumerate(notes):
        boundary = int(round(current.start / HOP_SECONDS))
        previous = notes[index - 1] if index else None
        if previous is not None and abs(previous.end - current.start) < 0.02:
            earliest = max(int(round(previous.start / HOP_SECONDS)) + 3, boundary - max_frames)

            def belongs(frame: int) -> bool:
                return bool(voiced[frame]) and abs(midi[frame] - current.pitch) < abs(
                    midi[frame] - previous.pitch
                )
        else:
            earliest = boundary - int(round(0.15 / HOP_SECONDS))
            if previous is not None:
                earliest = max(earliest, int(round(previous.end / HOP_SECONDS)))

            def belongs(frame: int) -> bool:
                return bool(voiced[frame]) and abs(midi[frame] - current.pitch) < 1.0

        frame = boundary
        while frame - 1 >= max(0, earliest) and belongs(frame - 1):
            frame -= 1
        if frame < boundary:
            current.start = frame * HOP_SECONDS
            if previous is not None and previous.end > current.start:
                previous.end = current.start


def _loudness_dips(loudness: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Attack times after loudness dips, with the size of the swell (dB).

    The attack is placed where the sound has risen halfway out of the dip,
    which is when a listener hears the new note (not at the quietest point).
    """
    from scipy.ndimage import maximum_filter1d, minimum_filter1d

    if loudness.size < 16:
        return np.zeros(0), np.zeros(0)
    curve = np.convolve(loudness, np.ones(3) / 3.0, mode="same")
    is_minimum = curve <= minimum_filter1d(curve, size=7)
    # Highest level in the following 150 ms (attacks take a moment to peak).
    ahead = maximum_filter1d(curve, size=15, origin=-7)
    rise = ahead - curve
    times: dict[int, float] = {}
    for frame in np.flatnonzero(is_minimum & (rise >= 3.0)):
        target = curve[frame] + rise[frame] / 2.0
        window = curve[frame : frame + 16]
        above = np.flatnonzero(window >= target)
        attack = int(frame + (above[0] if above.size else 0))
        times[attack] = max(times.get(attack, 0.0), float(rise[frame]))
    frames = np.asarray(sorted(times), dtype=int)
    return frames * HOP_SECONDS, np.asarray([times[f] for f in frames])


def _align_starts_to_attacks(
    notes: list[NoteEvent], attacks: tuple[np.ndarray, np.ndarray], min_seconds: float
) -> list[NoteEvent]:
    """Move each note start back to the attack that produced it.

    Pitch trackers register a new note only once the pitch has settled, often
    0.05-0.2 s after the attack (a singer's consonant, a scoop, a tongued
    note). Without this, the end of the previous note would swallow it.
    """
    times, strengths = attacks
    if times.size == 0:
        return notes
    result: list[NoteEvent] = []
    for item in notes:
        start = item.start
        earliest = start - 0.2
        if result:
            earliest = max(earliest, result[-1].start + min_seconds)
        # Pitch tracking only ever lags, so a start may move earlier, never
        # later, and only to a clear attack (weak wobbles are ignored).
        window = (times >= earliest) & (times <= start) & (strengths >= 5.0)
        if np.any(window):
            # The latest clear attack before the detected start is the real one.
            start = float(times[window].max())
        if result and result[-1].end > start:
            previous = result[-1]
            result[-1] = NoteEvent(previous.start, start, previous.pitch, previous.velocity, previous.part)
        if item.end - start >= min_seconds:
            result.append(NoteEvent(start, item.end, item.pitch, item.velocity, item.part))
    return result


def _split_repeated(
    notes: list[NoteEvent],
    strong: np.ndarray,
    min_seconds: float,
) -> list[NoteEvent]:
    """Split a long note where the same pitch is played/sung again.

    ``strong`` holds the times of clear re-attacks. The last 0.2 s of a note
    that runs straight into the next one is skipped: an attack there belongs
    to the next note (the pitch tracker reacts late).
    """
    result: list[NoteEvent] = []
    for index, item in enumerate(notes):
        limit = item.end - min_seconds
        following = notes[index + 1] if index + 1 < len(notes) else None
        if following is not None and following.start - item.end < 0.05:
            limit = min(limit, item.end - 0.2)
        cuts = strong[(strong > item.start + min_seconds) & (strong < limit)]
        start = item.start
        for cut in cuts:
            if cut - start < min_seconds:
                continue
            result.append(NoteEvent(start, float(cut), item.pitch, item.velocity, item.part))
            start = float(cut)
        result.append(NoteEvent(start, item.end, item.pitch, item.velocity, item.part))
    return result


def _merge_short_notes(notes: list[NoteEvent], min_seconds: float) -> list[NoteEvent]:
    """Drop blips and join a note with an identical neighbour across tiny gaps."""
    kept = [n for n in notes if n.end - n.start >= min_seconds]
    merged: list[NoteEvent] = []
    for item in kept:
        if (
            merged
            and merged[-1].pitch == item.pitch
            and item.start - merged[-1].end < 0.03
        ):
            merged[-1].end = item.end
            continue
        merged.append(item)
    return merged


def detect_onsets(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Onset times (seconds) of clear new attacks, used to split repeated notes."""
    import librosa

    envelope = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=160)
    if envelope.size == 0 or float(np.max(envelope)) <= 0:
        return np.zeros(0)
    onsets = librosa.onset.onset_detect(
        onset_envelope=envelope,
        sr=sample_rate,
        hop_length=160,
        units="time",
        backtrack=False,
        delta=0.25 * float(np.std(envelope)) + 0.1,
    )
    strengths = envelope[np.clip((onsets * sample_rate / 160).astype(int), 0, envelope.size - 1)]
    strong = strengths >= np.percentile(envelope, 90)
    return onsets[strong]


def transcribe_melody(
    audio: np.ndarray,
    *,
    part: str = "melody",
    low_hz: float = 65.0,
    high_hz: float = 1500.0,
    progress: ProgressCallback | None = None,
    split_repeated_notes: bool = True,
    model: str = "full",
    separated_track: bool = False,
) -> list[NoteEvent]:
    """Transcribe one melody line from 16 kHz mono audio.

    ``separated_track`` is for tracks split out of a full song: drum leakage
    is removed first and repeated notes need stronger evidence, so drum hits
    and separation wobbles are not mistaken for new notes.
    """
    if separated_track:
        import librosa

        audio = librosa.effects.harmonic(audio.astype(np.float32), margin=3.0)
    try:
        f0, confidence, loudness = track_pitch(
            audio, fmin=low_hz, fmax=high_hz, model=model, progress=progress
        )
    except ImportError as exc:  # pragma: no cover - install problem
        raise TranscriptionError("The pitch tracker (torchcrepe) is not installed.") from exc
    min_pitch = int(math.floor(69 + 12 * math.log2(low_hz / 440)))
    max_pitch = int(math.ceil(69 + 12 * math.log2(high_hz / 440)))
    onsets = detect_onsets(audio, MELODY_SAMPLE_RATE) if split_repeated_notes else None
    return segment_notes(
        f0,
        confidence,
        loudness,
        part=part,
        min_pitch=min_pitch,
        max_pitch=max_pitch,
        onset_times=onsets,
        separated=separated_track,
    )


# ---------------------------------------------------------------------------
# Other instruments (Basic Pitch)
# ---------------------------------------------------------------------------


def transcribe_general(
    audio_path: Path,
    *,
    part: str = "instrument",
    onset_threshold: float = 0.6,
    frame_threshold: float = 0.35,
    min_note_ms: float = 90.0,
) -> list[NoteEvent]:
    """Transcribe polyphonic non-piano audio with Spotify Basic Pitch."""
    try:
        from basic_pitch.inference import predict
    except ImportError as exc:  # pragma: no cover - install problem
        raise TranscriptionError("Basic Pitch is not installed.") from exc
    try:
        _, _, note_events = predict(
            str(audio_path),
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=min_note_ms,
            multiple_pitch_bends=False,
            melodia_trick=True,
        )
    except Exception as exc:
        LOGGER.exception("Basic Pitch failed for %s", audio_path)
        raise TranscriptionError("Basic Pitch could not transcribe this audio.") from exc
    notes = []
    for start, end, pitch_value, amplitude, _ in note_events or []:
        velocity = int(np.clip(round(float(amplitude) * 127), 1, 127))
        notes.append(NoteEvent(float(start), float(end), int(pitch_value), velocity, part))
    notes.sort(key=lambda item: (item.start, item.pitch))
    return notes
