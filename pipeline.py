"""End-to-end pipeline: audio file -> analysis -> sheet music.

Stage 1, :func:`analyze_audio`, is slow (machine-learning models) and depends
only on the recording. Stage 2, :func:`render_score`, is fast and turns the
analysis into notation, so changing a notation option (time signature, key,
layout...) does not require processing the audio again.

Command line use (handy for testing and batches):

    python pipeline.py recording.mp3 --out output_folder --pdf
"""

from __future__ import annotations

import argparse
import logging
import math
import tempfile
import time
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from audio_utils import (
    AudioValidationError,
    decode_audio,
    estimate_tuning_semitones,
    retune,
    speed_ratio,
    to_mono,
)
from beat_tracking import track_beats
from chords import compute_chroma, estimate_chords
from config import (
    ACCOMPANIMENT_PRESETS,
    DEFAULT_SENSITIVITY,
    LAYOUT_EASY_PIANO,
    LAYOUT_LEAD_SHEET,
    LAYOUT_PIANO_VOCAL,
    MODE_MELODY,
    MODE_OTHER,
    MODE_PIANO,
    MODE_SONG,
    SENSITIVITY_PRESETS,
    TEMPO_AUTO,
    TEMPO_DOUBLE,
    TEMPO_HALF,
    TUNING_CORRECTION_THRESHOLD,
)
from keys import KeyEstimate, estimate_key, pitch_class_histogram
from music_types import ChordLabel, NoteEvent, PedalEvent, QuantizedNote
from performance import collapse_rolled_chords
from notation import (
    PartSpec,
    StaffSpec,
    build_score,
    build_staff_events,
    monophonic,
    notation_pulse,
    split_hands,
)
from rhythm import (
    BeatGrid,
    QuantizationResult,
    align_bar_lines,
    apply_time_signature,
    build_beat_grid,
    choose_beat_level,
    estimate_onset_lag,
    is_compound,
    quantize_notes,
)
from transcription import (
    NoNotesDetectedError,
    clean_piano_notes,
    transcribe_general,
    transcribe_melody,
    transcribe_piano,
)

LOGGER = logging.getLogger(__name__)


def quiet_library_warnings() -> None:
    """Hide harmless notices from the ML libraries (they look alarming in the terminal)."""
    import warnings

    for pattern in (
        "pkg_resources is deprecated",
        "torch.utils.checkpoint: the use_reentrant parameter",
        "None of the inputs have requires_grad=True",
        "`torch.jit.script` is deprecated",
    ):
        warnings.filterwarnings("ignore", message=f".*{pattern}.*")
    for name in ("huggingface_hub", "httpx", "httpx2", "urllib3"):
        logging.getLogger(name).setLevel(logging.ERROR)


SAMPLE_RATE = 44100
Progress = Callable[[float, str], None]

KEY_TONIC_CHOICES = ("C", "D-", "D", "E-", "E", "F", "F#", "G", "A-", "A", "B-", "B")


# ---------------------------------------------------------------------------
# Stage 1: analysis
# ---------------------------------------------------------------------------


@dataclass
class AnalysisOptions:
    mode: str = MODE_PIANO
    high_quality_separation: bool = False
    start_seconds: float | None = None
    end_seconds: float | None = None


@dataclass
class Analysis:
    mode: str
    parts: dict[str, list[NoteEvent]]
    pedals: list[PedalEvent]
    beat_times: np.ndarray
    downbeat_times: np.ndarray
    duration: float
    tuning_semitones: float
    chroma: np.ndarray
    chroma_times: np.ndarray
    downbeat_strength: np.ndarray = field(default_factory=lambda: np.zeros(0))
    timings: dict[str, float] = field(default_factory=dict)
    covers_ending: bool = True  # False when only a time range was transcribed


def _rescale(notes: Sequence[NoteEvent], ratio: float) -> list[NoteEvent]:
    if ratio == 1.0:
        return list(notes)
    return [replace(n, start=n.start / ratio, end=n.end / ratio) for n in notes]


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    import soxr

    return soxr.resample(audio, source_rate, target_rate).astype(np.float32)


def analyze_audio(
    path: Path,
    options: AnalysisOptions,
    progress: Progress | None = None,
) -> Analysis:
    """Run the slow part of the pipeline on an audio file."""
    report = progress or (lambda fraction, message: None)
    timings: dict[str, float] = {}
    clock = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        timings[name] = round(now - clock, 2)
        clock = now

    report(0.01, "Reading the audio file...")
    stereo = decode_audio(
        Path(path),
        SAMPLE_RATE,
        start_seconds=options.start_seconds,
        end_seconds=options.end_seconds,
    )
    duration = stereo.shape[0] / SAMPLE_RATE
    if duration < 2.0:
        raise AudioValidationError("The recording is too short (under 2 seconds).")
    mono = to_mono(stereo)
    lap("decode")

    report(0.04, "Checking the tuning...")
    tuning = estimate_tuning_semitones(_resample(mono, SAMPLE_RATE, 22050), 22050)
    use_tuning = abs(tuning) >= TUNING_CORRECTION_THRESHOLD
    ratio = speed_ratio(tuning) if use_tuning else 1.0
    tuned = retune(stereo, SAMPLE_RATE, tuning) if use_tuning else stereo
    lap("tuning")

    # Every detected note is kept here; which ones are written is decided in
    # the (fast) notation stage, so nothing is lost for good.
    parts: dict[str, list[NoteEvent]] = {}
    pedals: list[PedalEvent] = []
    chroma_audio = to_mono(tuned)

    if options.mode == MODE_PIANO:
        notes, pedals = transcribe_piano(
            tuned,
            progress=lambda f: report(0.1 + 0.85 * f, "Transcribing the piano notes..."),
        )
        parts["piano"] = notes
        lap("piano")
    elif options.mode == MODE_SONG:
        from separation import separate_stems

        report(0.1, "Separating vocals, drums, bass and other instruments...")
        stems = separate_stems(
            tuned,
            high_quality=options.high_quality_separation,
            progress=lambda f: report(0.1 + 0.4 * f, "Separating vocals, drums, bass and other instruments..."),
        )
        lap("separation")
        report(0.5, "Transcribing the vocal melody...")
        vocals = _resample(to_mono(stems["vocals"]), SAMPLE_RATE, 16000)
        parts["melody"] = transcribe_melody(
            vocals,
            part="melody",
            low_hz=70,
            high_hz=1200,
            progress=lambda f: report(0.5 + 0.12 * f, "Transcribing the vocal melody..."),
            separated_track=True,
        )
        lap("melody")
        report(0.62, "Transcribing the bass line...")
        bass = _resample(to_mono(stems["bass"]), SAMPLE_RATE, 16000)
        parts["bass"] = transcribe_melody(
            bass,
            part="bass",
            low_hz=32,
            high_hz=330,
            progress=lambda f: report(0.62 + 0.08 * f, "Transcribing the bass line..."),
            separated_track=True,
        )
        lap("bass")
        accompaniment_notes, pedals = transcribe_piano(
            stems["other"],
            part="accompaniment",
            progress=lambda f: report(0.7 + 0.25 * f, "Transcribing the accompaniment..."),
        )
        parts["accompaniment"] = accompaniment_notes
        chroma_audio = to_mono(stems["other"] + stems["bass"])
        lap("accompaniment")
    elif options.mode == MODE_MELODY:
        mono16 = _resample(to_mono(tuned), SAMPLE_RATE, 16000)
        parts["melody"] = transcribe_melody(
            mono16,
            part="melody",
            low_hz=40,
            high_hz=2000,
            progress=lambda f: report(0.1 + 0.85 * f, "Tracking the melody..."),
        )
        lap("melody")
    else:
        import soundfile

        report(0.1, "Transcribing the notes...")
        with tempfile.TemporaryDirectory(prefix="piano_music_transposer_") as folder:
            wav_path = Path(folder) / "input.wav"
            soundfile.write(str(wav_path), tuned, SAMPLE_RATE)
            parts["instrument"] = transcribe_general(wav_path, part="instrument")
        lap("general")

    # Undo the tiny speed change caused by tuning correction.
    parts = {name: _rescale(notes, ratio) for name, notes in parts.items()}
    pedals = [PedalEvent(p.start / ratio, p.end / ratio) for p in pedals]

    if not any(parts.values()):
        raise NoNotesDetectedError(
            "No notes were detected. Check that the recording is not silent."
        )

    # The beat tracker runs after the notes are known: where it follows a slow
    # pulse, the notes show how many beats fit in between.
    report(0.95, "Finding the beat and bar lines...")
    beat_notes = parts["piano"] if options.mode == MODE_PIANO else None
    beats, downbeats, strength = track_beats(stereo, SAMPLE_RATE, beat_notes)
    lap("beats")

    report(0.98, "Analysing the harmony...")
    chroma, chroma_times = compute_chroma(chroma_audio, SAMPLE_RATE)
    chroma_times = chroma_times / ratio
    lap("chroma")
    report(1.0, "Analysis complete.")
    return Analysis(
        mode=options.mode,
        parts=parts,
        pedals=pedals,
        beat_times=np.asarray(beats, dtype=float),
        downbeat_times=np.asarray(downbeats, dtype=float),
        duration=duration,
        tuning_semitones=float(tuning),
        chroma=chroma,
        chroma_times=chroma_times,
        downbeat_strength=np.asarray(strength, dtype=float),
        timings=timings,
        covers_ending=options.end_seconds is None,
    )


# ---------------------------------------------------------------------------
# Stage 2: notation
# ---------------------------------------------------------------------------


@dataclass
class NotationOptions:
    title: str = "Transcription"
    layout: str = LAYOUT_EASY_PIANO
    time_signature: str = "Auto"
    tempo_choice: str = TEMPO_AUTO
    transpose_semitones: int = 0
    target_tonic: str | None = None  # e.g. "G" or "B-" (keeps major/minor)
    show_chords: bool | None = None  # None = sensible default for the mode
    simplify_rhythm: bool = False
    sensitivity: str = DEFAULT_SENSITIVITY


@dataclass
class RenderResult:
    score: object
    key: KeyEstimate
    original_key: KeyEstimate
    bpm: float
    time_signature: str
    note_count: int
    bar_count: int
    grid: BeatGrid
    chords: list[ChordLabel]
    quantized: dict[str, list[QuantizedNote]]
    messages: list[str] = field(default_factory=list)
    # The melody / bass lines as written in the score (song and melody modes).
    lines: dict[str, list[QuantizedNote]] = field(default_factory=dict)
    detected_count: int = 0  # notes the models heard
    written_count: int = 0  # noteheads actually in the score
    tempo_text: str = ""  # tempo in the notated beat, e.g. "♩. = 62"


def _key_from_notes(notes: Sequence[QuantizedNote], *, use_ending: bool = True) -> KeyEstimate:
    pitches = [n.pitch for n in notes]
    weights = [min(float(n.duration), 2.0) + 0.25 for n in notes]
    final_bass = None
    if use_ending and notes:
        # Lowest note of the final chord (anything still sounding at the last onset).
        last_start = max(n.start for n in notes)
        closing = [n.pitch for n in notes if n.end > last_start or n.start >= last_start - 1]
        final_bass = min(closing) % 12
    return estimate_key(pitch_class_histogram(pitches, weights), final_bass=final_bass)


def _semitones_to_tonic(current: int, target_name: str) -> int:
    from keys import LETTER_PITCH_CLASS

    letter = target_name[0]
    alter = target_name.count("#") - target_name.count("-")
    target = (LETTER_PITCH_CLASS[letter] + alter) % 12
    shift = (target - current) % 12
    return shift - 12 if shift > 6 else shift


def _shift_octave_into_range(notes: list[QuantizedNote], low: int) -> list[QuantizedNote]:
    """Move a whole line up by octaves until its middle sits at/above ``low``."""
    if not notes:
        return notes
    median = float(np.median([n.pitch for n in notes]))
    shift = 0
    while median + shift < low:
        shift += 12
    return [n.copy(pitch=n.pitch + shift) for n in notes] if shift else notes


def _pedal_checker(pedals: Sequence[PedalEvent], grid: BeatGrid):
    if not pedals:
        return None
    starts = grid.to_beats([p.start for p in pedals])
    ends = grid.to_beats([p.end for p in pedals])
    intervals = sorted(zip(starts.tolist(), ends.tolist()))

    def pedal_down(beat_position: float) -> bool:
        return any(start <= beat_position < end for start, end in intervals)

    return pedal_down


def _select_notes(
    parts: dict[str, list[NoteEvent]], sensitivity: str
) -> dict[str, list[NoteEvent]]:
    """Apply the sensitivity choice to piano-model output."""
    piano = SENSITIVITY_PRESETS.get(sensitivity, SENSITIVITY_PRESETS[DEFAULT_SENSITIVITY])
    backing = ACCOMPANIMENT_PRESETS.get(sensitivity, ACCOMPANIMENT_PRESETS[DEFAULT_SENSITIVITY])
    selected = dict(parts)
    if "piano" in selected:
        selected["piano"] = clean_piano_notes(selected["piano"], **piano)
    if "accompaniment" in selected:
        selected["accompaniment"] = clean_piano_notes(selected["accompaniment"], **backing)
    return selected


def render_score(analysis: Analysis, options: NotationOptions) -> RenderResult:
    """Turn an analysis into a music21 score (fast; safe to call repeatedly)."""
    messages: list[str] = []
    detected = sum(len(notes) for notes in analysis.parts.values())
    selected = _select_notes(analysis.parts, options.sensitivity)
    # Rolled chords become one chord (written with an arpeggio sign).
    for name in ("piano", "accompaniment", "instrument"):
        if selected.get(name):
            selected[name] = collapse_rolled_chords(selected[name], analysis.beat_times)
    analysis = replace(analysis, parts=selected)
    all_notes = [n for notes in analysis.parts.values() for n in notes]
    if not all_notes:
        raise NoNotesDetectedError("No notes were detected in this recording.")

    # The beat level is chosen from the main musical material.
    level_notes = analysis.parts.get("melody") or all_notes
    if analysis.mode == MODE_PIANO:
        level_notes = all_notes
    grid = build_beat_grid(
        analysis.beat_times,
        analysis.downbeat_times,
        analysis.duration,
        downbeat_strength=analysis.downbeat_strength,
        notes=all_notes,
        # The piano model times notes to about 10 ms: beats can sit on them.
        lock_to_notes=analysis.mode == MODE_PIANO,
    )
    grid = choose_beat_level(grid, level_notes)
    if options.tempo_choice == TEMPO_HALF:
        halved = grid.halved()
        if halved is grid:
            messages.append("The tempo could not be halved for this time signature.")
        grid = halved
    elif options.tempo_choice == TEMPO_DOUBLE:
        grid = grid.doubled()
    if options.time_signature != "Auto":
        grid = apply_time_signature(grid, options.time_signature, [])
    # Pauses are not counted in strict beats: re-seat the bar lines after them.
    grid = align_bar_lines(grid, all_notes)

    # Pitch-tracked lines (melody, bass) register notes slightly late; shift
    # them back so they land on the beat they were played on.
    aligned: list[NoteEvent] = []
    for name, notes in analysis.parts.items():
        lag = estimate_onset_lag(notes, grid) if name in ("melody", "bass") else 0.0
        if abs(lag) >= 0.02:
            notes = [replace(n, start=n.start - lag, end=n.end - lag) for n in notes]
        aligned.extend(notes)
    all_notes = aligned

    subdivisions = (1, 2, 4) if options.simplify_rhythm else (1, 2, 3, 4, 6, 8)
    # Sung / pitch-tracked lines have fuzzier timing than a piano model's
    # onsets, so they are snapped with more tolerance (simpler rhythms).
    looseness_by_part = {
        "melody": 1.6 if analysis.mode == MODE_SONG else 1.3,
        "bass": 1.6,
    }
    groups: dict[float, list[NoteEvent]] = {}
    for item in all_notes:
        groups.setdefault(looseness_by_part.get(item.part, 1.0), []).append(item)
    quantized_notes: list[QuantizedNote] = []
    merged_subdivisions: dict[int, int] = {}
    for looseness in sorted(groups):
        partial = quantize_notes(
            groups[looseness], grid, allowed_subdivisions=subdivisions, looseness=looseness
        )
        quantized_notes.extend(partial.notes)
        for beat, value in partial.subdivisions.items():
            merged_subdivisions.setdefault(beat, value)
    quantized = QuantizationResult(quantized_notes, merged_subdivisions)
    if options.time_signature == "Auto":
        if grid.beat_unit == 1 and is_compound(quantized.subdivisions):
            grid = replace(grid, beat_unit=Fraction(3, 2))

    by_part: dict[str, list[QuantizedNote]] = {name: [] for name in analysis.parts}
    for item in quantized.notes:
        by_part.setdefault(item.part, []).append(item)

    # The final bass note only hints at the key if the real ending was analysed.
    original_key = _key_from_notes(quantized.notes, use_ending=analysis.covers_ending)
    shift = int(options.transpose_semitones)
    if options.target_tonic:
        shift = _semitones_to_tonic(original_key.tonic, options.target_tonic)
    key_estimate = original_key.transposed(shift) if shift else original_key
    if shift:
        by_part = {
            name: [n.copy(pitch=int(np.clip(n.pitch + shift, 0, 127))) for n in notes]
            for name, notes in by_part.items()
        }

    show_chords = options.show_chords
    if show_chords is None:
        show_chords = analysis.mode == MODE_SONG
    chords: list[ChordLabel] = []
    if show_chords:
        starts = [n.start for n in quantized.notes]
        ends = [n.end for n in quantized.notes]
        bass_notes = analysis.parts.get("bass") or [
            n for n in all_notes if n.pitch < 52
        ]
        chords = estimate_chords(
            analysis.chroma,
            analysis.chroma_times,
            grid,
            first_beat=int(math.floor(min(starts))),
            last_beat=int(math.ceil(max(ends))),
            bass_notes=bass_notes,
            key_estimate=original_key,
        )
        if shift:
            chords = [c.copy(root=(c.root + shift) % 12) for c in chords]

    pedal_down = _pedal_checker(analysis.pedals, grid)
    lines: dict[str, list[QuantizedNote]] = {}
    parts = _arrange(analysis.mode, options.layout, by_part, pedal_down, chords, grid, lines)
    bpm = grid.bpm
    numerator, denominator = grid.time_signature
    score = build_score(parts, grid, key_estimate, title=options.title, bpm=bpm)
    bar_count = len(score.parts[0].getElementsByClass("Measure")) if score.parts else 0
    note_count = sum(len(n) for n in by_part.values())
    return RenderResult(
        score=score,
        key=key_estimate,
        original_key=original_key,
        bpm=bpm,
        time_signature=f"{numerator}/{denominator}",
        note_count=note_count,
        bar_count=bar_count,
        grid=grid,
        chords=chords,
        quantized=by_part,
        messages=messages,
        lines=lines,
        detected_count=detected,
        written_count=count_written_notes(score),
        tempo_text=_tempo_text(grid),
    )


def _tempo_text(grid: BeatGrid) -> str:
    """Tempo in the beat a musician would count, e.g. '♩. = 62' in 6/8."""
    numerator, denominator = grid.time_signature
    pulse = notation_pulse(numerator, denominator)
    per_minute = grid.bpm * float(grid.beat_unit / pulse)
    symbol = {
        Fraction(3, 2): "♩.",
        Fraction(1): "♩",
        Fraction(1, 2): "♪",
        Fraction(2): "half note",
    }.get(pulse, "beat")
    return f"{symbol} = {per_minute:.0f}"


def count_written_notes(score) -> int:
    """Noteheads in the score, counting a tied note once (chord names excluded)."""
    from music21 import harmony

    total = 0
    for element in score.recurse().notes:
        if isinstance(element, harmony.Harmony):
            continue
        if element.tie is not None and element.tie.type in ("stop", "continue"):
            continue
        total += len(element.pitches)
    return total


def _piano_staves(
    notes: Sequence[QuantizedNote],
    pedal_down,
    placed: list[QuantizedNote] | None = None,
    *,
    beat_seconds: float = 0.5,
) -> list[StaffSpec]:
    hands = split_hands(notes, beat_seconds=beat_seconds)
    if placed is not None:
        placed.extend(hands)
    right = [n for n in hands if n.staff == 0]
    left = [n for n in hands if n.staff == 1]
    return [
        StaffSpec(build_staff_events(right, pedal_down=pedal_down, keyboard=True), "treble"),
        StaffSpec(build_staff_events(left, pedal_down=pedal_down, keyboard=True), "bass"),
    ]


def _lowest_line(notes: Sequence[QuantizedNote]) -> list[QuantizedNote]:
    """The lowest note at each start time (a simple bass line)."""
    lowest: dict[Fraction, QuantizedNote] = {}
    for item in notes:
        current = lowest.get(item.start)
        if current is None or item.pitch < current.pitch:
            lowest[item.start] = item
    return [lowest[start] for start in sorted(lowest)]


def _fill_bass_gaps(
    bass: list[QuantizedNote],
    accompaniment: Sequence[QuantizedNote],
    grid: BeatGrid,
) -> list[QuantizedNote]:
    """Use the accompaniment's lowest notes in bars where the bass track is silent.

    Separation sometimes leaves (some) bass notes in the "other" track, and
    voice + piano songs have no bass instrument at all.
    """
    def bar_of(position: Fraction) -> int:
        return math.floor((float(position) - grid.downbeat_phase) / grid.beats_per_bar)

    covered = {bar_of(n.start) for n in bass}
    low = [n for n in accompaniment if n.pitch < 55 and bar_of(n.start) not in covered]
    if not low:
        return bass
    return sorted(bass + _lowest_line(low), key=lambda n: n.start)


def _arrange(
    mode: str,
    layout_name: str,
    by_part: dict[str, list[QuantizedNote]],
    pedal_down,
    chords: list[ChordLabel],
    grid: BeatGrid,
    lines: dict[str, list[QuantizedNote]],
) -> list[PartSpec]:
    beat_seconds = grid.median_interval
    if mode == MODE_PIANO:
        placed: list[QuantizedNote] = []
        staves = _piano_staves(
            by_part.get("piano", []), pedal_down, placed, beat_seconds=beat_seconds
        )
        by_part["piano"] = placed  # keep the hand assignment for reporting
        return [PartSpec("Piano", staves, "Pno.", 0, chords)]

    if mode == MODE_MELODY:
        line = monophonic(by_part.get("melody", []))
        lines["melody"] = line
        return [PartSpec("Melody", [StaffSpec(build_staff_events(line), "auto")], "Mel.", 73, chords)]

    if mode == MODE_OTHER:
        notes = by_part.get("instrument", [])
        pitches = [n.pitch for n in notes] or [60]
        if max(pitches) - min(pitches) > 24:
            staves = _piano_staves(notes, None, beat_seconds=beat_seconds)
        else:
            staves = [StaffSpec(build_staff_events(notes), "auto")]
        return [PartSpec("Instrument", staves, "Inst.", 0, chords)]

    # Song mode.
    melody = monophonic(by_part.get("melody", []))
    bass = monophonic(by_part.get("bass", []))
    accompaniment = by_part.get("accompaniment", [])
    if len(melody) < 8 and accompaniment:
        # Instrumental song: use the top line of the accompaniment as the melody.
        melody = monophonic([n for n in accompaniment if n.pitch >= 60])
    if accompaniment:
        bass = _fill_bass_gaps(bass, accompaniment, grid)
    lines["melody"] = melody
    lines["bass"] = bass

    if layout_name == LAYOUT_LEAD_SHEET:
        return [PartSpec("Melody", [StaffSpec(build_staff_events(melody), "auto")], "Mel.", 0, chords)]

    if layout_name == LAYOUT_PIANO_VOCAL:
        piano_notes = list(accompaniment) + list(bass)
        return [
            PartSpec("Melody", [StaffSpec(build_staff_events(melody), "auto")], "Mel.", 53, chords),
            PartSpec(
                "Piano",
                _piano_staves(piano_notes, pedal_down, beat_seconds=beat_seconds),
                "Pno.",
                0,
            ),
        ]

    # Easy piano: melody in the right hand, bass line in the left hand.
    right = _shift_octave_into_range(melody, 62)
    left = _shift_octave_into_range(bass, 43)
    return [
        PartSpec(
            "Piano",
            [
                StaffSpec(build_staff_events(right), "treble"),
                # A bass note rings until the next one (up to two beats).
                StaffSpec(build_staff_events(left, legato_gap=Fraction(2)), "bass"),
            ],
            "Pno.",
            0,
            chords,
        )
    ]


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    from exporters import (
        export_pdf,
        find_musescore_executable,
        score_to_midi_bytes,
        score_to_musicxml_bytes,
    )

    parser = argparse.ArgumentParser(description="Transcribe audio to sheet music.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--mode", default=MODE_PIANO, choices=[MODE_PIANO, MODE_SONG, MODE_MELODY, MODE_OTHER])
    parser.add_argument("--layout", default=LAYOUT_EASY_PIANO)
    parser.add_argument("--out", type=Path, default=Path("output"))
    parser.add_argument("--time-signature", default="Auto")
    parser.add_argument("--tempo", default=TEMPO_AUTO)
    parser.add_argument("--transpose", type=int, default=0)
    parser.add_argument("--chords", action="store_true", help="Add chord names.")
    parser.add_argument("--title", default=None, help="Title printed on the sheet.")
    parser.add_argument(
        "--notes",
        default=DEFAULT_SENSITIVITY,
        choices=list(SENSITIVITY_PRESETS),
        help="Which detected notes to write.",
    )
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--end", type=float, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    quiet_library_warnings()
    last = {"message": ""}

    def show(fraction: float, message: str) -> None:
        if message != last["message"]:
            print(f"[{fraction * 100:5.1f}%] {message}", flush=True)
            last["message"] = message

    analysis = analyze_audio(
        args.audio,
        AnalysisOptions(mode=args.mode, start_seconds=args.start, end_seconds=args.end),
        show,
    )
    result = render_score(
        analysis,
        NotationOptions(
            title=args.title or args.audio.stem,
            layout=args.layout,
            time_signature=args.time_signature,
            tempo_choice=args.tempo,
            transpose_semitones=args.transpose,
            show_chords=True if args.chords else None,
            sensitivity=args.notes,
        ),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    xml_path = args.out / f"{args.audio.stem}.musicxml"
    xml_path.write_bytes(score_to_musicxml_bytes(result.score))
    (args.out / f"{args.audio.stem}.mid").write_bytes(score_to_midi_bytes(result.score))
    print(
        f"Key {result.key.display_name} | {result.time_signature} | {result.tempo_text} | "
        f"{result.written_count} of {result.detected_count} detected notes written | "
        f"{result.bar_count} bars | tuning {analysis.tuning_semitones * 100:+.0f} cents | "
        f"timings {analysis.timings}"
    )
    if args.pdf:
        musescore = find_musescore_executable()
        if musescore:
            export_pdf(xml_path, args.out / f"{args.audio.stem}.pdf", musescore)
    print(f"Wrote {xml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
