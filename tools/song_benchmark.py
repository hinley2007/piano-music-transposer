"""Song-mode benchmark: a synthetic pop song (voice, piano, bass, drums).

The song is written as MIDI with a known melody, bass line and chord
progression, rendered to MP3 with MuseScore, and then transcribed in "song"
mode. Reported: melody notation F1, bass notation F1, chord-name accuracy per
beat, plus detected key / meter / tempo.

    python tools/song_benchmark.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pretty_midi  # noqa: E402

from config import MODE_SONG  # noqa: E402
from exporters import find_musescore_executable  # noqa: E402
from pipeline import AnalysisOptions, NotationOptions, analyze_audio, render_score  # noqa: E402

BPM = 96
BEAT = 60.0 / BPM
# G major: G - D - Em - C, one chord per bar, four times.
PROGRESSION = [(7, "maj", 43), (2, "maj", 38), (4, "min", 40), (0, "maj", 36)] * 4
CHORD_TONES = {"maj": (0, 4, 7), "min": (0, 3, 7)}
# (start beat within the 4-bar phrase, length in beats, MIDI pitch)
PHRASE_A = [
    (0, 1, 71), (1, 1, 71), (2, 1.5, 74), (3.5, 0.5, 72),
    (4, 1, 69), (5, 1, 69), (6, 2, 66),
    (8, 1, 67), (9, 0.5, 69), (9.5, 0.5, 71), (10, 1, 72), (11, 1, 71),
    (12, 1, 67), (13, 1, 64), (14, 2, 67),
]
PHRASE_B = [
    (0, 0.5, 74), (0.5, 0.5, 74), (1, 1, 76), (2, 1, 74), (3, 1, 71),
    (4, 1.5, 69), (5.5, 0.5, 71), (6, 1, 69), (7, 1, 66),
    (8, 1, 67), (9, 1, 71), (10, 1, 76), (11, 1, 74),
    (12, 2, 72), (14, 2, 71),
]
LEAD_IN_BEATS = 4  # one bar of drums + chords before the voice enters


def build_song() -> tuple[pretty_midi.PrettyMIDI, list[tuple[Fraction, int]], list[tuple[Fraction, int]], list[tuple[int, str]]]:
    midi = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    voice = pretty_midi.Instrument(program=53, name="Voice")  # "Voice Oohs"
    piano = pretty_midi.Instrument(program=0, name="Piano")
    bass = pretty_midi.Instrument(program=33, name="Bass")
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")

    melody_truth: list[tuple[Fraction, int]] = []
    bass_truth: list[tuple[Fraction, int]] = []
    chord_truth: list[tuple[int, str]] = []  # per beat: (root pitch class, quality)

    total_bars = len(PROGRESSION) + 1
    for bar in range(total_bars):
        chord_index = max(0, bar - 1) if bar > 0 else 0
        root, quality, bass_note = PROGRESSION[chord_index]
        start = bar * 4 * BEAT
        # Drums: kick 1 & 3, snare 2 & 4, closed hi-hat on eighths.
        for beat in range(4):
            time_ = start + beat * BEAT
            drums.notes.append(pretty_midi.Note(100, 36 if beat % 2 == 0 else 38, time_, time_ + 0.1))
        for eighth in range(8):
            time_ = start + eighth * BEAT / 2
            drums.notes.append(pretty_midi.Note(70, 42, time_, time_ + 0.05))
        if bar == 0:
            continue
        # Piano: block chords on every beat (middle register).
        tones = [60 + ((root + interval) % 12) for interval in CHORD_TONES[quality]]
        for beat in range(4):
            time_ = start + beat * BEAT
            for tone in tones:
                piano.notes.append(pretty_midi.Note(62, tone, time_, time_ + BEAT * 0.9))
            chord_truth.append((root, quality))
        # Bass: root on beats 1 and 3 (half notes).
        for beat in (0, 2):
            time_ = start + beat * BEAT
            bass.notes.append(pretty_midi.Note(95, bass_note, time_, time_ + BEAT * 1.9))
            bass_truth.append((Fraction(bar * 4 + beat), bass_note))

    phrases = [PHRASE_A, PHRASE_B, PHRASE_A, PHRASE_B]
    for index, phrase in enumerate(phrases):
        offset = LEAD_IN_BEATS + index * 16
        for beat, length, pitch_value in phrase:
            time_ = (offset + beat) * BEAT
            voice.notes.append(pretty_midi.Note(105, pitch_value, time_, time_ + length * BEAT * 0.95))
            melody_truth.append((Fraction(offset) + Fraction(beat).limit_denominator(8), pitch_value))

    midi.instruments.extend([voice, piano, bass, drums])
    return midi, melody_truth, bass_truth, chord_truth


def f1(truth: list, predicted: list) -> float:
    reference, found = Counter(truth), Counter(predicted)
    matches = sum(min(count, found[item]) for item, count in reference.items())
    if not matches:
        return 0.0
    precision, recall = matches / len(predicted), matches / len(truth)
    return 2 * precision * recall / (precision + recall)


def best_alignment(truth, predicted_raw) -> tuple[float, Fraction]:
    best = (0.0, Fraction(0))
    for step in range(-64, 65):
        shift = Fraction(step, 4)
        score = f1(truth, [(start + shift, p) for start, p in predicted_raw])
        if score > best[0]:
            best = (score, shift)
    return best


def main() -> int:
    musescore = find_musescore_executable()
    if musescore is None:
        print("MuseScore is required.")
        return 1
    midi, melody_truth, bass_truth, chord_truth = build_song()
    with tempfile.TemporaryDirectory() as folder:
        midi_path = Path(folder) / "song.mid"
        audio_path = Path(folder) / "song.mp3"
        midi.write(str(midi_path))
        subprocess.run([str(musescore), "-o", str(audio_path), str(midi_path)], capture_output=True, timeout=300)
        if len(sys.argv) > 1:
            Path(sys.argv[1]).write_bytes(audio_path.read_bytes())
        started = time.perf_counter()
        analysis = analyze_audio(audio_path, AnalysisOptions(mode=MODE_SONG))
        result = render_score(analysis, NotationOptions(title="song", layout="easy_piano", show_chords=True))
        elapsed = time.perf_counter() - started

    grid = result.grid
    melody = [(Fraction(n.start), n.pitch) for n in result.lines.get("melody", [])]
    melody_f1, shift = best_alignment(melody_truth, melody)
    bass_raw = [(Fraction(n.start), n.pitch) for n in result.lines.get("bass", [])]
    bass_f1 = f1(bass_truth, [(s + shift, p) for s, p in bass_raw]) if bass_raw else 0.0
    # Chord names per beat (beat positions aligned with the same shift).
    by_beat = {}
    for label in result.chords:
        for beat in range(int(label.start), int(label.end)):
            by_beat[Fraction(beat) + shift] = (label.root, label.quality)
    first_chord_beat = LEAD_IN_BEATS
    correct = sum(
        1 for index, chord in enumerate(chord_truth) if by_beat.get(Fraction(first_chord_beat + index)) == chord
    )
    print(
        f"melody F1={melody_f1:.3f} bass F1={bass_f1:.3f} chords={correct / len(chord_truth):.3f} "
        f"key={result.key.display_name} meter={result.time_signature} bpm={result.bpm:.0f} "
        f"(true G major, 4/4, {BPM}) shift={shift} beat_unit={grid.beat_unit} t={elapsed:.0f}s"
    )
    print("timings", analysis.timings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
