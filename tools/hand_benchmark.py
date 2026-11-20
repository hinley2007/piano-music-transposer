"""Benchmark for splitting piano notes between the two staves (no audio needed).

Real two-staff piano scores from music21's corpus are stripped of their staff
information; ``split_hands`` must put each note back on the right staff. A
"bell" study in the style of Liszt's La Campanella (right hand leaping between
a melody and a repeated high note over sparse left-hand chords) is included
because simple nearest-hand rules fail on it.

    python tools/hand_benchmark.py
"""

from __future__ import annotations

import sys
import warnings
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from music21 import chord, corpus, stream, tempo  # noqa: E402

from music_types import QuantizedNote  # noqa: E402
from notation import split_hands  # noqa: E402

# (corpus path, index of the right-hand staff, index of the left-hand staff)
PIECES = {
    "mozart_k545": ("mozart/k545/movement1_exposition.mxl", 0, 1),
    "joplin_rag": ("joplin/maple_leaf_rag.mxl", 0, 1),
    "polonaise1": ("schumann_clara/polonaise_op1n1.mxl", 0, 1),
    "polonaise2": ("schumann_clara/polonaise_op1n2.mxl", 0, 1),
    "polonaise3": ("schumann_clara/polonaise_op1n3.mxl", 0, 1),
    "polonaise4": ("schumann_clara/polonaise_op1n4.mxl", 0, 1),
    "chopin_mazurka": ("chopin/mazurka06-2.krn", 0, 1),
    "cpe_bach": ("cpebach/h186.mxl", 0, 1),
    "schoenberg_2": ("schoenberg/opus19/movement2.mxl", 0, 1),
    "schoenberg_6": ("schoenberg/opus19/movement6.mxl", 0, 1),
    "lindenbaum_pf": ("schubert/Lindenbaum.xml", 1, 2),
    "dichterliebe_pf": ("schumann_robert/dichterliebe_no2.xml", 1, 2),
}


def score_notes(path: str, right: int, left: int) -> tuple[list[QuantizedNote], float]:
    score = corpus.parse(path)
    if isinstance(score, stream.Opus):
        score = score.scores[0]
    stripped = score.stripTies()
    marks = list(stripped.flatten().getElementsByClass(tempo.MetronomeMark))
    bpm = float(marks[0].getQuarterBPM()) if marks and marks[0].getQuarterBPM() else 100.0
    notes: list[QuantizedNote] = []
    parts = list(stripped.parts)
    for staff, index in ((0, right), (1, left)):
        for element in parts[index].flatten().notes:
            if element.duration.isGrace or element.quarterLength == 0:
                continue
            start = Fraction(element.getOffsetInHierarchy(stripped)).limit_denominator(96)
            end = start + Fraction(element.quarterLength).limit_denominator(96)
            pitches = element.pitches if isinstance(element, chord.Chord) else [element.pitch]
            for p in pitches:
                notes.append(
                    QuantizedNote(start, end, p.midi, 80, "piano", staff, float(start), float(end))
                )
    return notes, 60.0 / bpm


def bell_study() -> tuple[list[QuantizedNote], float]:
    """Right hand: melody note then a high 'bell' on every sixteenth pair.
    Left hand: a chord on eighths 1 and 4 of each 6/8 bar."""
    notes: list[QuantizedNote] = []
    bell = 99  # D#7
    melody = [87, 87, 85, 83, 83, 82, 80, 79, 80, 82, 83, 85, 87, 85, 83, 82, 80, 79, 80, 80, 82, 83, 80, 75]
    chords = [(56, 63, 71), (59, 68, 75), (51, 63, 67), (56, 63, 71)]
    eighth = Fraction(1, 2)
    for index, pitch_value in enumerate(melody):
        start = index * eighth
        notes.append(QuantizedNote(start, start + Fraction(1, 8), pitch_value, 80, "piano", 0, float(start), float(start) + 0.125))
        bell_start = start + Fraction(1, 4)
        notes.append(QuantizedNote(bell_start, bell_start + Fraction(1, 8), bell, 70, "piano", 0, float(bell_start), float(bell_start) + 0.125))
        if index % 3 == 0:
            for tone in chords[(index // 3) % len(chords)]:
                notes.append(QuantizedNote(start, start + Fraction(1, 4), tone, 70, "piano", 1, float(start), float(start) + 0.25))
    return notes, 0.58  # seconds per quarter note (eighth = 0.29 s)


def leap_study() -> tuple[list[QuantizedNote], float]:
    """Later Campanella texture: two-octave right-hand leaps, left-hand octave + chord."""
    notes: list[QuantizedNote] = []
    sixteenth = Fraction(1, 4)
    melody = [75, 76, 78, 80, 78, 76, 75, 73, 71, 73, 75, 76]
    for index, pitch_value in enumerate(melody * 2):
        start = index * 2 * sixteenth
        notes.append(QuantizedNote(start, start + sixteenth, pitch_value, 80, "piano", 0, float(start), float(start) + 0.25))
        top = start + sixteenth
        notes.append(QuantizedNote(top, top + sixteenth, pitch_value + 24, 80, "piano", 0, float(top), float(top) + 0.25))
        if index % 3 == 0:
            notes.append(QuantizedNote(start, start + Fraction(1, 2), 44, 80, "piano", 1, float(start), float(start) + 0.5))
        if index % 3 == 1:
            for tone in (56, 59, 63):
                notes.append(QuantizedNote(start, start + Fraction(1, 2), tone, 70, "piano", 1, float(start), float(start) + 0.5))
    return notes, 0.58


def evaluate(notes: list[QuantizedNote], beat_seconds: float) -> tuple[float, int]:
    truth = {}
    for item in notes:
        truth.setdefault((item.start, item.pitch), []).append(item.staff)
    blank = [item.copy(staff=0) for item in notes]
    try:
        placed = split_hands(blank, beat_seconds=beat_seconds)
    except TypeError:  # older signature
        placed = split_hands(blank)
    right = total = 0
    for item in placed:
        expected = truth[(item.start, item.pitch)]
        total += 1
        right += int(item.staff in expected)
    return right / max(1, total), total


def main() -> int:
    rows = []
    for name, (path, right, left) in PIECES.items():
        try:
            notes, beat_seconds = score_notes(path, right, left)
        except Exception as exc:
            print(f"{name}: skipped ({exc!r})")
            continue
        accuracy, count = evaluate(notes, beat_seconds)
        rows.append((name, accuracy, count))
    for name, builder in (("bell_study", bell_study), ("leap_study", leap_study)):
        notes, beat_seconds = builder()
        accuracy, count = evaluate(notes, beat_seconds)
        rows.append((name, accuracy, count))
    for name, accuracy, count in rows:
        print(f"{name:<16} notes={count:<5} staff accuracy={accuracy:.3f}")
    mean = sum(a for _, a, _ in rows) / len(rows)
    weighted = sum(a * c for _, a, c in rows) / sum(c for _, _, c in rows)
    print(f"MEAN per piece={mean:.3f}  per note={weighted:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
