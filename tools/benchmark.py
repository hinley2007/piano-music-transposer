"""Accuracy benchmark for the transcription pipeline.

Known public-domain scores (from music21's built-in corpus) are rendered to
audio with MuseScore, transcribed, and compared with the original notation.

    python tools/benchmark.py                 # all pieces
    python tools/benchmark.py k545 joplin     # selected pieces
    python tools/benchmark.py --rubato        # add tempo drift like a human player
    python tools/benchmark.py --rubato --seed 3   # a different random tempo curve

Scores reported per piece:
  notation F1  - right pitch at the right beat position (what you read)
  detect F1    - right pitch at the right time (+-50 ms), before notation
  staff acc    - matched piano notes placed in the right hand's staff
  key / meter  - detected key and time signature against the original
"""

from __future__ import annotations

import argparse
import copy
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from music21 import chord, converter, corpus, key, meter, note, stream, tempo  # noqa: E402

from config import MODE_MELODY, MODE_PIANO, MODE_SONG  # noqa: E402
from exporters import find_musescore_executable  # noqa: E402
from pipeline import AnalysisOptions, NotationOptions, analyze_audio, render_score  # noqa: E402

PIECES = {
    "k545": ("mozart/k545/movement1_exposition.mxl", MODE_PIANO, None),
    "joplin": ("joplin/maple_leaf_rag.mxl", MODE_PIANO, 24),
    "polonaise": ("schumann_clara/polonaise_op1n1.mxl", MODE_PIANO, 24),
    "mazurka": ("chopin/mazurka06-2.krn", MODE_PIANO, 24),
    "chorale": ("bach/bwv66.6", MODE_PIANO, None),
    "lindenbaum": ("schubert/Lindenbaum.xml", MODE_SONG, 24),
    "folk": ("essenFolksong/han1.abc", MODE_MELODY, None),
}


@dataclass
class TruthNote:
    offset: Fraction  # quarter lengths from the start of the excerpt
    pitch: int
    staff: int
    seconds: float


def load_piece(name: str, max_bars: int | None) -> stream.Score:
    path, _, _ = PIECES[name]
    score = corpus.parse(path)
    if isinstance(score, stream.Opus):
        score = score.scores[0]
    try:
        # MuseScore plays repeats, so the answer key must contain them too.
        score = score.expandRepeats()
    except Exception:
        pass
    from music21 import bar, repeat, spanner

    for item in list(score.recurse().getElementsByClass((repeat.RepeatMark, bar.Repeat))):
        item.activeSite.remove(item)
    for item in list(score.recurse().getElementsByClass(spanner.RepeatBracket)):
        item.activeSite.remove(item)
    # MuseScore pads irregular bars in the middle of a piece (e.g. a pickup bar
    # repeated by expandRepeats), which would desynchronise audio and answer key.
    limit = max_bars or 10_000
    measures = list(score.parts[0].getElementsByClass(stream.Measure))
    for index, measure in enumerate(measures[1:], start=1):
        signature = measure.getContextByClass(meter.TimeSignature)
        if signature and measure.duration.quarterLength != signature.barDuration.quarterLength:
            limit = min(limit, index)
            break
    for part in score.parts:
        for index, measure in enumerate(list(part.getElementsByClass(stream.Measure))):
            if index >= limit:
                part.remove(measure)
    return score


def prepare_for_rendering(score: stream.Score, mode: str, rubato: bool, seed: int) -> stream.Score:
    """Copy the score, set sounds, and optionally add gradual tempo changes."""
    rendered = copy.deepcopy(score)
    from music21 import instrument

    for index, part in enumerate(rendered.parts):
        for old in part.recurse().getElementsByClass(instrument.Instrument):
            old.activeSite.remove(old)
        # MuseScore picks the sound from the part name, so name parts explicitly.
        if mode == MODE_PIANO:
            part.insert(0, instrument.Piano())
            part.partName = "Piano"
        elif mode == MODE_MELODY:
            # Flute does not transpose (a clarinet part would sound a tone lower).
            part.insert(0, instrument.Flute())
            part.partName = "Flute"
        elif index == 0:
            part.insert(0, instrument.Vocalist())
            part.partName = "Voice"
        else:
            part.insert(0, instrument.Piano())
            part.partName = "Piano"
    base = 100.0
    marks = list(rendered.flatten().getElementsByClass(tempo.MetronomeMark))
    if marks and marks[0].number:
        base = float(marks[0].number)
    # One explicit tempo source so music21 and MuseScore agree on timing.
    for part in rendered.parts:
        for old in list(part.recurse().getElementsByClass(tempo.TempoIndication)):
            old.activeSite.remove(old)
    from music21 import expressions

    for old in list(rendered.recurse().getElementsByClass(expressions.Fermata)):
        old.activeSite.remove(old)
    for element in rendered.recurse().notes:
        element.expressions = [e for e in element.expressions if not isinstance(e, expressions.Fermata)]
    top = rendered.parts[0]
    measures = list(top.getElementsByClass(stream.Measure))
    rng = random.Random(seed)
    current = base
    for index, measure in enumerate(measures):
        if index == 0 or rubato:
            if rubato and index:
                current = float(np.clip(current * rng.uniform(0.93, 1.07), base * 0.82, base * 1.18))
            measure.insert(0, tempo.MetronomeMark(number=round(current)))
    return rendered


def truth_notes(score: stream.Score) -> list[TruthNote]:
    stripped = score.stripTies()
    result: list[TruthNote] = []
    # A pickup bar starts part-way through a bar: measure from its bar line.
    first_measure = stripped.parts[0].getElementsByClass(stream.Measure).first()
    pickup = Fraction(first_measure.paddingLeft).limit_denominator(48) if first_measure else Fraction(0)
    seconds_map = {}
    for entry in stripped.flatten().secondsMap:
        element = entry["element"]
        seconds_map[id(element)] = entry["offsetSeconds"]
    for staff_index, part in enumerate(stripped.parts):
        for element in part.flatten().notes:
            if element.duration.isGrace or element.quarterLength == 0:
                continue
            offset = Fraction(element.getOffsetInHierarchy(stripped)).limit_denominator(48) + pickup
            seconds = seconds_map.get(id(element), float(offset) * 0.6)
            pitches = element.pitches if isinstance(element, chord.Chord) else [element.pitch]
            for p in pitches:
                result.append(TruthNote(offset, p.midi, staff_index, seconds))
    return result


def render_audio(score: stream.Score, folder: Path, musescore: Path) -> Path:
    xml = folder / "truth.musicxml"
    score.write("musicxml", fp=str(xml))
    audio = folder / "truth.mp3"
    subprocess.run([str(musescore), "-o", str(audio), str(xml)], capture_output=True, timeout=300)
    if not audio.is_file():
        raise RuntimeError("MuseScore did not render audio")
    return audio


def f1(matches: int, predicted: int, reference: int) -> float:
    if predicted == 0 or reference == 0:
        return 0.0
    precision, recall = matches / predicted, matches / reference
    return 0.0 if matches == 0 else 2 * precision * recall / (precision + recall)


def count_matches(truth: list[tuple[Fraction, int]], predicted: list[tuple[Fraction, int]]) -> int:
    reference = Counter(truth)
    found = Counter(predicted)
    return sum(min(count, found[item]) for item, count in reference.items())


def notation_score(truth: list[TruthNote], quantized: dict, grid, bar_ql: Fraction, origin: int):
    """Best F1 over small shifts and tempo-level scale factors."""
    # Positions measured from the first bar line, exactly as written in the score.
    predicted_raw = [
        ((Fraction(n.start) - origin) * grid.beat_unit, n.pitch, n.staff, n.part)
        for notes in quantized.values()
        for n in notes
    ]
    reference = [(t.offset, t.pitch) for t in truth]
    best = (0.0, Fraction(0), Fraction(1), 0)
    first_truth = min(t.offset for t in truth)
    first_pred = min(p[0] for p in predicted_raw)
    for scale in (Fraction(1), Fraction(1, 2), Fraction(2)):
        # Search around the offset that lines up the first notes (e.g. after an intro).
        base = Fraction(round((first_truth - first_pred * scale) * 4), 4)
        base -= base % bar_ql  # keep bar-line alignment testable
        for step in range(-32, 33):
            shift = base + Fraction(step, 4)
            predicted = [(p[0] * scale + shift, p[1]) for p in predicted_raw]
            matches = count_matches(reference, predicted)
            score = f1(matches, len(predicted), len(reference))
            if score > best[0]:
                best = (score, shift, scale, matches)
    score, shift, scale, matches = best
    # Staff agreement for matched notes (piano only).
    staff_of = {}
    for t in truth:
        staff_of.setdefault((t.offset, t.pitch), t.staff)
    agree = total = 0
    for offset, pitch_value, staff, part in predicted_raw:
        position = offset * scale + shift
        if (position, pitch_value) in staff_of and part == "piano":
            total += 1
            agree += int(staff_of[(position, pitch_value)] == staff)
    return score, scale, shift, (agree / total if total else float("nan"))


def detection_score(truth: list[TruthNote], analysis) -> float:
    import mir_eval

    events = [n for notes in analysis.parts.values() for n in notes]
    if not events:
        return 0.0
    ref_onsets = np.asarray(sorted(t.seconds for t in truth))
    est_onsets = np.asarray(sorted(n.start for n in events))
    # Align the global start offset (MuseScore adds a little lead-in).
    best_shift, best_count = 0.0, -1
    for shift in np.arange(-1.0, 1.0, 0.01):
        count = int(np.sum(np.min(np.abs(ref_onsets[:, None] - (est_onsets[None, :200] + shift)), axis=1) < 0.05))
        if count > best_count:
            best_shift, best_count = shift, count
    pad = 5.0  # keep every time positive for mir_eval
    ref_intervals = np.asarray([[t.seconds + pad, t.seconds + pad + 0.1] for t in truth])
    ref_pitches = np.asarray([440.0 * 2 ** ((t.pitch - 69) / 12) for t in truth])
    est_intervals = np.asarray([[n.start + best_shift + pad, n.end + best_shift + pad] for n in events])
    est_pitches = np.asarray([440.0 * 2 ** ((n.pitch - 69) / 12) for n in events])
    _, _, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches, est_intervals, est_pitches, onset_tolerance=0.05, offset_ratio=None
    )
    return float(f)


def run_piece(name: str, rubato: bool, musescore: Path, seed: int = 0) -> dict:
    path, mode, max_bars = PIECES[name]
    score = load_piece(name, max_bars)
    rendered = prepare_for_rendering(score, mode, rubato, seed)
    truth = truth_notes(rendered)
    if mode == MODE_SONG:
        truth = [t for t in truth if t.staff == 0]  # evaluate the vocal melody
    true_key = score.analyze("key")
    signatures = list(score.recurse().getElementsByClass(meter.TimeSignature))
    true_meter = signatures[0].ratioString if signatures else "?"
    with tempfile.TemporaryDirectory() as folder:
        audio = render_audio(rendered, Path(folder), musescore)
        started = time.perf_counter()
        analysis = analyze_audio(audio, AnalysisOptions(mode=mode))
        result = render_score(analysis, NotationOptions(title=name, layout="lead_sheet"))
        elapsed = time.perf_counter() - started
    if mode == MODE_SONG:
        quantized = {"melody": result.quantized.get("melody", [])}
    else:
        quantized = result.quantized
    bar_ql = Fraction(result.grid.beats_per_bar) * result.grid.beat_unit
    # Measure positions from the start of the whole piece, like the answer key.
    all_starts = [n.start for notes in result.quantized.values() for n in notes]
    origin = result.grid.bar_start_beat(float(min(all_starts)))
    notation_f1, scale, shift, staff_acc = notation_score(
        truth, quantized, result.grid, bar_ql, origin
    )
    if mode == MODE_SONG:
        from types import SimpleNamespace

        detect_f1 = detection_score(truth, SimpleNamespace(parts={"melody": analysis.parts.get("melody", [])}))
    else:
        detect_f1 = detection_score(truth, analysis)
    same_key = (true_key.tonic.pitchClass == result.original_key.tonic and true_key.mode == result.original_key.mode)
    return {
        "piece": name,
        "notes": len(truth),
        "detect_f1": detect_f1,
        "notation_f1": notation_f1,
        "staff_acc": staff_acc,
        "scale": str(scale),
        "bar_aligned": (shift % bar_ql == 0),
        "key": f"{result.original_key.display_name} ({'ok' if same_key else 'true ' + true_key.name})",
        "meter": f"{result.time_signature} (true {true_meter})",
        "bpm": round(result.bpm),
        "seconds": round(elapsed, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pieces", nargs="*", default=list(PIECES))
    parser.add_argument("--rubato", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="which random tempo curve --rubato uses")
    args = parser.parse_args()
    musescore = find_musescore_executable()
    if musescore is None:
        print("MuseScore is required for the benchmark.")
        return 1
    rows = []
    for name in args.pieces:
        try:
            row = run_piece(name, args.rubato, musescore, args.seed)
        except Exception as exc:  # keep going with other pieces
            print(f"{name}: FAILED ({exc!r})")
            continue
        rows.append(row)
        print(
            f"{row['piece']:<11} notes={row['notes']:<4} detect={row['detect_f1']:.3f} "
            f"notation={row['notation_f1']:.3f} staff={row['staff_acc']:.3f} "
            f"scale={row['scale']:<3} bars_ok={row['bar_aligned']!s:<5} "
            f"key={row['key']} meter={row['meter']} bpm={row['bpm']} t={row['seconds']}s",
            flush=True,
        )
    if rows:
        print(
            f"MEAN detect={np.mean([r['detect_f1'] for r in rows]):.3f} "
            f"notation={np.mean([r['notation_f1'] for r in rows]):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
