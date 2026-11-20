"""Arrange quantized notes on staves and build a clean music21 score.

This module decides which hand plays each piano note, turns overlapping notes
into readable chords, fills gaps with sensibly split rests, and writes bars
with correct ties so that MusicXML viewers show tidy notation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from typing import Callable, Sequence

import numpy as np
from music21 import (
    articulations,
    chord,
    clef,
    duration,
    expressions,
    harmony,
    instrument,
    key,
    layout,
    metadata,
    meter,
    note,
    pitch,
    stream,
    tempo,
    tie,
)

from keys import KeyEstimate, Spelling, pitch_class_name, spell_pitch
from music_types import ChordLabel, QuantizedNote
from rhythm import BeatGrid

MAX_NOTES_PER_HAND = 5


# ---------------------------------------------------------------------------
# Piano hand assignment
# ---------------------------------------------------------------------------


def _span_cost(pitches: Sequence[int]) -> float:
    """How hard it is for one hand to cover these pitches at once."""
    if len(pitches) < 2:
        return 0.0
    span = max(pitches) - min(pitches)
    if span <= 9:
        return 0.0
    if span <= 12:
        return (span - 9) * 0.06
    if span <= 16:
        return 0.18 + (span - 12) * 0.45
    return 6.0 + (span - 16)


@dataclass(frozen=True)
class _HandState:
    last: tuple[int, ...]  # pitches of the hand's most recent chord/note
    last_time: float  # beat position of that chord
    held: tuple[tuple[int, float], ...]  # (pitch, release beat) still sounding
    trail: tuple[tuple[int, int, float], ...] = ()  # (low, high, beat) of recent chords


@dataclass(frozen=True)
class _Hypothesis:
    cost: float
    right: _HandState
    left: _HandState
    recent: tuple[tuple[int, int, float], ...]  # (pitch, hand, start) of recent notes
    choice: int  # split index chosen for the latest group
    parent: "_Hypothesis | None"


_DEFAULT_RIGHT = _HandState((67,), -1e9, ())
_DEFAULT_LEFT = _HandState((50,), -1e9, ())
BEAM_WIDTH = 24
SANDWICH_MAX_LEAP = 26  # semitones

# Weights of the hand model. Times are in seconds, distances in semitones.
HAND_MODEL = {
    "move": 1.0,  # per octave between a note and the hand's previous notes...
    "move_cap": 12.0,  # ...but a leap is a leap: no extra cost beyond this
    "reach": 16.0,  # what a hand covers without travelling (a tenth)
    "far": 1.0,  # per octave beyond reach from where the hand was just before
    "memory": 1.2,  # how long (s) a hand's earlier position still matters
    "register": 0.08,  # per semitone: right hand very low, left hand very high
    "held": 0.35,  # adding notes to a hand that is holding others
    "continuity": 0.5,  # a stepwise line changing hands
    "sandwich": 2.5,  # see _sandwich_cost
    "crossing": 0.6,  # left hand above the right hand
}


def _length_mismatch_cost(notes: Sequence[QuantizedNote], beat_seconds: float) -> float:
    """A long melody note and a quick accompaniment note are rarely one chord."""
    if len(notes) < 2:
        return 0.0
    lengths = [n.raw_end - n.raw_start for n in notes if n.raw_end > n.raw_start]
    if len(lengths) < 2:
        return 0.0
    longest, shortest = max(lengths), min(lengths)
    # Allow for a little unevenness in a chord's release. An eighth under
    # a quarter is still evidence for two voices when the grid changes level.
    if longest > 1.65 * shortest and (longest - shortest) * beat_seconds > 0.08:
        return 0.85
    return 0.0


def _hand_cost(
    pitches: Sequence[int],
    state: _HandState,
    start: float,
    hand: int,
    beat_seconds: float,
) -> float:
    """Effort for one hand to play ``pitches`` at beat ``start``."""
    if not pitches:
        return 0.0
    model = HAND_MODEL
    idle = (start - state.last_time) * beat_seconds
    # A hand that has been resting can move anywhere cheaply.
    if idle <= 1.0:
        weight = 1.0
    elif idle >= 3.0:
        weight = 0.3
    else:
        weight = 1.0 - 0.7 * (idle - 1.0) / 2.0
    cost = 0.0
    for p in pitches:
        move = min(abs(p - q) for q in state.last)
        cost += model["move"] * weight * min(move, model["move_cap"]) / 12.0
        if hand == 0 and p < 53:
            cost += model["register"] * (53 - p)
        if hand == 1 and p > 69:
            cost += model["register"] * (p - 69)
    # Where was this hand a moment ago? Hopping between two distant registers
    # costs every time, however cheap each landing looks on its own.
    low, high = min(pitches), max(pitches)
    far = 0.0
    for trail_low, trail_high, trail_time in state.trail:
        age = (start - trail_time) * beat_seconds
        if age <= 0 or age >= model["memory"]:
            continue
        excess = max(high - trail_low, trail_high - low) - model["reach"]
        if excess > 0:
            far = max(far, (1.0 - age / model["memory"]) * excess / 12.0)
    cost += model["far"] * far
    held = [p for p, release in state.held if (release - start) * beat_seconds > 0.12]
    if held:
        # Playing new notes while still holding others is possible but rarer.
        cost += model["held"] * len(pitches)
    cost += _span_cost(list(pitches) + held)
    # A hand has five fingers: spread bigger chords over both hands.
    if len(pitches) > MAX_NOTES_PER_HAND:
        cost += 1.5 * (len(pitches) - MAX_NOTES_PER_HAND)
    return cost


def _continuity_cost(
    pitches: Sequence[int],
    hand: int,
    recent: Sequence[tuple[int, int, float]],
    start: float,
    beat_seconds: float,
) -> float:
    """Stepwise runs usually stay in one hand."""
    cost = 0.0
    for p in pitches:
        for other_pitch, other_hand, other_start in reversed(recent):
            if (start - other_start) * beat_seconds > 0.28:
                break
            if other_start < start and 0 < abs(p - other_pitch) <= 2:
                if other_hand != hand:
                    cost += HAND_MODEL["continuity"]
                break
    return cost


def _sandwich_cost(
    left: Sequence[int],
    right: Sequence[int],
    recent: Sequence[tuple[int, int, float]],
    start: float,
    beat_seconds: float,
) -> float:
    """A note between two quick strikes of the same pitch shares that pitch's hand.

    Broken octaves, Alberti basses and "bell" figures (melody note, repeated
    high note, melody note, ...) are one-hand patterns, even when the moving
    note lies in the other hand's register.
    """
    # The last four onset groups, newest first: middle, strike, middle, strike.
    groups: list[list[tuple[int, int, float]]] = []
    for entry in reversed(recent):
        if groups and groups[-1][0][2] == entry[2]:
            groups[-1].append(entry)
        elif len(groups) < 4:
            groups.append([entry])
        else:
            break
    if len(groups) < 4:
        return 0.0
    times = [start] + [group[0][2] for group in groups]
    if any(not 0 < (later - earlier) * beat_seconds <= 0.3 for later, earlier in zip(times, times[1:])):
        return 0.0
    middle, strike, _, earlier_strike = groups
    middle_pitches = sorted(r[0] for r in middle)
    cost = 0.0
    for hand, pitches in ((1, left), (0, right)):
        # The repeated note is a single key (a "bell", the top of a broken
        # octave), not a chord: repeated chords are an accompaniment that the
        # other hand plays around.
        if len(pitches) != 1:
            continue
        for p in pitches:
            # An established figure: this is at least the third strike of the
            # same key by the same hand, on every other note.
            if [r[0] for r in strike if r[1] == hand] != [p]:
                continue
            if [r[0] for r in earlier_strike if r[1] == hand] != [p]:
                continue
            # The repeated note must lie clearly on one side of what came in
            # between; its partner is the nearest of those notes, and must
            # stand apart from any chord struck with it.
            if p > middle_pitches[-1]:
                partner = max(middle, key=lambda r: r[0])
                exposed = len(middle) == 1 or middle_pitches[-1] - middle_pitches[-2] >= 5
            elif p < middle_pitches[0]:
                partner = min(middle, key=lambda r: r[0])
                exposed = len(middle) == 1 or middle_pitches[1] - middle_pitches[0] >= 5
            else:
                continue
            # Beyond about two octaves it is two hands answering each other
            # (a bass note between right-hand chords), not one hand's figure.
            if not exposed or abs(p - partner[0]) > SANDWICH_MAX_LEAP:
                continue
            if partner[1] != hand:
                cost += HAND_MODEL["sandwich"]
    return cost


def split_hands(
    notes: Sequence[QuantizedNote], *, beat_seconds: float = 0.5
) -> list[QuantizedNote]:
    """Return copies of piano notes with ``staff`` 0 (right hand) or 1 (left hand).

    A beam search considers every way of splitting each chord between the
    hands and keeps the most plausible histories. Costs reward small hand
    movements, playable stretches, keeping stepwise runs and alternating
    figures in one hand, and not adding notes to a hand that is still holding
    a long note. ``beat_seconds`` (length of one grid beat) converts the
    notes' beat positions into real time.
    """
    by_start: dict[Fraction, list[QuantizedNote]] = defaultdict(list)
    for item in notes:
        by_start[item.start].append(item)
    starts = sorted(by_start)
    groups = [sorted(by_start[start], key=lambda item: item.pitch) for start in starts]
    model = HAND_MODEL

    beam = [_Hypothesis(0.0, _DEFAULT_RIGHT, _DEFAULT_LEFT, (), -1, None)]
    for start, group in zip(starts, groups):
        position = float(start)
        pitches = [item.pitch for item in group]
        duration_costs = [
            _length_mismatch_cost(group[:split], beat_seconds)
            + _length_mismatch_cost(group[split:], beat_seconds)
            for split in range(len(group) + 1)
        ]
        candidates: dict[tuple, _Hypothesis] = {}
        for hypothesis in beam:
            for split in range(len(group) + 1):
                left, right = pitches[:split], pitches[split:]
                cost = hypothesis.cost
                cost += _hand_cost(left, hypothesis.left, position, 1, beat_seconds)
                cost += _hand_cost(right, hypothesis.right, position, 0, beat_seconds)
                cost += _continuity_cost(left, 1, hypothesis.recent, position, beat_seconds)
                cost += _continuity_cost(right, 0, hypothesis.recent, position, beat_seconds)
                cost += _sandwich_cost(left, right, hypothesis.recent, position, beat_seconds)
                cost += duration_costs[split]
                # Crossing: left hand above the right hand's current notes.
                if (
                    left
                    and hypothesis.right.last
                    and max(left) > max(hypothesis.right.last) + 2
                    and (position - hypothesis.right.last_time) * beat_seconds < 1.0
                ):
                    cost += model["crossing"]
                new_right = hypothesis.right
                new_left = hypothesis.left
                if right:
                    held = tuple(
                        (p, r) for p, r in hypothesis.right.held if r > position
                    ) + tuple((item.pitch, item.raw_end) for item in group[split:])
                    trail = hypothesis.right.trail + ((min(right), max(right), position),)
                    new_right = _HandState(tuple(right), position, held[-6:], trail[-6:])
                if left:
                    held = tuple(
                        (p, r) for p, r in hypothesis.left.held if r > position
                    ) + tuple((item.pitch, item.raw_end) for item in group[:split])
                    trail = hypothesis.left.trail + ((min(left), max(left), position),)
                    new_left = _HandState(tuple(left), position, held[-6:], trail[-6:])
                recent = hypothesis.recent + tuple(
                    (p, 1 if index < split else 0, position) for index, p in enumerate(pitches)
                )
                candidate = _Hypothesis(cost, new_right, new_left, recent[-20:], split, hypothesis)
                signature = (
                    new_right.last,
                    new_right.last_time,
                    new_left.last,
                    new_left.last_time,
                )
                existing = candidates.get(signature)
                if existing is None or candidate.cost < existing.cost:
                    candidates[signature] = candidate
        beam = sorted(candidates.values(), key=lambda h: h.cost)[:BEAM_WIDTH]

    best = beam[0]
    choices: list[int] = []
    node: _Hypothesis | None = best
    while node is not None and node.choice >= 0:
        choices.append(node.choice)
        node = node.parent
    choices.reverse()

    # Every note is placed in one hand or the other: nothing is ever dropped.
    result: list[QuantizedNote] = []
    for group, split in zip(groups, choices):
        result.extend(item.copy(staff=1) for item in group[:split])
        result.extend(item.copy(staff=0) for item in group[split:])
    return result


# ---------------------------------------------------------------------------
# Staff events (chords + durations without overlaps)
# ---------------------------------------------------------------------------


@dataclass
class StaffEvent:
    start: Fraction  # beats
    end: Fraction  # beats
    pitches: list[int]
    velocity: int = 80
    rolled: bool = False  # written with an arpeggio sign
    staccato: bool = False  # played much shorter than written


def build_staff_events(
    notes: Sequence[QuantizedNote],
    *,
    pedal_down: Callable[[float], bool] | None = None,
    legato_gap: Fraction = Fraction(1, 2),
    pedal_gap: Fraction = Fraction(2),
    keyboard: bool = False,
) -> list[StaffEvent]:
    """Merge simultaneous notes into chords and make durations readable.

    Within one staff a chord lasts until the next chord starts (no overlaps).
    Short gaps are closed (legato), and gaps under a held sustain pedal are
    closed too, which matches how piano music is normally written.

    With ``keyboard`` set, a detached note is written out to the end of its
    beat and marked staccato, the way piano music is printed, instead of as a
    tiny note followed by a row of rests.
    """
    by_start: dict[Fraction, list[QuantizedNote]] = defaultdict(list)
    for item in notes:
        by_start[item.start].append(item)
    starts = sorted(by_start)
    events: list[StaffEvent] = []
    for index, start in enumerate(starts):
        group = by_start[start]
        pitches = sorted({item.pitch for item in group})
        natural_end = max(item.end for item in group)
        raw_start = min(item.raw_start for item in group)
        raw_end = max(item.raw_end for item in group)
        velocity = max(item.velocity for item in group)
        following = starts[index + 1] if index + 1 < len(starts) else None
        pedalled = bool(pedal_down and pedal_down(raw_end))
        end = natural_end
        if following is not None:
            if end >= following:
                end = following
            else:
                gap = following - end
                limit = pedal_gap if pedalled else legato_gap
                if gap <= limit:
                    end = following
        if end <= start:
            end = start + Fraction(1, 4)
        staccato = False
        if keyboard:
            # Fill the beat the note starts in (or one whole beat from a beat).
            beat_end = Fraction(math.floor(start) + 1)
            target = beat_end if following is None else min(beat_end, following)
            if target > end:
                end = target
            sounded = max(raw_end - raw_start, 0.0)
            staccato = not pedalled and sounded <= 0.4 * float(end - start)
        rolled = sum(1 for item in group if item.rolled) >= 2 and len(pitches) >= 2
        events.append(StaffEvent(start, end, pitches, velocity, rolled, staccato))
    return events


def monophonic(notes: Sequence[QuantizedNote]) -> list[QuantizedNote]:
    """Keep only the highest note at each start (for a single melody line)."""
    best: dict[Fraction, QuantizedNote] = {}
    for item in notes:
        current = best.get(item.start)
        if current is None or item.pitch > current.pitch:
            best[item.start] = item
    return [best[start] for start in sorted(best)]


# ---------------------------------------------------------------------------
# Readable splitting of notes and rests
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4096)
def is_simple_duration(quarter_length: Fraction) -> bool:
    """True if one notehead (at most one dot, at most a triplet) can show it."""
    if quarter_length <= 0:
        return False
    try:
        value = duration.Duration(quarter_length)
    except Exception:
        return False
    if value.type in ("complex", "inexpressible", "zero"):
        return False
    if value.dots > 1:
        return False
    if len(value.tuplets) > 1:
        return False
    if value.tuplets and value.tuplets[0].numberNotesActual != 3:
        return False
    return True


def is_binary_position(value: Fraction) -> bool:
    """Positions on a plain (non-triplet) 32nd-note grid."""
    return (Fraction(value) * 8).denominator == 1


SUB_BEAT_FRACTIONS = (
    Fraction(3, 4),
    Fraction(2, 3),
    Fraction(1, 2),
    Fraction(3, 8),
    Fraction(1, 3),
    Fraction(1, 4),
    Fraction(3, 16),
    Fraction(1, 6),
    Fraction(1, 8),
    Fraction(1, 12),
    Fraction(1, 16),
)
COMPOUND_PULSE = Fraction(3, 2)  # dotted quarter: the beat of 6/8, 9/8, 12/8
EIGHTH = Fraction(1, 2)


def notation_pulse(numerator: int, denominator: int) -> Fraction:
    """The beat (in quarter notes) that notes and rests are grouped by."""
    if denominator == 8 and numerator in (6, 9, 12):
        return COMPOUND_PULSE
    return Fraction(4, denominator)


def split_readable(
    start: Fraction, end: Fraction, beat_ql: Fraction, bar_ql: Fraction
) -> list[tuple[Fraction, Fraction]]:
    """Split a span inside one bar into note values aligned with the beats.

    ``beat_ql`` is the notated pulse: a dotted quarter in compound meters,
    where each pulse is made of three eighths.
    """
    compound = beat_ql == COMPOUND_PULSE
    pieces: list[tuple[Fraction, Fraction]] = []
    position = start
    while position < end:
        if position == 0 and end == bar_ql and is_simple_duration(bar_ql):
            pieces.append((position, end))
            break
        beat_index = math.floor(position / beat_ql)
        beat_start = beat_index * beat_ql
        beat_end = beat_start + beat_ql
        length: Fraction | None = None
        if position == beat_start and end - position >= beat_ql:
            whole_beats = int((end - position) // beat_ql)
            for count in (4, 3, 2, 1):
                if count > whole_beats:
                    continue
                if count in (3, 4) and position != 0:
                    continue
                if count == 2 and beat_index % 2 != 0:
                    continue
                candidate = beat_ql * count
                if is_simple_duration(candidate):
                    length = candidate
                    break
        if length is None and compound:
            limit = min(end, beat_end)
            inside = position - beat_start
            if (inside / EIGHTH).denominator == 1:
                # On one of the pulse's three eighths: a quarter or an eighth.
                for candidate in (Fraction(1), EIGHTH):
                    if position + candidate <= limit:
                        length = candidate
                        break
            if length is None:
                # Inside an eighth: finish that eighth as in a simple meter.
                boundary = (math.floor(position / EIGHTH) + 1) * EIGHTH
                stop = min(limit, boundary)
                pieces.extend(split_readable(position, stop, EIGHTH, bar_ql))
                position = stop
                continue
        if length is None:
            limit = min(end, beat_end)
            inside = position - beat_start
            for fraction in SUB_BEAT_FRACTIONS:
                candidate = beat_ql * fraction
                if position + candidate > limit:
                    continue
                unit = beat_ql / fraction.denominator
                if (inside / unit).denominator != 1:
                    continue
                if not is_simple_duration(candidate):
                    continue
                length = candidate
                break
        if length is None:
            length = min(end, beat_end) - position
        pieces.append((position, position + length))
        position += length
    return pieces


# ---------------------------------------------------------------------------
# Score building
# ---------------------------------------------------------------------------


@dataclass
class StaffSpec:
    events: list[StaffEvent]
    clef_name: str = "auto"  # treble, bass, treble8vb, bass8vb, auto


@dataclass
class PartSpec:
    name: str
    staves: list[StaffSpec]
    abbreviation: str = ""
    midi_program: int = 0  # General MIDI program (0 = piano)
    chords: list[ChordLabel] = field(default_factory=list)


def _make_clef(name: str) -> clef.Clef:
    return {
        "treble": clef.TrebleClef,
        "bass": clef.BassClef,
        "treble8vb": clef.Treble8vbClef,
        "bass8vb": clef.Bass8vbClef,
    }[name]()


def choose_clef_name(events: Sequence[StaffEvent], requested: str) -> str:
    if requested != "auto":
        return requested
    pitches = [p for event in events for p in event.pitches]
    if not pitches:
        return "treble"
    median = float(np.median(pitches))
    if median >= 59:
        return "treble"
    if median >= 50:
        return "treble8vb"
    return "bass"


CLEF_CHANGE_COST = 12.0  # a clef change is worth one note five ledger lines out
_LETTER_STEP = (0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6)  # staff step of each pitch class
_STAFF_LINES = {"treble": (30, 38), "bass": (18, 26)}  # bottom and top line (C4 = 28)


def ledger_lines(midi: int, clef_name: str) -> int:
    """How many ledger lines a pitch needs in a treble or bass clef."""
    step = 7 * (midi // 12 - 1) + _LETTER_STEP[midi % 12]
    bottom, top = _STAFF_LINES[clef_name]
    if step < bottom:
        return (bottom - step) // 2
    if step > top:
        return (step - top) // 2
    return 0


def bar_clefs(
    events: Sequence[StaffEvent],
    start_clef: str,
    bar_count: int,
    to_ql: Callable[[Fraction], Fraction],
    bar_ql: Fraction,
) -> list[str]:
    """Clef for every bar of a treble/bass staff.

    The sequence of clefs is the one that needs the fewest ledger lines over
    the whole staff, where every change of clef costs as much as a handful of
    ledger lines. The left hand climbing into the treble for a passage, or
    dropping to low bass octaves, changes the clef; a single stray note does
    not.
    """
    if start_clef not in ("treble", "bass"):
        return [start_clef] * bar_count
    names = ("treble", "bass")
    cost = np.zeros((bar_count, 2))
    for event in events:
        position = to_ql(event.start)
        bar = int(position // bar_ql)
        if 0 <= bar < bar_count:
            for p in event.pitches:
                # One or two ledger lines read easily; five or six do not.
                cost[bar, 0] += ledger_lines(p, "treble") ** 2 / 2.0
                cost[bar, 1] += ledger_lines(p, "bass") ** 2 / 2.0

    # Viterbi over the two clefs.
    total = cost[0].copy()
    total[1 - names.index(start_clef)] += 0.5  # the staff's usual clef wins a tie
    back = np.zeros((bar_count, 2), dtype=np.int64)
    for bar in range(1, bar_count):
        new_total = np.empty(2)
        for clef_index in range(2):
            stay = total[clef_index]
            switch = total[1 - clef_index] + CLEF_CHANGE_COST
            if stay <= switch:
                new_total[clef_index], back[bar, clef_index] = stay, clef_index
            else:
                new_total[clef_index], back[bar, clef_index] = switch, 1 - clef_index
        total = new_total + cost[bar]
    choice = int(np.argmin(total))
    sequence = [choice]
    for bar in range(bar_count - 1, 0, -1):
        choice = int(back[bar, choice])
        sequence.append(choice)
    sequence.reverse()
    return [names[index] for index in sequence]


def _make_pitch(spelled: Spelling) -> pitch.Pitch:
    return pitch.Pitch(spelled.name)


def _spell_events(events: Sequence[StaffEvent], key_estimate: KeyEstimate) -> list[list[pitch.Pitch]]:
    spelled: list[list[pitch.Pitch]] = []
    for index, event in enumerate(events):
        following = None
        if (
            len(event.pitches) == 1
            and index + 1 < len(events)
            and len(events[index + 1].pitches) == 1
        ):
            following = events[index + 1].pitches[0]
        pitches = [
            _make_pitch(spell_pitch(p, key_estimate, next_pitch=following))
            for p in event.pitches
        ]
        spelled.append(pitches)
    return spelled


def chord_figure(label: ChordLabel, key_estimate: KeyEstimate) -> str:
    """music21 chord-symbol text such as 'B-m7' or 'D/F#'."""
    root = pitch_class_name(label.root, key_estimate)
    suffix = {
        "maj": "",
        "min": "m",
        "7": "7",
        "maj7": "maj7",
        "min7": "m7",
        "dim": "dim",
        "aug": "+",
        "sus4": "sus4",
        "sus2": "sus2",
        "hdim7": "m7b5",
    }.get(label.quality, "")
    figure = f"{root}{suffix}"
    if label.bass is not None and label.bass != label.root:
        figure += "/" + pitch_class_name(label.bass, key_estimate)
    return figure


def _chord_symbol(label: ChordLabel, key_estimate: KeyEstimate) -> harmony.ChordSymbol | None:
    figure = chord_figure(label, key_estimate)
    try:
        symbol = harmony.ChordSymbol(figure)
    except Exception:
        return None
    symbol.writeAsChord = False
    return symbol


def build_score(
    parts: Sequence[PartSpec],
    grid: BeatGrid,
    key_estimate: KeyEstimate,
    *,
    title: str,
    bpm: float,
) -> stream.Score:
    """Create a music21 Score with bars, ties, rests, and chord symbols."""
    all_events = [event for part in parts for staff in part.staves for event in staff.events]
    if not all_events:
        raise ValueError("There are no notes to write.")
    first_beat = min(event.start for event in all_events)
    last_beat = max(event.end for event in all_events)
    origin = grid.bar_start_beat(float(first_beat))
    beat_ql = grid.beat_unit
    bar_beats = grid.beats_per_bar
    bar_ql = beat_ql * bar_beats
    total_beats = last_beat - origin
    bar_count = max(1, math.ceil(total_beats / bar_beats))
    numerator, denominator = grid.time_signature
    # The grid may count eighth notes while the music is felt (and grouped)
    # in dotted quarters, as in 6/8.
    pulse_ql = notation_pulse(numerator, denominator)

    def to_ql(beat_position: Fraction) -> Fraction:
        return (Fraction(beat_position) - origin) * beat_ql

    score = stream.Score()
    score.insert(0, metadata.Metadata())
    score.metadata.title = title
    score.metadata.composer = "Transcribed with Piano Music Transposer"

    first_staff = True
    for part_spec in parts:
        staff_streams: list[stream.Part] = []
        grand = len(part_spec.staves) > 1
        for staff_index, staff in enumerate(part_spec.staves):
            container: stream.Part = stream.PartStaff() if grand else stream.Part()
            container.id = f"{part_spec.name}-{staff_index}".replace(" ", "_")
            container.partName = part_spec.name
            container.partAbbreviation = part_spec.abbreviation or part_spec.name[:4]
            inst = instrument.Instrument()
            inst.instrumentName = part_spec.name
            inst.midiProgram = part_spec.midi_program
            container.insert(0, inst)

            measures = [stream.Measure(number=index + 1) for index in range(bar_count)]
            opening = measures[0]
            first_clef = choose_clef_name(staff.events, staff.clef_name)
            clef_names = bar_clefs(staff.events, first_clef, bar_count, to_ql, bar_ql)
            previous_clef = None
            for index, clef_name in enumerate(clef_names):
                if clef_name != previous_clef:
                    measures[index].insert(0, _make_clef(clef_name))
                    previous_clef = clef_name
            opening.insert(0, key.Key(key_estimate.tonic_name, key_estimate.mode))
            opening.insert(0, meter.TimeSignature(f"{numerator}/{denominator}"))
            if first_staff:
                # Tempo in the notated pulse (e.g. dotted quarter = 59 in 6/8).
                mark = tempo.MetronomeMark(
                    number=int(round(bpm * float(beat_ql / pulse_ql))),
                    referent=duration.Duration(pulse_ql),
                )
                opening.insert(0, mark)

            _write_staff(measures, staff.events, key_estimate, to_ql, pulse_ql, bar_ql)
            for line in ottava_lines(measures, clef_names):
                container.insert(0, line)

            if staff_index == 0 and part_spec.chords:
                for label in part_spec.chords:
                    position = to_ql(label.start)
                    if position < 0:
                        continue
                    bar = int(position // bar_ql)
                    if bar >= bar_count:
                        continue
                    symbol = _chord_symbol(label, key_estimate)
                    if symbol is not None:
                        measures[bar].insert(position - bar * bar_ql, symbol)

            for index, measure in enumerate(measures):
                container.insert(index * bar_ql, measure)
            staff_streams.append(container)
            score.insert(0, container)
            first_staff = False
        if grand:
            group = layout.StaffGroup(
                staff_streams,
                name=part_spec.name,
                abbreviation=part_spec.abbreviation,
                symbol="brace",
                barTogether=True,
            )
            score.insert(0, group)
    return score


def _write_staff(
    measures: list[stream.Measure],
    events: Sequence[StaffEvent],
    key_estimate: KeyEstimate,
    to_ql: Callable[[Fraction], Fraction],
    beat_ql: Fraction,
    bar_ql: Fraction,
) -> None:
    ordered = sorted(events, key=lambda event: event.start)
    spelled = _spell_events(ordered, key_estimate)
    total_ql = bar_ql * len(measures)
    cursor = Fraction(0)
    for event, pitches in zip(ordered, spelled):
        start = max(to_ql(event.start), Fraction(0))
        end = min(to_ql(event.end), total_ql)
        if end <= start:
            continue
        if start > cursor:
            _place_rests(measures, cursor, start, beat_ql, bar_ql)
        start = max(start, cursor)
        if end <= start:
            continue
        _place_sound(measures, start, end, pitches, event, beat_ql, bar_ql)
        cursor = end
    if cursor < total_ql:
        _place_rests(measures, cursor, total_ql, beat_ql, bar_ql)


OTTAVA_START = 91  # G6: four ledger lines above the treble staff
OTTAVA_KEEP = 84  # C6: stay under the line while the music remains this high
OTTAVA_FLOOR = 72  # nothing below C5 goes under an 8va line


def ottava_lines(measures: Sequence[stream.Measure], clef_names: Sequence[str]) -> list:
    """8va lines over stretches of very high treble-clef bars (fewer ledger lines).

    Notes keep their sounding pitch; the line only changes how they are drawn.
    """
    from music21 import spanner

    def bar_range(measure: stream.Measure) -> tuple[int, int] | None:
        pitches = [p.midi for element in measure.notes for p in element.pitches]
        return (min(pitches), max(pitches)) if pitches else None

    lines = []
    run: list = []
    pending: list = []  # notes of bars that only continue a run if a high bar follows

    def close() -> None:
        nonlocal run, pending
        if len(run) >= 2:
            lines.append(spanner.Ottava(*run, type="8va", transposing=False))
        run, pending = [], []

    for measure, clef_name in zip(measures, clef_names):
        span = bar_range(measure)
        if span is None:
            continue  # an empty bar neither starts nor ends a line
        low, high = span
        elements = list(measure.notes)
        treble = clef_name == "treble"
        if treble and low >= OTTAVA_FLOOR and high >= OTTAVA_START:
            run.extend(pending)
            pending = []
            run.extend(elements)
        elif run and treble and low >= OTTAVA_FLOOR and high >= OTTAVA_KEEP:
            pending.extend(elements)
        else:
            close()
    close()
    return lines


def _bar_segments(start: Fraction, end: Fraction, bar_ql: Fraction):
    position = start
    while position < end:
        bar = int(position // bar_ql)
        bar_end = (bar + 1) * bar_ql
        segment_end = min(end, bar_end)
        yield bar, position - bar * bar_ql, segment_end - bar * bar_ql
        position = segment_end


def _place_rests(measures, start, end, beat_ql, bar_ql) -> None:
    for bar, local_start, local_end in _bar_segments(start, end, bar_ql):
        if local_start == 0 and local_end == bar_ql:
            rest = note.Rest(quarterLength=bar_ql)
            rest.fullMeasure = True
            measures[bar].insert(0, rest)
            continue
        for piece_start, piece_end in split_readable(local_start, local_end, beat_ql, bar_ql):
            measures[bar].insert(piece_start, note.Rest(quarterLength=piece_end - piece_start))


def _place_sound(measures, start, end, pitches, event: StaffEvent, beat_ql, bar_ql) -> None:
    compound = beat_ql == COMPOUND_PULSE
    pieces: list[tuple[int, Fraction, Fraction]] = []
    for bar, local_start, local_end in _bar_segments(start, end, bar_ql):
        single = (
            is_binary_position(local_start)
            and is_binary_position(local_end)
            and is_simple_duration(local_end - local_start)
        )
        if single and compound:
            # In 6/8 a note must not hide the dotted-quarter pulse: it may
            # cross a pulse only if it starts on one.
            first_pulse = math.floor(local_start / beat_ql)
            last_pulse = math.ceil(local_end / beat_ql) - 1
            if last_pulse > first_pulse and local_start != first_pulse * beat_ql:
                single = False
        if single:
            pieces.append((bar, local_start, local_end))
        else:
            for piece_start, piece_end in split_readable(local_start, local_end, beat_ql, bar_ql):
                pieces.append((bar, piece_start, piece_end))
    count = len(pieces)
    for index, (bar, piece_start, piece_end) in enumerate(pieces):
        length = piece_end - piece_start
        # Fresh Pitch objects per notehead: accidental display is stored on them.
        fresh = [pitch.Pitch(p.nameWithOctave) for p in pitches]
        if len(fresh) == 1:
            element = note.Note(fresh[0], quarterLength=length)
        else:
            element = chord.Chord(fresh, quarterLength=length)
        element.volume.velocity = int(np.clip(event.velocity, 1, 127))
        if count > 1:
            if index == 0:
                element.tie = tie.Tie("start")
            elif index == count - 1:
                element.tie = tie.Tie("stop")
            else:
                element.tie = tie.Tie("continue")
        if index == 0:
            if event.rolled and len(fresh) > 1:
                element.expressions.append(expressions.ArpeggioMark("normal"))
            # A dot on a note of an eighth or longer played clearly detached.
            if event.staccato and count == 1 and length >= EIGHTH:
                element.articulations.append(articulations.Staccato())
        measures[bar].insert(piece_start, element)
