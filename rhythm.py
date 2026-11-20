"""Turn performance timing (seconds) into notated rhythm (beats).

The key idea: notes are not snapped to a fixed tempo. Instead every note time
is measured against the beats that were actually tracked in the recording, so
speeding up or slowing down does not break the notation. Each beat then gets
its own subdivision (straight 8ths, 16ths, triplets...) chosen by comparing
how well each grid explains the notes against how complicated it is to read.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Iterable, Sequence

import numpy as np

from music_types import NoteEvent, QuantizedNote

# Cost of each subdivision (higher = harder to read). Units are comparable to a
# squared timing error measured in "typical timing jitters".
SUBDIVISION_PRIOR = {1: 0.0, 2: 1.0, 4: 2.0, 3: 3.0, 8: 4.5, 6: 4.5}
# Weight of the "don't invent uneven rhythms" term (spacing errors).
SPACING_WEIGHT = 0.5
# Binary (2, 4, 8) and ternary (3, 6) grids; 1 fits both.
SUBDIVISION_FAMILY = {1: 0, 2: 1, 4: 1, 8: 1, 3: 2, 6: 2}
FAMILY_SWITCH_COST = 1.0
COLLISION_COST = 6.0
DEFAULT_SUBDIVISIONS = (1, 2, 3, 4, 6, 8)


# ---------------------------------------------------------------------------
# Beat grid
# ---------------------------------------------------------------------------


@dataclass
class BeatGrid:
    """Tracked beat times plus the meter used to draw bar lines."""

    beat_times: np.ndarray
    beats_per_bar: int = 4
    downbeat_phase: int = 0
    beat_unit: Fraction = Fraction(1)  # quarter lengths per beat
    downbeat_times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # Share of beats the tracker thought were bar starts (near 1 = "one-in-a-bar").
    downbeat_density: float = 0.0

    def __post_init__(self) -> None:
        self.beat_times = np.asarray(self.beat_times, dtype=float)
        self.downbeat_times = np.asarray(self.downbeat_times, dtype=float)
        if self.beat_times.size < 2:
            raise ValueError("A beat grid needs at least two beats.")
        self.beats_per_bar = max(1, int(self.beats_per_bar))
        self.downbeat_phase = int(self.downbeat_phase) % self.beats_per_bar

    # -- tempo -------------------------------------------------------------
    @property
    def median_interval(self) -> float:
        return float(np.median(np.diff(self.beat_times)))

    @property
    def bpm(self) -> float:
        """Beats per minute at the tracked beat level."""
        return 60.0 / self.median_interval

    @property
    def is_compound(self) -> bool:
        return self.beat_unit == Fraction(3, 2)

    @property
    def time_signature(self) -> tuple[int, int]:
        if self.beat_unit == Fraction(3, 2):
            return self.beats_per_bar * 3, 8
        if self.beat_unit == Fraction(1, 2):
            return self.beats_per_bar, 8
        if self.beat_unit == Fraction(2):
            return self.beats_per_bar, 2
        return self.beats_per_bar, 4

    # -- conversions -------------------------------------------------------
    def _edge_intervals(self) -> tuple[float, float]:
        diffs = np.diff(self.beat_times)
        return float(np.median(diffs[:4])), float(np.median(diffs[-4:]))

    def to_beats(self, seconds: Iterable[float] | float) -> np.ndarray:
        """Convert times in seconds to (fractional) beat positions."""
        times = np.atleast_1d(np.asarray(seconds, dtype=float))
        beats = self.beat_times
        indices = np.arange(beats.size, dtype=float)
        result = np.interp(times, beats, indices)
        first_ibi, last_ibi = self._edge_intervals()
        before = times < beats[0]
        after = times > beats[-1]
        result[before] = (times[before] - beats[0]) / first_ibi
        result[after] = beats.size - 1 + (times[after] - beats[-1]) / last_ibi
        return result

    def to_seconds(self, beat_positions: Iterable[float] | float) -> np.ndarray:
        """Convert beat positions back to seconds."""
        positions = np.atleast_1d(np.asarray(beat_positions, dtype=float))
        beats = self.beat_times
        indices = np.arange(beats.size, dtype=float)
        result = np.interp(positions, indices, beats)
        first_ibi, last_ibi = self._edge_intervals()
        before = positions < 0
        after = positions > beats.size - 1
        result[before] = beats[0] + positions[before] * first_ibi
        result[after] = beats[-1] + (positions[after] - (beats.size - 1)) * last_ibi
        return result

    def local_interval(self, beat_position: float) -> float:
        """Seconds per beat around a beat position."""
        index = int(np.clip(math.floor(beat_position), 0, self.beat_times.size - 2))
        return float(self.beat_times[index + 1] - self.beat_times[index])

    # -- tempo level changes -------------------------------------------------
    def doubled(self) -> "BeatGrid":
        """Twice as many beats (each old beat split in half)."""
        beats = self.beat_times
        mids = (beats[:-1] + beats[1:]) / 2.0
        merged = np.empty(beats.size + mids.size)
        merged[0::2] = beats
        merged[1::2] = mids
        return replace(
            self,
            beat_times=merged,
            beats_per_bar=self.beats_per_bar * 2,
            downbeat_phase=self.downbeat_phase * 2,
        )

    def tripled_one_in_a_bar(self) -> "BeatGrid":
        """Each tracked beat was a whole bar of three: split it into three beats."""
        beats = self.beat_times
        steps = np.diff(beats)
        result = np.empty(beats.size + 2 * steps.size)
        result[0::3] = beats
        result[1::3] = beats[:-1] + steps / 3.0
        result[2::3] = beats[:-1] + 2.0 * steps / 3.0
        return replace(
            self,
            beat_times=result,
            beats_per_bar=3,
            downbeat_phase=0,
            beat_unit=Fraction(1),
        )

    def halved(self) -> "BeatGrid":
        """Half as many beats (only possible with an even number per bar)."""
        if self.beats_per_bar % 2 != 0 or self.beat_times.size < 6:
            return self
        start = self.downbeat_phase % 2
        beats = self.beat_times[start::2]
        new_phase = ((self.downbeat_phase - start) // 2) % (self.beats_per_bar // 2)
        return replace(
            self,
            beat_times=beats,
            beats_per_bar=self.beats_per_bar // 2,
            downbeat_phase=new_phase,
        )

    def bar_start_beat(self, beat_position: float) -> int:
        """Beat index of the bar line at or before ``beat_position``."""
        bar = math.floor((beat_position - self.downbeat_phase) / self.beats_per_bar)
        return self.downbeat_phase + bar * self.beats_per_bar


def constant_beat_grid(duration: float, bpm: float = 100.0) -> BeatGrid:
    """A steady grid used when no beats could be tracked."""
    interval = 60.0 / bpm
    count = max(4, int(math.ceil((duration + interval) / interval)) + 1)
    return BeatGrid(np.arange(count) * interval, beats_per_bar=4)


def _clean_beats(beat_times: Sequence[float]) -> np.ndarray:
    beats = np.unique(np.asarray(beat_times, dtype=float))
    if beats.size < 3:
        return beats
    median = float(np.median(np.diff(beats)))
    keep = [beats[0]]
    for value in beats[1:]:
        if value - keep[-1] >= 0.4 * median:
            keep.append(value)
    return np.asarray(keep)


def _fill_missing_beats(beats: np.ndarray) -> np.ndarray:
    """Insert beats where the tracker skipped some (e.g. during a long note)."""
    if beats.size < 4:
        return beats
    intervals = np.diff(beats)
    filled = [beats[0]]
    for index, interval in enumerate(intervals):
        window = intervals[max(0, index - 8) : index + 9]
        local = float(np.median(window))
        ratio = interval / local
        if ratio > 1.6:
            count = int(round(ratio))
            for step in range(1, count):
                filled.append(beats[index] + interval * step / count)
        filled.append(beats[index + 1])
    return np.asarray(filled)


def _extend_beats(beats: np.ndarray, duration: float) -> np.ndarray:
    """Continue the beat grid before the first and after the last tracked beat."""
    intervals = np.diff(beats)
    first = float(np.median(intervals[:4]))
    last = float(np.median(intervals[-4:]))
    before = []
    value = beats[0] - first
    while value > -first:
        before.append(value)
        value -= first
    after = []
    value = beats[-1] + last
    while value < duration + 2 * last:
        after.append(value)
        value += last
    return np.concatenate([np.asarray(before[::-1]), beats, np.asarray(after)])


def _downbeat_indices(beats: np.ndarray, downbeats: Sequence[float]) -> list[int]:
    if len(downbeats) == 0:
        return []
    median = float(np.median(np.diff(beats)))
    indices = []
    for time in downbeats:
        index = int(np.argmin(np.abs(beats - time)))
        if abs(beats[index] - time) <= 0.3 * median:
            indices.append(index)
    return sorted(set(indices))


def downbeat_gap_shares(beats: np.ndarray, downbeats: Sequence[float]) -> dict[int, float]:
    """How often the tracker's bar marks are 2, 3, 4 or 6 beats apart (0..1).

    A mark two bars later counts half, and for the longer bars a mark one
    beat early or late (rubato) counts a little.
    """
    indices = _downbeat_indices(beats, downbeats)
    gaps = [int(g) for g in np.diff(indices) if 1 <= g <= 13]
    if len(gaps) < 4:
        return {}
    counts = Counter(gaps)
    shares = {}
    for length in (2, 3, 4, 6):
        value = counts[length] + 0.5 * counts[2 * length]
        if length >= 4:
            value += 0.3 * (counts[length - 1] + counts[length + 1])
        shares[length] = value / len(gaps)
    return shares


def accent_per_beat(beats: np.ndarray, notes: Sequence[NoteEvent]) -> np.ndarray:
    """Accompaniment accent on every beat: chords and bass notes.

    A chord counts by its size, a bass note by how low it is. Single notes in
    the upper register (a melody moving on every beat) count for nothing, so
    they do not blur the pattern that marks the meter.
    """
    accents = np.zeros(beats.size)
    if len(notes) < 4 or beats.size < 2:
        return accents
    starts = np.asarray([n.start for n in notes], dtype=float)
    pitches = np.asarray([n.pitch for n in notes], dtype=float)
    period = np.gradient(beats)
    low_reference = float(np.percentile(pitches, 40))
    for group in _onset_groups(starts, 0.035):
        time = float(np.median(starts[group]))
        index = int(np.clip(np.searchsorted(beats, time), 1, beats.size - 1))
        if time - beats[index - 1] < beats[index] - time:
            index -= 1
        if abs(beats[index] - time) > 0.2 * period[index]:
            continue
        lowest = float(pitches[group].min())
        accents[index] += (min(len(group), 4) - 1) + float(
            np.clip((low_reference - lowest) / 12.0, 0.0, 2.0)
        )
    return accents


def accent_repetition(accents: np.ndarray) -> dict[int, float]:
    """How strongly the accent pattern repeats every 2, 3, 4 and 6 beats (-1..1).

    This is an autocorrelation, so it measures the bar length without needing
    to know where the bar lines are, and a pause that shifts the count only
    disturbs it locally.
    """
    if accents.size < 24:
        return {}
    # Compare each beat with its neighbourhood, not with the whole piece: loud
    # passages and quiet ones would otherwise look alike at every distance.
    # Twelve beats hold a whole number of every pattern being tested.
    local_mean = np.convolve(accents, np.ones(12) / 12.0, mode="same")
    centred = accents - local_mean
    energy = float(np.sum(centred * centred))
    if energy <= 0:
        return {}
    return {
        lag: float(np.sum(centred[:-lag] * centred[lag:]) / energy) for lag in (2, 3, 4, 6)
    }


def infer_meter(
    beats: np.ndarray, downbeats: Sequence[float]
) -> tuple[int, int] | None:
    """Return (beats_per_bar, phase) from tracked downbeats, or None."""
    indices = _downbeat_indices(beats, downbeats)
    if len(indices) < 3:
        return None
    gaps = Counter(
        int(gap) for gap in np.diff(indices) if 2 <= gap <= 7
    )
    if not gaps:
        return None
    # Prefer common meters when counts tie.
    preference = {4: 0.3, 3: 0.2, 2: 0.1}
    beats_per_bar = max(gaps, key=lambda gap: gaps[gap] + preference.get(gap, 0.0))
    phases = Counter(index % beats_per_bar for index in indices)
    phase = phases.most_common(1)[0][0]
    return beats_per_bar, phase


def _emphasis_per_beat(beats: np.ndarray, notes: Sequence[NoteEvent]) -> np.ndarray:
    """Musical weight of notes starting on each beat (long, low, loud = heavy)."""
    weights = np.zeros(beats.size)
    if not notes or beats.size < 2:
        return weights
    period = float(np.median(np.diff(beats)))
    for item in notes:
        index = int(np.searchsorted(beats, item.start))
        candidates = [i for i in (index - 1, index) if 0 <= i < beats.size]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda i: abs(beats[i] - item.start))
        if abs(beats[nearest] - item.start) > 0.15 * period:
            continue
        length = min(item.end - item.start, 2.0 * period)
        bass = 2.0 if item.pitch < 52 else 1.0
        weights[nearest] += length * bass * (0.5 + item.velocity / 127)
    return weights


def _strength_for(beats: np.ndarray, original: Sequence[float], strength: Sequence[float]) -> np.ndarray:
    original = np.asarray(original, dtype=float)
    strength = np.asarray(strength, dtype=float)
    result = np.zeros(beats.size)
    if original.size == 0 or strength.size != original.size:
        return result
    period = float(np.median(np.diff(beats)))
    for index, value in enumerate(beats):
        nearest = int(np.argmin(np.abs(original - value)))
        if abs(original[nearest] - value) <= 0.3 * period:
            result[index] = strength[nearest]
    return result


LOCK_REACH = 0.5  # a beat may move this far (in beats) to reach a note
LOCK_CLOSENESS = 0.6  # ...at a cost that grows with the distance moved
LOCK_EVENNESS = 0.15  # successive beats should stay this evenly spaced
LOCK_VIRTUAL = 0.3  # cost of leaving a beat where no note was played


def lock_beats_to_notes(beats: np.ndarray, notes: Sequence[NoteEvent]) -> np.ndarray:
    """Move every beat onto the note that was played on it.

    An audio beat tracker is accurate to a few hundredths of a second, and
    where it only hears every second or third beat the ones in between are
    interpolated. That is enough to drift between the notes of a fast passage,
    which then read as odd 32nd-note rhythms. The transcribed notes are timed
    far more precisely, so each beat is moved to a nearby note start, chosen
    so that the beats stay evenly spaced (and stay near the tracker's).
    """
    count = beats.size
    if count < 4 or len(notes) < 8:
        return beats
    starts = np.asarray([n.start for n in notes], dtype=float)
    pitches = np.asarray([n.pitch for n in notes], dtype=float)
    groups = _onset_groups(starts, 0.035)
    times = np.asarray([float(np.median(starts[g])) for g in groups])
    lows = np.asarray([float(pitches[g].min()) for g in groups])
    sizes = np.asarray([len(g) for g in groups])
    period = np.gradient(beats)

    # Small preferences that settle a choice between two equally near notes:
    # chords, and the low note of a low-high figure, tend to be on the beat.
    salience = 0.15 * (np.minimum(sizes, 4) - 1.0)
    for index in range(1, times.size - 1):
        local = period[int(np.clip(np.searchsorted(beats, times[index]), 0, count - 1))]
        near_before = times[index] - times[index - 1] <= 0.75 * local
        near_after = times[index + 1] - times[index] <= 0.75 * local
        if not (near_before and near_after):
            continue
        if lows[index] <= lows[index - 1] - 5 and lows[index] <= lows[index + 1] - 5:
            salience[index] += 0.25
        elif lows[index] >= lows[index - 1] + 5 and lows[index] >= lows[index + 1] + 5:
            salience[index] -= 0.15

    options: list[list[tuple[float, float]]] = []  # per beat: (time, cost)
    for index in range(count):
        local = period[index]
        low = np.searchsorted(times, beats[index] - LOCK_REACH * local, side="left")
        high = np.searchsorted(times, beats[index] + LOCK_REACH * local, side="right")
        choices = [(float(beats[index]), LOCK_VIRTUAL)]
        for group in range(low, high):
            moved = (times[group] - beats[index]) / (LOCK_CLOSENESS * local)
            choices.append((float(times[group]), float(moved * moved - salience[group])))
        options.append(choices)

    # Viterbi over pairs of consecutive choices (so evenness can be judged).
    previous_pairs: dict[tuple[int, int], tuple[float, tuple[int, int] | None]] = {}
    for a, (time_a, cost_a) in enumerate(options[0]):
        for b, (time_b, cost_b) in enumerate(options[1]):
            if time_b - time_a < 0.3 * period[0]:
                continue
            previous_pairs[(a, b)] = (cost_a + cost_b, None)
    history = [previous_pairs]
    for index in range(2, count):
        local = period[index]
        current: dict[tuple[int, int], tuple[float, tuple[int, int] | None]] = {}
        for (a, b), (total, _) in previous_pairs.items():
            time_a, time_b = options[index - 2][a][0], options[index - 1][b][0]
            for c, (time_c, cost_c) in enumerate(options[index]):
                step = time_c - time_b
                if step < 0.3 * local:
                    continue
                uneven = (step - (time_b - time_a)) / (LOCK_EVENNESS * local)
                value = total + cost_c + uneven * uneven
                best = current.get((b, c))
                if best is None or value < best[0]:
                    current[(b, c)] = (value, (a, b))
        if not current:
            return beats
        history.append(current)
        previous_pairs = current

    pair = min(previous_pairs, key=lambda key: previous_pairs[key][0])
    chosen = [pair[1], pair[0]]
    for index in range(count - 1, 1, -1):
        pair = history[index - 1][pair][1]
        chosen.append(pair[0])
    chosen.reverse()
    return np.asarray([options[index][choice][0] for index, choice in enumerate(chosen)])


def _meter_from_note_emphasis(weights: np.ndarray) -> tuple[int, int] | None:
    """Distinguish triple from common time using long, low and loud notes."""
    count = weights.size
    if count < 12 or np.allclose(weights, weights.mean()):
        return None
    flat_error = float(np.sum((weights - weights.mean()) ** 2))
    positions = np.arange(count)
    # Compare the repeating pattern with a flat one using the Bayesian
    # information criterion, accounting for the extra mean fitted per slot.
    best_gain = 6.0  # require strong evidence before overriding the default
    best = None
    # Without tracker bar marks, half-bar accents in 4/4 look like 2/4 and
    # pairs of 3/4 bars can look like six-beat bars.
    for length in (3, 4):
        if count < 4 * length:
            continue
        profile = np.array([weights[phase::length].mean() for phase in range(length)])
        error = float(np.sum((weights - profile[positions % length]) ** 2))
        gain = -count * math.log(max(error / flat_error, 1e-12)) - (length - 1) * math.log(count)
        if gain > best_gain:
            best_gain = gain
            best = length, int(np.argmax(profile))
    return best


def build_beat_grid(
    beat_times: Sequence[float],
    downbeat_times: Sequence[float],
    duration: float,
    *,
    downbeat_strength: Sequence[float] | None = None,
    notes: Sequence[NoteEvent] = (),
    lock_to_notes: bool = False,
) -> BeatGrid:
    """Create a clean, gap-free beat grid that covers the whole recording.

    The bar length and bar-line position combine the tracker's downbeat
    probability with where long/low/loud notes fall. With ``lock_to_notes``
    (for precisely timed notes, i.e. piano) each beat is also moved onto the
    note played on it.
    """
    beats = _clean_beats(beat_times)
    if beats.size < 4:
        return constant_beat_grid(duration)
    beats = _fill_missing_beats(beats)
    tracked = beats
    if lock_to_notes:
        beats = lock_beats_to_notes(beats, notes)
    meter = None
    density = 0.0
    # Ignore silence before and after the performance when looking for accents.
    if notes:
        first = beats.searchsorted(min(n.start for n in notes) - 0.1)
        last = beats.searchsorted(max(n.end for n in notes), side="right")
        playing = slice(first, last)
    else:
        playing = slice(0, beats.size)
    if downbeat_strength is not None and len(downbeat_strength) == len(beat_times):
        from beat_tracking import meter_scores

        # Tracker evidence is looked up at the tracker's own beat times.
        strength = _strength_for(tracked, beat_times, downbeat_strength)
        if strength[playing].size:
            density = float(np.mean(strength[playing] > 0.5))
        scores = meter_scores(
            strength,
            float(np.median(np.diff(beats))),
            _emphasis_per_beat(beats, notes),
            downbeat_gap_shares(tracked, downbeat_times),
            accent_repetition(accent_per_beat(beats, notes)),
        )
        if scores:
            _, beats_per_bar, phase = scores[0]
            meter = beats_per_bar, phase
    if meter is None:
        meter = infer_meter(beats, downbeat_times)
        if meter is None and notes:
            inferred = _meter_from_note_emphasis(_emphasis_per_beat(beats, notes)[playing])
            if inferred is not None:
                length, phase = inferred
                meter = length, (phase + playing.start) % length
    extended = _extend_beats(beats, duration)
    # Indices shifted because beats were prepended.
    offset = int(np.argmin(np.abs(extended - beats[0])))
    if meter is None:
        beats_per_bar, phase = 4, 0
    else:
        beats_per_bar, phase = meter
        phase = (phase + offset) % beats_per_bar
    return BeatGrid(
        extended,
        beats_per_bar=beats_per_bar,
        downbeat_phase=phase,
        downbeat_times=np.asarray(downbeat_times, dtype=float),
        downbeat_density=density,
    )


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------


@dataclass
class QuantizationResult:
    notes: list[QuantizedNote]
    subdivisions: dict[int, int]  # beat index -> chosen subdivision


def _onset_groups(onsets: np.ndarray, tolerance: float) -> list[list[int]]:
    """Group notes that start (almost) together, e.g. the notes of a chord."""
    order = np.argsort(onsets, kind="stable")
    groups: list[list[int]] = []
    current: list[int] = []
    for index in order:
        if current and onsets[index] - onsets[current[0]] > tolerance:
            groups.append(current)
            current = []
        current.append(int(index))
    if current:
        groups.append(current)
    return groups


def _timing_scales(
    grid: BeatGrid, position: float, looseness: float = 1.0
) -> tuple[float, float]:
    """Return (window lead, timing jitter) in beats around a position.

    ``looseness`` > 1 is for less precise onsets (sung or pitch-tracked
    notes): larger expected jitter makes simpler rhythms win.
    """
    seconds_per_beat = grid.local_interval(position)
    lead = float(np.clip(0.05 * looseness / seconds_per_beat, 0.04, 0.15))
    jitter = float(np.clip(0.035 * looseness / seconds_per_beat, 0.04, 0.10 * looseness))
    return lead, jitter


def _snap(value: float, subdivision: int) -> Fraction:
    return Fraction(int(round(value * subdivision)), subdivision)


def _window_cost(
    items: list[tuple[float, float]], subdivision: int
) -> float:
    cost = SUBDIVISION_PRIOR[subdivision]
    snapped_positions = set()
    ordered = sorted(items)
    snapped_list = []
    for local, jitter in ordered:
        snapped = _snap(local, subdivision)
        snapped_positions.add(snapped)
        snapped_list.append(float(snapped))
        error = (local - float(snapped)) / jitter
        cost += error * error
    # Evenly played notes should stay evenly spaced (e.g. sextuplets must not
    # become an uneven 32nd-note pattern).
    for index in range(1, len(ordered)):
        heard = ordered[index][0] - ordered[index - 1][0]
        written = snapped_list[index] - snapped_list[index - 1]
        error = (heard - written) / ordered[index][1]
        cost += SPACING_WEIGHT * error * error
    collisions = len(items) - len(snapped_positions)
    return cost + COLLISION_COST * collisions


def choose_subdivisions(
    windows: dict[int, list[tuple[float, float]]],
    allowed: Sequence[int] = DEFAULT_SUBDIVISIONS,
) -> tuple[dict[int, int], float]:
    """Pick one subdivision per beat with a Viterbi pass over the beats.

    ``windows`` maps beat index -> list of (position inside beat, jitter).
    Returns the choice per beat and the total cost.
    """
    if not windows:
        return {}, 0.0
    allowed = tuple(s for s in allowed if s in SUBDIVISION_PRIOR)
    beat_indices = sorted(windows)
    emissions = [
        [_window_cost(windows[index], s) for s in allowed] for index in beat_indices
    ]
    total = np.asarray(emissions[0], dtype=float)
    back: list[list[int]] = []
    for position in range(1, len(beat_indices)):
        gap = beat_indices[position] - beat_indices[position - 1]
        switch = FAMILY_SWITCH_COST if gap <= 2 else 0.0
        new_total = np.empty(len(allowed))
        pointers = []
        for current, sub in enumerate(allowed):
            best_value = math.inf
            best_previous = 0
            for previous, prev_sub in enumerate(allowed):
                family_a = SUBDIVISION_FAMILY[sub]
                family_b = SUBDIVISION_FAMILY[prev_sub]
                penalty = switch if family_a and family_b and family_a != family_b else 0.0
                value = total[previous] + penalty
                if value < best_value:
                    best_value = value
                    best_previous = previous
            new_total[current] = best_value + emissions[position][current]
            pointers.append(best_previous)
        back.append(pointers)
        total = new_total
    best = int(np.argmin(total))
    total_cost = float(total[best])
    choice = [best]
    for pointers in reversed(back):
        choice.append(pointers[choice[-1]])
    choice.reverse()
    return {index: allowed[c] for index, c in zip(beat_indices, choice)}, total_cost


def _assign_window(position: float, lead: float) -> tuple[int, float]:
    index = math.floor(position + lead)
    return index, position - index


def quantize_notes(
    notes: Sequence[NoteEvent],
    grid: BeatGrid,
    *,
    allowed_subdivisions: Sequence[int] = DEFAULT_SUBDIVISIONS,
    chord_tolerance: float = 0.035,
    looseness: float = 1.0,
) -> QuantizationResult:
    """Snap note starts/ends to the beat grid and return quantized notes."""
    if not notes:
        return QuantizationResult([], {})
    starts_seconds = np.asarray([note.start for note in notes], dtype=float)
    ends_seconds = np.asarray([note.end for note in notes], dtype=float)
    start_beats = grid.to_beats(starts_seconds)
    end_beats = grid.to_beats(ends_seconds)

    groups = _onset_groups(starts_seconds, chord_tolerance)
    group_positions = [float(np.median(start_beats[group])) for group in groups]

    windows: dict[int, list[tuple[float, float]]] = {}
    group_windows: list[tuple[int, float]] = []
    for position in group_positions:
        lead, jitter = _timing_scales(grid, position, looseness)
        index, local = _assign_window(position, lead)
        windows.setdefault(index, []).append((local, jitter))
        group_windows.append((index, local))

    subdivisions, _ = choose_subdivisions(windows, allowed_subdivisions)

    note_start: dict[int, Fraction] = {}
    for group, (index, local) in zip(groups, group_windows):
        snapped = index + _snap(local, subdivisions[index])
        for note_index in group:
            note_start[note_index] = snapped

    families = {index: SUBDIVISION_FAMILY[s] for index, s in subdivisions.items()}

    def end_subdivision(beat_index: int) -> int:
        if beat_index in subdivisions:
            return subdivisions[beat_index]
        nearby = [
            families[i]
            for i in (beat_index - 1, beat_index + 1, beat_index - 2, beat_index + 2)
            if i in families and families[i]
        ]
        return 3 if nearby and nearby[0] == 2 else 2

    quantized: list[QuantizedNote] = []
    for index, note in enumerate(notes):
        start = note_start[index]
        lead, _ = _timing_scales(grid, end_beats[index], looseness)
        end_index, end_local = _assign_window(float(end_beats[index]), lead)
        end = end_index + _snap(end_local, end_subdivision(end_index))
        if end <= start:
            start_index = math.floor(start)
            step = Fraction(1, subdivisions.get(start_index, 2) or 2)
            if subdivisions.get(start_index, 1) == 1:
                step = Fraction(1, 2)
            end = start + step
        quantized.append(
            QuantizedNote(
                start=start,
                end=end,
                pitch=note.pitch,
                velocity=note.velocity,
                part=note.part,
                raw_start=float(start_beats[index]),
                raw_end=float(end_beats[index]),
                rolled=note.rolled,
            )
        )
    return QuantizationResult(quantized, subdivisions)


def estimate_onset_lag(notes: Sequence[NoteEvent], grid: BeatGrid) -> float:
    """How late (seconds) a pitch-tracked line sits behind the beat grid.

    Pitch tracking reports a note once its pitch has settled, and singers often
    sit slightly behind the beat. The shift that best lines the note starts up
    with eighth-note positions is that systematic delay. Notes far from any
    eighth (16ths, triplets) count only a little, so they do not mislead it.
    """
    if len(notes) < 8:
        return 0.0
    starts = np.asarray([n.start for n in notes], dtype=float)
    eighth = grid.median_interval / 2.0
    limit = min(0.2, 0.4 * eighth)
    best_lag, best_cost = 0.0, math.inf
    for lag in np.arange(-0.04, limit + 1e-9, 0.01):
        positions = grid.to_beats(starts - lag) * 2.0
        distance = np.abs(positions - np.round(positions)) / 2.0  # in beats
        cost = float(np.sum(np.minimum(distance, 0.12) ** 2))
        cost += len(starts) * 0.0004 * (lag / 0.05) ** 2  # prefer small shifts
        if cost < best_cost:
            best_lag, best_cost = float(lag), cost
    return best_lag


def quantization_cost(notes: Sequence[NoteEvent], grid: BeatGrid) -> float:
    """Average cost per onset group of explaining the notes with this grid."""
    if not notes:
        return 0.0
    starts_seconds = np.asarray([note.start for note in notes], dtype=float)
    groups = _onset_groups(starts_seconds, 0.035)
    positions = grid.to_beats([float(np.median(starts_seconds[g])) for g in groups])
    windows: dict[int, list[tuple[float, float]]] = {}
    for position in positions:
        lead, jitter = _timing_scales(grid, float(position))
        index, local = _assign_window(float(position), lead)
        windows.setdefault(index, []).append((local, jitter))
    _, cost = choose_subdivisions(windows)
    return cost / max(1, len(groups))


# ---------------------------------------------------------------------------
# Choosing how to notate the beat
# ---------------------------------------------------------------------------


def median_onset_interval(notes: Sequence[NoteEvent], grid: BeatGrid) -> float:
    """Median gap between successive note starts, measured in beats."""
    if len(notes) < 3:
        return 1.0
    starts = np.asarray(sorted({round(note.start, 3) for note in notes}))
    groups = _onset_groups(starts, 0.035)
    group_times = np.asarray([starts[g[0]] for g in groups])
    if group_times.size < 3:
        return 1.0
    gaps = np.diff(grid.to_beats(group_times))
    gaps = gaps[(gaps > 0.05) & (gaps < 4.0)]
    if gaps.size == 0:
        return 1.0
    return float(np.median(gaps))


def _ternary_share(notes: Sequence[NoteEvent], grid: BeatGrid) -> float:
    """Share of off-beat note starts that fall on thirds rather than halves/quarters."""
    if not notes:
        return 0.0
    starts = np.asarray(sorted({round(n.start, 3) for n in notes}))
    positions = grid.to_beats(starts)
    local = positions - np.floor(positions)
    thirds = np.sum((np.abs(local - 1 / 3) < 0.06) | (np.abs(local - 2 / 3) < 0.06))
    binary = np.sum(
        (np.abs(local - 0.5) < 0.06) | (np.abs(local - 0.25) < 0.05) | (np.abs(local - 0.75) < 0.05)
    )
    total = thirds + binary
    return float(thirds / total) if total >= 6 else 0.0


def _beat_emphasis(
    notes: Sequence[NoteEvent], grid: BeatGrid, *, bass_weight: float = 1.5
) -> np.ndarray:
    """How much musical weight (long, low, loud notes) falls on each beat of the bar."""
    weights = np.zeros(grid.beats_per_bar)
    if not notes:
        return weights
    positions = grid.to_beats([n.start for n in notes])
    lowest = np.percentile([n.pitch for n in notes], 25)
    for item, position in zip(notes, positions):
        nearest = round(float(position))
        if abs(position - nearest) > 0.15:
            continue
        slot = (nearest - grid.downbeat_phase) % grid.beats_per_bar
        length = min(item.end - item.start, 2.0)
        bass = bass_weight if item.pitch <= lowest else 1.0
        weights[slot] += length * bass * (0.5 + item.velocity / 127)
    return weights


def accent_cycle(notes: Sequence[NoteEvent], grid: BeatGrid, cycle: int) -> np.ndarray:
    """Accent weight on each beat of a repeating ``cycle`` of beats.

    Accents are what mark the meter in an accompaniment: chords (several
    notes at once) and bass notes, the lower the heavier. Single melody notes
    count for nothing, so a melody running on every beat does not blur it.
    Slot 0 is the grid's current bar-line position.
    """
    weights = np.zeros(cycle)
    accents = accent_per_beat(grid.beat_times, notes)
    for index, accent in enumerate(accents):
        weights[(index - grid.downbeat_phase) % cycle] += accent
    return weights


BAR_ACCENTS = {  # how much weight each beat of the bar normally carries
    2: (1.0, 0.0),
    3: (1.0, 0.0, 0.0),
    4: (1.0, 0.0, 0.4, 0.0),
    6: (1.0, 0.0, 0.0, 0.45, 0.0, 0.0),
}
SLIP_COST = 1.2  # evidence needed before a pause is allowed to shift the bar lines


def align_bar_lines(grid: BeatGrid, notes: Sequence[NoteEvent]) -> BeatGrid:
    """Re-seat the bar lines after pauses.

    A fermata or a breath between sections is not held for a whole number of
    beats, so the music after it can start on a different beat of the
    tracker's count and every later bar line is then one or two beats off.
    Here a beat may be dropped or added inside a silence when that puts the
    accents of the following section (the tracker's bar marks, chords and
    bass notes) back on the bar lines.
    """
    beats = grid.beat_times
    length = grid.beats_per_bar
    template = BAR_ACCENTS.get(length)
    if template is None or len(notes) < 8 or beats.size < 4 * length:
        return grid
    count = beats.size
    period = np.gradient(beats)

    # Evidence that each beat starts a bar.
    evidence = np.zeros(count)
    starts = np.asarray([n.start for n in notes], dtype=float)
    accents = accent_per_beat(beats, notes)
    for time in grid.downbeat_times:
        nearest = int(np.argmin(np.abs(beats - time)))
        # The tracker hears a rolled chord when it starts, a little before the
        # beat the chord belongs to: of the beats within reach, the bar mark
        # goes to the accented one.
        reach = 0.55 * period[nearest]
        close = [i for i in (nearest - 1, nearest, nearest + 1) if 0 <= i < count and abs(beats[i] - time) <= reach]
        if not close:
            continue
        index = max(close, key=lambda i: (accents[i], -abs(beats[i] - time)))
        if accents[index] <= 0:
            index = nearest
            if abs(beats[index] - time) > 0.3 * period[index]:
                continue
        evidence[index] += 1.0
    positive = accents[accents > 0]
    if positive.size:
        evidence += 0.6 * np.clip(accents / float(np.percentile(positive, 90)), 0.0, 1.5)
    evidence -= evidence.mean()
    weights = np.asarray(template) - float(np.mean(template))

    # Beats inside a silence may be dropped or doubled.
    onset_times = np.unique(starts)
    free = np.zeros(count, dtype=bool)
    for index, beat in enumerate(beats):
        after = int(np.searchsorted(onset_times, beat))
        previous = onset_times[after - 1] if after > 0 else -np.inf
        following = onset_times[after] if after < onset_times.size else np.inf
        quiet = min(beat - previous, following - beat) > 0.6 * period[index]
        long_gap = (following - previous) >= 2.2 * period[index]
        free[index] = quiet and long_gap

    # Viterbi over the position in the bar of every beat.
    NORMAL, DROP, ADD = 0, 1, 2
    score = np.array([weights[p] * evidence[0] for p in range(length)])
    back: list[np.ndarray] = []
    moves: list[np.ndarray] = []
    for index in range(1, count):
        new_score = np.empty(length)
        pointer = np.empty(length, dtype=np.int64)
        move = np.zeros(length, dtype=np.int64)
        for position in range(length):
            best_value = score[(position - 1) % length]
            best_previous, best_move = (position - 1) % length, NORMAL
            if free[index]:
                # This beat is surplus: it repeats the previous position.
                value = score[position] - SLIP_COST
                if value > best_value:
                    best_value, best_previous, best_move = value, position, DROP
                # A beat is missing before this one.
                value = score[(position - 2) % length] - SLIP_COST
                if value > best_value:
                    best_value, best_previous, best_move = value, (position - 2) % length, ADD
            new_score[position] = best_value + weights[position] * evidence[index]
            pointer[position] = best_previous
            move[position] = best_move
        score = new_score
        back.append(pointer)
        moves.append(move)
    position = int(np.argmax(score))
    path = [position]
    path_moves = [NORMAL]
    for index in range(count - 1, 0, -1):
        path_moves.append(int(moves[index - 1][position]))
        position = int(back[index - 1][position])
        path.append(position)
    path.reverse()
    path_moves = path_moves[1:][::-1]  # move used to arrive at beats 1..count-1
    if not any(path_moves):
        first_bar = next((i for i, p in enumerate(path) if p == 0), 0)
        return replace(grid, downbeat_phase=first_bar % length)

    new_beats: list[float] = [float(beats[0])]
    positions: list[int] = [path[0]]
    for index in range(1, count):
        move = path_moves[index - 1]
        if move == DROP:
            continue
        if move == ADD:
            new_beats.append(float((beats[index - 1] + beats[index]) / 2.0))
            positions.append((path[index] - 1) % length)
        new_beats.append(float(beats[index]))
        positions.append(path[index])
    first_bar = next((i for i, p in enumerate(positions) if p == 0), 0)
    return replace(grid, beat_times=np.asarray(new_beats), downbeat_phase=first_bar % length)


def six_beat_grouping(notes: Sequence[NoteEvent], grid: BeatGrid) -> tuple[int, int]:
    """Are bars of six quick beats felt in threes (6/8) or in twos (3/4)?

    Returns (beats per group, bar-line phase). The decision compares how
    strongly the accents repeat every three beats against every two; that
    comparison does not depend on where the bar line currently sits.
    """
    cycle = accent_cycle(notes, grid, 6)
    if cycle.sum() <= 0:
        return 3, grid.downbeat_phase
    k = np.arange(6)
    threes = abs(np.sum(cycle * np.exp(-2j * np.pi * k / 3)))
    twos = abs(np.sum(cycle * np.exp(-1j * np.pi * k)))
    if twos > 1.15 * threes:
        # 3/4: the bar starts on the heaviest of the accented positions.
        return 2, int((grid.downbeat_phase + int(np.argmax(cycle))) % 6)
    # 6/8: accents on two positions three beats apart; the heavier (lower
    # bass) one is the bar line. The tracker's own choice wins a near tie.
    pair = int(np.argmax(cycle[:3] + cycle[3:]))
    first, second = float(cycle[pair]), float(cycle[pair + 3])
    if pair == 0 and first >= 0.9 * second:
        offset = 0
    else:
        offset = pair if first >= second else pair + 3
    return 3, int((grid.downbeat_phase + offset) % 6)


def choose_beat_level(grid: BeatGrid, notes: Sequence[NoteEvent]) -> BeatGrid:
    """Double or halve the tracked beat so the notation uses familiar values.

    Beat trackers sometimes follow half notes or eighth notes instead of the
    quarter-note beat a musician would write. A beat that is very slow while
    notes are short (or very fast while notes are long) is switched to the
    more natural level, and six-beat bars become 6/8 or 3/4.
    """
    bpm = grid.bpm
    typical_gap = median_onset_interval(notes, grid)
    beats_per_bar = grid.beats_per_bar

    if grid.downbeat_density >= 0.6 and bpm < 100 and _ternary_share(notes, grid) >= 0.5:
        # Waltz/mazurka "one-in-a-bar": the tracker followed whole bars. Which
        # third is the bar line is decided by where the bass notes fall.
        tripled = grid.tripled_one_in_a_bar()
        emphasis = _beat_emphasis(notes, tripled, bass_weight=3.0)
        return replace(tripled, downbeat_phase=int(np.argmax(emphasis)))

    if beats_per_bar in (6, 9, 12):
        # The tracker followed eighth notes.
        if beats_per_bar == 6:
            group, phase = six_beat_grouping(notes, grid)
            if group == 2 and bpm > 150:
                return replace(grid, downbeat_phase=phase).halved()  # felt in twos: 3/4
            return replace(grid, beat_unit=Fraction(1, 2), downbeat_phase=phase)  # 6/8
        return replace(grid, beat_unit=Fraction(1, 2))  # 9/8 or 12/8

    can_double = beats_per_bar in (1, 2, 4)
    if can_double and (
        bpm < 45 or (bpm < 60 and typical_gap <= 0.5) or (bpm < 90 and typical_gap <= 0.3)
    ):
        return grid.doubled()
    if bpm > 200 or (bpm > 160 and typical_gap >= 0.9):
        return grid.halved()
    return grid


def is_compound(subdivisions: dict[int, int]) -> bool:
    """True when most subdivided beats split into three (e.g. 6/8 or 12/8)."""
    subdivided = [s for s in subdivisions.values() if s > 1]
    if len(subdivided) < 8:
        return False
    ternary = sum(1 for s in subdivided if SUBDIVISION_FAMILY[s] == 2)
    return ternary / len(subdivided) >= 0.6


def phase_from_notes(
    notes: Sequence[QuantizedNote], beats_per_bar: int
) -> int:
    """Guess where bar lines go from note emphasis (long/low/loud notes)."""
    if beats_per_bar <= 1 or not notes:
        return 0
    weights = np.zeros(beats_per_bar)
    for item in notes:
        if item.start.denominator != 1:
            continue
        beat = int(item.start) % beats_per_bar
        length = min(float(item.duration), 4.0)
        bass_bonus = 1.5 if item.pitch < 55 else 1.0
        weights[beat] += length * bass_bonus * (0.5 + item.velocity / 127)
    return int(np.argmax(weights))


def apply_time_signature(grid: BeatGrid, choice: str, notes: Sequence[QuantizedNote]) -> BeatGrid:
    """Force a user-chosen time signature onto the grid."""
    numerator, denominator = (int(part) for part in choice.split("/"))
    eighth_level = grid.beat_unit == Fraction(1, 2) or grid.bpm > 140
    if denominator == 8 and numerator % 3 == 0 and eighth_level:
        # The grid already beats in eighth notes: one beat per eighth.
        beats_per_bar = numerator
        beat_unit = Fraction(1, 2)
    elif denominator == 8 and numerator % 3 == 0:
        beats_per_bar = numerator // 3
        beat_unit = Fraction(3, 2)
    else:
        beats_per_bar = numerator
        beat_unit = Fraction(4, denominator)
    if beats_per_bar == grid.beats_per_bar:
        phase = grid.downbeat_phase
    else:
        indices = _downbeat_indices(grid.beat_times, grid.downbeat_times)
        if len(indices) >= 3:
            phase = Counter(i % beats_per_bar for i in indices).most_common(1)[0][0]
        else:
            phase = phase_from_notes(notes, beats_per_bar)
    return replace(
        grid,
        beats_per_bar=beats_per_bar,
        downbeat_phase=phase,
        beat_unit=beat_unit,
    )
