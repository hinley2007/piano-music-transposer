"""Beat and downbeat tracking.

beat_this (a neural beat tracker) gives a beat probability and a downbeat
probability 50 times per second. Picking peaks directly is fragile in
syncopated music (ragtime, funk...), where accents land between beats. So the
beats are re-decoded with a dynamic-programming search that rewards strong
beat probability *and* a steady (but gently changing) tempo. The meter is then
chosen by testing every bar length and position against the downbeat
probability.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

LOGGER = logging.getLogger(__name__)

FPS = 50  # beat_this frame rate
METER_CANDIDATES = (2, 3, 4, 6)
METER_PRIOR = {4: 0.03, 3: 0.02, 2: 0.0, 6: -0.02}
GAP_WEIGHT = 0.8  # weight of the spacing between the tracker's bar marks
REPETITION_WEIGHT = 0.5  # weight of the accompaniment's accent pattern

_LOCK = threading.Lock()
_TRACKER = None


def _load_tracker():
    global _TRACKER
    with _LOCK:
        if _TRACKER is None:
            from beat_this.inference import Audio2Beats

            _TRACKER = Audio2Beats(checkpoint_path="final0", device="cpu", dbn=False)
        return _TRACKER


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def track_beats(
    audio: np.ndarray, sample_rate: int, notes=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (beat times, downbeat times, downbeat strength per beat).

    ``notes`` are the transcribed notes, if already known. They settle how a
    slow tracker pulse divides into beats (see :func:`_period_from_tracker`).
    """
    contour = note_contour(notes) if notes else None
    try:
        tracker = _load_tracker()
        spectrogram = tracker.signal2spect(audio, sample_rate)
        beat_logits, downbeat_logits = tracker.spect2frames(spectrogram)
        raw_beats, raw_downbeats = tracker.frames2beats(beat_logits, downbeat_logits)
        raw_beats = np.asarray(raw_beats, dtype=float)
        beat_activation = _sigmoid(beat_logits.cpu().numpy().astype(float))
        downbeat_activation = _sigmoid(downbeat_logits.cpu().numpy().astype(float))
        if raw_beats.size >= 8:
            beats = steady_beats(beat_activation, raw_beats, contour=contour)
            strength = downbeat_strength(beats, downbeat_activation)
            # The tracker's own bar marks: their spacing shows the bar length
            # even when a pause shifts the count part-way through the piece.
            downbeats = np.asarray(raw_downbeats, dtype=float)
            if downbeats.size < 4:
                downbeats = choose_downbeats(beats, strength)
            return beats, downbeats, strength
        LOGGER.warning("beat_this found only %d beats; using librosa instead", raw_beats.size)
    except Exception:
        LOGGER.warning("beat_this failed; using librosa instead", exc_info=True)
    beats = _librosa_beats(audio, sample_rate)
    return beats, np.zeros(0), np.zeros(0)


def _autocorrelation(values: np.ndarray) -> np.ndarray:
    centred = values - values.mean()
    spectrum = np.fft.rfft(centred, 2 * centred.size)
    result = np.fft.irfft(spectrum * np.conj(spectrum))[: centred.size]
    return result / (result[0] + 1e-9)


def _acf_peak(acf: np.ndarray, lag: float) -> tuple[float, float]:
    """Highest autocorrelation within +-10% of a lag, with sub-frame refinement."""
    low = max(1, int(np.floor(lag * 0.9)))
    high = min(acf.size - 2, int(np.ceil(lag * 1.1)))
    if high <= low:
        return 0.0, lag
    best = low + int(np.argmax(acf[low : high + 1]))
    left, centre, right = acf[best - 1], acf[best], acf[best + 1]
    denominator = left - 2 * centre + right
    refined = best + (0.5 * (left - right) / denominator if denominator < 0 else 0.0)
    return float(centre), float(refined)


def estimate_period(activation: np.ndarray, raw_beats: np.ndarray) -> float:
    """Beat period (seconds): the tracker's own level, unless the beat probability
    clearly repeats at twice that period (the tracker mixed in off-beat eighths)."""
    acf = _autocorrelation(activation)
    raw_lag = float(np.median(np.diff(raw_beats))) * FPS
    value, lag = _acf_peak(acf, raw_lag)
    if 2 * raw_lag <= 1.6 * FPS:
        double_value, double_lag = _acf_peak(acf, 2 * raw_lag)
        if double_value > 1.25 * max(value, 1e-6):
            return double_lag / FPS
    return lag / FPS


LEVEL_FACTORS = (0.5, 1.0, 2.0, 3.0, 4.0)  # tracker interval = factor x beat


def note_contour(notes) -> tuple[np.ndarray, np.ndarray]:
    """(times, lowest pitch) of the onset groups of a list of notes."""
    ordered = sorted(notes, key=lambda n: n.start)
    times: list[float] = []
    lows: list[float] = []
    for item in ordered:
        if times and item.start - times[-1] <= 0.035:
            lows[-1] = min(lows[-1], item.pitch)
        else:
            times.append(item.start)
            lows.append(item.pitch)
    return np.asarray(times), np.asarray(lows, dtype=float)


def _figure_size(contour: tuple[np.ndarray, np.ndarray], start: float, end: float) -> int | None:
    """Do the notes around a tracker interval move in pairs (2) or in threes (3)?

    The pitch steps of a two-note figure (low-high-low-high, as in broken
    octaves or a melody with a repeated "bell") repeat every two notes, those
    of a three-note figure every three. Unlike timing, this is not disturbed
    by notes being played a little unevenly. None when there is no clear figure.
    """
    times, lows = contour
    span = end - start
    inside = (times >= start - span) & (times < end + span)
    pitches = lows[inside]
    if pitches.size < 9:
        return None
    steps = np.diff(pitches)
    energy = float(np.sum(steps * steps))
    if energy <= 0:
        return None
    pairs = float(np.sum(steps[:-2] * steps[2:])) / energy
    threes = float(np.sum(steps[:-3] * steps[3:])) / energy
    if pairs - threes > 0.3:
        return 2
    if threes - pairs > 0.3:
        return 3
    return None


NOTES_PER_BEAT = (1, 2, 3, 4, 6, 8)


def _note_pulse(times: np.ndarray, centre: float, window: float = 2.0) -> float | None:
    """The spacing most of the notes around ``centre`` are played at (seconds).

    None unless at least six gaps, and over a third of all gaps nearby, agree
    to within about 12%: a ritardando or a free passage has no steady pulse.
    """
    low = np.searchsorted(times, centre - window)
    high = np.searchsorted(times, centre + window)
    gaps = np.diff(times[low:high])
    gaps = gaps[(gaps >= 0.05) & (gaps <= 0.7)]
    if gaps.size < 8:
        return None
    ordered = np.sort(gaps)
    best_count, best_value = 0, 0.0
    for value in ordered:
        members = ordered[(ordered >= value / 1.12) & (ordered <= value * 1.12)]
        if members.size > best_count:
            best_count, best_value = members.size, float(np.median(members))
    if best_count < 6 or best_count < 0.35 * gaps.size:
        return None
    return best_value


def _fit_note_pulse(period: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Refine seconds per beat and count the regular note pulses in each beat."""
    frames = np.arange(period.size)
    step = FPS // 2
    anchors = frames[::step]
    locked = period[anchors].copy()
    subdivisions = np.zeros(anchors.size, dtype=int)
    for index, frame in enumerate(anchors):
        pulse = _note_pulse(times, frame / FPS)
        if pulse is None:
            continue
        count = min(NOTES_PER_BEAT, key=lambda n: abs(np.log(period[frame] / (n * pulse))))
        if abs(np.log(count * pulse / period[frame])) <= np.log(1.22):
            locked[index] = count * pulse
            subdivisions[index] = count
    nearest = np.minimum((frames + step // 2) // step, anchors.size - 1)
    return np.interp(frames, anchors, locked), subdivisions[nearest]


def _phase_support(
    activation: np.ndarray, times: np.ndarray, period: np.ndarray, subdivisions: np.ndarray
) -> np.ndarray:
    """Pool beat evidence across a run of evenly subdivided piano notes.

    An accented offbeat can win locally in syncopated music. If every onset
    gap spans a known number of subdivisions, we can follow each possible
    beat position through the whole run, even as the player changes tempo.
    """
    from scipy.ndimage import maximum_filter1d

    frames = np.arange(activation.size, dtype=float)
    support = np.zeros(activation.size)
    onsets = times * FPS
    onsets = onsets[(onsets >= 0) & (onsets <= frames[-1])]
    if onsets.size < 2:
        return support
    middle = np.rint((onsets[:-1] + onsets[1:]) / 2).astype(int)
    divisions = subdivisions[middle]
    gaps = np.diff(onsets) * divisions / (period[middle] * FPS)
    steps = np.rint(gaps).astype(int)
    # An irregular attack, change of subdivision or long rest ends the run.
    regular = (
        (divisions >= 2) & (steps >= 1) & (steps <= 2 * divisions)
        & (np.abs(gaps - steps) <= 0.2)
    )
    labels = np.where(regular, divisions, 0)
    bounds = np.r_[0, np.flatnonzero(np.diff(labels)) + 1, labels.size]
    peaks = maximum_filter1d(activation, 3)  # allow one tracker frame of onset error
    for first, end in zip(bounds[:-1], bounds[1:]):
        count = labels[first]
        if count == 0:
            continue
        pulses = np.r_[0, np.cumsum(steps[first:end])]
        cycles = pulses[-1] // count
        if cycles < 6:
            continue
        run = onsets[first:end + 1]
        ticks = np.interp(np.arange(cycles * count), pulses, run)
        evidence = np.interp(ticks, frames, peaks).reshape(cycles, count)
        phase = evidence.mean(axis=0)
        # Apply a correction only if several consecutive beats favor a
        # competing phase somewhere in this otherwise regular passage.
        local = evidence[:-2] + evidence[1:-1] + evidence[2:]
        if not np.any(local.max(axis=1) > local[:, phase.argmax()]):
            continue
        region = slice(int(np.ceil(run[0])), int(np.floor(run[-1])) + 1)
        position = np.interp(frames[region], run, pulses) % count
        support[region] = np.interp(position, np.arange(count + 1), np.r_[phase, phase[0]])
        support[region] -= phase.mean()
    return support


def _period_from_tracker(
    raw_beats: np.ndarray,
    global_period: float,
    frames: int,
    contour: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    """Seconds per beat at every frame, read from the tracker's own beat spacing.

    Beat trackers change level within a piece: in La Campanella the intro is
    followed in eighth notes, the theme in dotted quarters (three times
    slower). Each interval between tracked beats is therefore taken as
    1/2, 1, 2, 3 or 4 beats, whichever keeps the tempo near the piece's main
    pulse. When an interval could be two long beats or three short ones, the
    notes decide: six notes moving in pairs are three beats of two (6/8),
    six notes moving in threes are two beats of three.
    """
    beats = np.unique(raw_beats)
    keep = [0]
    for index in range(1, beats.size):
        if beats[index] - beats[keep[-1]] >= 0.4 * global_period:
            keep.append(index)
    beats = beats[keep]
    if beats.size < 4:
        return np.full(frames, global_period)
    intervals = np.diff(beats)
    periods = np.empty(intervals.size)
    for index, interval in enumerate(intervals):
        ratio = interval / global_period
        options = [f for f in LEVEL_FACTORS if 0.68 <= ratio / f <= 1.47]
        if not options:
            options = [min(LEVEL_FACTORS, key=lambda f: abs(np.log(ratio / f)))]
        choice = min(options, key=lambda f: abs(np.log(ratio / f)))
        if 2.0 in options and 3.0 in options and contour is not None:
            figure = _figure_size(contour, beats[index], beats[index + 1])
            if figure == 2:
                choice = 3.0  # three beats, each a pair of notes
            elif figure == 3:
                choice = 2.0  # two beats, each a group of three
        periods[index] = interval / choice
    # Remove isolated odd readings, then follow gradual tempo changes.
    smoothed = np.array(
        [np.median(periods[max(0, i - 2) : i + 3]) for i in range(periods.size)]
    )
    smoothed = np.clip(smoothed, 0.6 * global_period, 1.6 * global_period)
    centres = (beats[:-1] + beats[1:]) / 2.0
    return np.interp(np.arange(frames) / FPS, centres, smoothed)


def steady_beats(
    activation: np.ndarray,
    raw_beats: np.ndarray,
    *,
    tightness: float = 6.0,
    contour: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Choose beat frames maximising beat probability with a steady tempo."""
    frames = activation.size
    global_period = estimate_period(activation, raw_beats)
    period = _period_from_tracker(raw_beats, global_period, frames, contour)
    score = np.empty(frames)
    back = np.full(frames, -1, dtype=np.int64)
    # Centre the activation so that weak frames are not worth adding as beats.
    gain = activation - 0.1
    if contour is not None:
        period, subdivisions = _fit_note_pulse(period, contour[0])
        # Give a consistent phase up to four beats' worth of supporting evidence.
        gain += 4.0 * _phase_support(activation, contour[0], period, subdivisions)
    period *= FPS
    for t in range(frames):
        beat_period = period[t]
        latest = int(t - 0.67 * beat_period)
        if latest < 0:
            score[t] = gain[t]
            continue
        earliest = max(0, int(t - 1.5 * beat_period))
        previous = np.arange(earliest, latest + 1)
        transition = score[previous] - tightness * np.log((t - previous) / beat_period) ** 2
        best = int(np.argmax(transition))
        if transition[best] > 0:
            score[t] = gain[t] + transition[best]
            back[t] = previous[best]
        else:
            score[t] = gain[t]
    # The last beat lies within one period of the end of the piece.
    tail = max(0, frames - int(period[-1] * 1.2))
    current = tail + int(np.argmax(score[tail:]))
    path = []
    while current >= 0:
        path.append(current)
        current = back[current]
    beats = np.asarray(path[::-1], dtype=float) / FPS
    # Keep the raw beats if decoding collapsed (e.g. near-silent recordings).
    active_seconds = float(np.sum(activation > 0.05)) / FPS
    if beats.size < 3 or beats.size < 0.3 * active_seconds / (global_period + 1e-9):
        return raw_beats
    return beats


def downbeat_strength(beats: np.ndarray, downbeat_activation: np.ndarray) -> np.ndarray:
    """Downbeat probability at each beat (max within +-40 ms)."""
    frames = np.clip(np.round(beats * FPS).astype(int), 0, downbeat_activation.size - 1)
    return np.array([downbeat_activation[max(0, f - 2) : f + 3].max() for f in frames])


def meter_scores(
    strength: np.ndarray,
    beat_period: float,
    emphasis: np.ndarray | None = None,
    gap_shares: dict[int, float] | None = None,
    repetition: dict[int, float] | None = None,
) -> list[tuple[float, int, int]]:
    """Score every (beats per bar, phase); higher is better.

    ``gap_shares`` says how often the tracker's bar marks are that many beats
    apart, and ``repetition`` how strongly the accompaniment's accents repeat
    at that distance. Unlike the other evidence, neither depends on the bar
    lines staying in step through the whole piece.
    """
    count = strength.size
    informative = count > 0 and float(np.mean(strength > 0.5)) < 0.6
    results = []
    for meter_length in METER_CANDIDATES:
        if count < 2 * meter_length:
            continue
        bar_seconds = meter_length * beat_period
        bar_prior = -0.12 * abs(np.log2(bar_seconds / 2.2))
        for phase in range(meter_length):
            mask = np.zeros(count, dtype=bool)
            mask[phase::meter_length] = True
            score = bar_prior + METER_PRIOR[meter_length]
            if gap_shares:
                score += GAP_WEIGHT * gap_shares.get(meter_length, 0.0)
            if repetition:
                score += REPETITION_WEIGHT * repetition.get(meter_length, 0.0)
            if informative:
                score += float(strength[mask].mean() - strength[~mask].mean())
            if emphasis is not None and emphasis.size == count and emphasis.max() > 0:
                normalised = emphasis / emphasis.max()
                score += 0.5 * float(normalised[mask].mean() - normalised[~mask].mean())
            results.append((score, meter_length, phase))
    results.sort(reverse=True)
    return results


def choose_downbeats(beats: np.ndarray, strength: np.ndarray) -> np.ndarray:
    """Downbeats from the downbeat probability alone (refined later with notes)."""
    if beats.size < 8:
        return beats[:1]
    scores = meter_scores(strength, float(np.median(np.diff(beats))))
    if not scores:
        return beats[:1]
    _, meter_length, phase = scores[0]
    return beats[phase::meter_length]


def _librosa_beats(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    import librosa

    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    _, beats = librosa.beat.beat_track(y=mono, sr=sample_rate, units="time")
    return np.asarray(beats, dtype=float)
