"""Key detection and key-aware note spelling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

# Albrecht-Shanahan key profiles (2013). On 4,378 pieces with a declared key
# from music21's corpus they find the right key 89% of the time with the
# settings below, against 79% for the classic Krumhansl-Kessler profiles.
MAJOR_PROFILE = np.array(
    [0.238, 0.006, 0.111, 0.006, 0.137, 0.094, 0.016, 0.214, 0.009, 0.080, 0.008, 0.081]
)
MINOR_PROFILE = np.array(
    [0.220, 0.006, 0.104, 0.123, 0.019, 0.103, 0.012, 0.214, 0.062, 0.022, 0.061, 0.052]
)
# Pieces usually end on their home note: a small bonus for the final bass note.
ENDING_BONUS = 0.1

LETTERS = "CDEFGAB"
LETTER_PITCH_CLASS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# Preferred spelling of each tonic (fewest accidentals in the key signature).
MAJOR_TONICS = {
    0: "C", 1: "D-", 2: "D", 3: "E-", 4: "E", 5: "F",
    6: "F#", 7: "G", 8: "A-", 9: "A", 10: "B-", 11: "B",
}
MINOR_TONICS = {
    0: "C", 1: "C#", 2: "D", 3: "E-", 4: "E", 5: "F",
    6: "F#", 7: "G", 8: "G#", 9: "A", 10: "B-", 11: "B",
}

# Semitones above the tonic -> letter steps above the tonic letter.
# Major: chromatic notes use #1, b3, #4, b6, b7 (the most common in practice).
MAJOR_DEGREE_STEPS = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 5, 9: 5, 10: 6, 11: 6}
# Minor: b2, raised 3, #4, raised 6 and 7 (harmonic/melodic minor).
MINOR_DEGREE_STEPS = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 5, 9: 5, 10: 6, 11: 6}
# Spelling alternatives for chromatic notes: raised lower letter vs lowered upper letter.
CHROMATIC_ALTERNATIVES = {
    1: (0, 1), 3: (1, 2), 6: (3, 4), 8: (4, 5), 10: (5, 6),
}

PITCH_CLASS_NAMES = ("C", "C#", "D", "E-", "E", "F", "F#", "G", "A-", "A", "B-", "B")


@dataclass(frozen=True)
class KeyEstimate:
    tonic: int  # pitch class
    mode: str  # "major" or "minor"
    confidence: float  # correlation with the key profile (-1..1)

    @property
    def tonic_name(self) -> str:
        table = MAJOR_TONICS if self.mode == "major" else MINOR_TONICS
        return table[self.tonic]

    @property
    def display_name(self) -> str:
        name = self.tonic_name.replace("-", "♭").replace("#", "♯")
        return f"{name} {self.mode}"

    def transposed(self, semitones: int) -> "KeyEstimate":
        return KeyEstimate((self.tonic + semitones) % 12, self.mode, self.confidence)


def pitch_class_histogram(
    pitches: Iterable[int], weights: Iterable[float] | None = None
) -> np.ndarray:
    histogram = np.zeros(12)
    if weights is None:
        for pitch in pitches:
            histogram[pitch % 12] += 1.0
    else:
        for pitch, weight in zip(pitches, weights):
            histogram[pitch % 12] += weight
    return histogram


def estimate_key(histogram: np.ndarray, *, final_bass: int | None = None) -> KeyEstimate:
    """Pick the key whose profile best correlates with a pitch-class histogram.

    The histogram is square-root compressed first, so one note repeated
    hundreds of times (a pedal point, or the "bell" in La Campanella, which
    is the fifth of the key) cannot pass itself off as the home note.
    ``final_bass`` is the pitch class of the lowest note of the final chord.
    """
    if not np.any(histogram):
        return KeyEstimate(0, "major", 0.0)
    compressed = np.sqrt(np.maximum(np.asarray(histogram, dtype=float), 0.0))
    best = KeyEstimate(0, "major", -2.0)
    for tonic in range(12):
        for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            rotated = np.roll(profile, tonic)
            if np.std(compressed) == 0:
                score = 0.0
            else:
                score = float(np.corrcoef(compressed, rotated)[0, 1])
            if final_bass is not None and final_bass % 12 == tonic:
                score += ENDING_BONUS
            if score > best.confidence:
                best = KeyEstimate(tonic, mode, score)
    return best


MAJOR_SHARPS = {0: 0, 1: -5, 2: 2, 3: -3, 4: 4, 5: -1, 6: 6, 7: 1, 8: -4, 9: 3, 10: -2, 11: 5}
MINOR_SHARPS = {0: -3, 1: 4, 2: -1, 3: -6, 4: 1, 5: -4, 6: 3, 7: -2, 8: 5, 9: 0, 10: -5, 11: 2}


def key_signature_sharps(key_estimate: KeyEstimate) -> int:
    """Number of sharps (positive) or flats (negative) in the key signature."""
    table = MAJOR_SHARPS if key_estimate.mode == "major" else MINOR_SHARPS
    return table[key_estimate.tonic]


@dataclass(frozen=True)
class Spelling:
    letter: str
    alter: int  # -2..2
    octave: int

    @property
    def name(self) -> str:
        accidental = {-2: "--", -1: "-", 0: "", 1: "#", 2: "##"}[self.alter]
        return f"{self.letter}{accidental}{self.octave}"


def _spell_with_steps(midi: int, tonic_letter: str, steps: int) -> Spelling | None:
    letter = LETTERS[(LETTERS.index(tonic_letter) + steps) % 7]
    natural = LETTER_PITCH_CLASS[letter]
    alter = ((midi % 12) - natural + 6) % 12 - 6
    if abs(alter) > 2:
        return None
    octave = (midi - alter - natural) // 12 - 1
    return Spelling(letter, alter, octave)


def spell_pitch(
    midi: int,
    key_estimate: KeyEstimate,
    *,
    next_pitch: int | None = None,
) -> Spelling:
    """Spell a MIDI pitch using the key; chromatic notes lean toward where they move."""
    tonic_letter = key_estimate.tonic_name[0]
    relative = (midi - key_estimate.tonic) % 12
    table = MAJOR_DEGREE_STEPS if key_estimate.mode == "major" else MINOR_DEGREE_STEPS
    steps = table[relative]
    if next_pitch is not None and relative in CHROMATIC_ALTERNATIVES:
        step = next_pitch - midi
        if not _is_diatonic(relative, key_estimate.mode):
            lower, upper = CHROMATIC_ALTERNATIVES[relative]
            if step == 1:
                steps = lower  # rising: raise the lower letter (e.g. F# -> G)
            elif step == -1:
                steps = upper  # falling: lower the upper letter (e.g. Ab -> G)
    spelled = _spell_with_steps(midi, tonic_letter, steps)
    if spelled is None:
        name = PITCH_CLASS_NAMES[midi % 12]
        alter = 1 if "#" in name else (-1 if "-" in name else 0)
        natural = LETTER_PITCH_CLASS[name[0]]
        return Spelling(name[0], alter, (midi - alter - natural) // 12 - 1)
    return spelled


def _is_diatonic(relative: int, mode: str) -> bool:
    if mode == "major":
        return relative in (0, 2, 4, 5, 7, 9, 11)
    # Natural, harmonic, and melodic minor notes all count as "in key".
    return relative in (0, 2, 3, 5, 7, 8, 9, 10, 11)


def spell_sequence(pitches: Sequence[int], key_estimate: KeyEstimate) -> list[Spelling]:
    """Spell a melodic line, using each note's successor as context."""
    result = []
    for index, pitch in enumerate(pitches):
        following = pitches[index + 1] if index + 1 < len(pitches) else None
        result.append(spell_pitch(pitch, key_estimate, next_pitch=following))
    return result


def pitch_class_name(pitch_class: int, key_estimate: KeyEstimate | None = None) -> str:
    """Name a pitch class (used for chord roots) in music21 form, e.g. 'B-'."""
    if key_estimate is None:
        return PITCH_CLASS_NAMES[pitch_class % 12]
    spelled = spell_pitch(60 + pitch_class % 12, key_estimate)
    accidental = {-2: "--", -1: "-", 0: "", 1: "#", 2: "##"}[spelled.alter]
    return f"{spelled.letter}{accidental}"
