"""Small data types shared by the transcription pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction


@dataclass
class NoteEvent:
    """A note heard in the recording, with times in seconds."""

    start: float
    end: float
    pitch: int
    velocity: int = 80
    part: str = "piano"
    rolled: bool = False  # part of a rolled (arpeggiated) chord

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class PedalEvent:
    """A sustain-pedal press, with times in seconds."""

    start: float
    end: float


@dataclass
class QuantizedNote:
    """A note placed on the musical grid, with times in beats."""

    start: Fraction
    end: Fraction
    pitch: int
    velocity: int = 80
    part: str = "piano"
    staff: int = 0
    # Unquantized beat positions are kept for later decisions (e.g. legato).
    raw_start: float = 0.0
    raw_end: float = 0.0
    rolled: bool = False  # part of a rolled (arpeggiated) chord

    @property
    def duration(self) -> Fraction:
        return self.end - self.start

    def copy(self, **changes) -> "QuantizedNote":
        return replace(self, **changes)


@dataclass
class ChordLabel:
    """A chord symbol that starts at a beat position."""

    start: Fraction
    end: Fraction
    root: int  # pitch class 0-11
    quality: str  # e.g. "maj", "min", "7", "maj7", "min7", "dim", "sus4"
    bass: int | None = None  # pitch class of a slash-chord bass, if any

    def copy(self, **changes) -> "ChordLabel":
        return replace(self, **changes)


