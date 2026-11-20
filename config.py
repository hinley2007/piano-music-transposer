"""Shared configuration for Piano Music Transposer."""

from __future__ import annotations

from types import MappingProxyType

ALLOWED_AUDIO_EXTENSIONS = frozenset({".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"})
MAX_UPLOAD_SIZE_MB = 50
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

MUSESCORE_ENV_VAR = "MUSESCORE_PATH"

# What kind of recording is transcribed. The app only does piano; the older
# song / melody / other modes are still reachable from the command line.
MODE_PIANO = "piano"
MODE_SONG = "song"
MODE_MELODY = "melody"
MODE_OTHER = "other"

MODE_LABELS = {
    MODE_PIANO: "Piano recording",
    MODE_SONG: "Song with vocals / band",
    MODE_MELODY: "Single melody (voice or one instrument)",
    MODE_OTHER: "Other instrument (guitar, ensemble...)",
}

# Song-mode score layouts.
LAYOUT_PIANO_VOCAL = "piano_vocal"
LAYOUT_EASY_PIANO = "easy_piano"
LAYOUT_LEAD_SHEET = "lead_sheet"

LAYOUT_LABELS = {
    LAYOUT_EASY_PIANO: "Piano arrangement (melody + bass line + chord names)",
    LAYOUT_PIANO_VOCAL: "Melody staff + full piano accompaniment",
    LAYOUT_LEAD_SHEET: "Lead sheet (melody + chord names)",
}

# Tuning is only corrected when it is clearly off (in semitones).
TUNING_CORRECTION_THRESHOLD = 0.08

# Rhythm grid choices: beats can be split into 1, 2, 3, 4, 6, or 8 parts.
SUBDIVISIONS = (1, 2, 3, 4, 6, 8)

# Sensitivity: which detected notes are written in the sheet.
# For piano recordings only loudness is used (as a fraction of the loud-note
# level). Note length is NOT used: real staccato notes can last only 30 ms.
SENSITIVITY_PRESETS = {
    "Clean (fewer notes)": {"relative_velocity": 0.5},
    "Balanced": {"relative_velocity": 0.12},
    "Detailed (more notes)": {"relative_velocity": 0.0},
}
# Song accompaniment is the piano model listening to guitars, synths, strings...
# which produces more false notes, so it is filtered harder (and by length).
ACCOMPANIMENT_PRESETS = {
    "Clean (fewer notes)": {"relative_velocity": 0.45, "min_seconds": 0.09},
    "Balanced": {"relative_velocity": 0.32, "min_seconds": 0.075},
    "Detailed (more notes)": {"relative_velocity": 0.2, "min_seconds": 0.06},
}
DEFAULT_SENSITIVITY = "Balanced"

TIME_SIGNATURE_CHOICES = ("Auto", "2/4", "3/4", "4/4", "6/8", "12/8")
# Settings used only by the old Basic Pitch pipeline (music_processing.py),
# which the app no longer uses. Kept so that module and its tests still work.
DEFAULT_TEMPO_BPM = 120.0
DEFAULT_TIME_SIGNATURE = "4/4"
MIN_NOTE_DURATION_QUARTER_LENGTH = 0.125
SUPPORTED_QUANTIZATION = MappingProxyType(
    {"Quarter notes": 1.0, "Eighth notes": 0.5, "Sixteenth notes": 0.25}
)
KEY_CONFIDENCE_THRESHOLD = 0.25

TEMPO_AUTO = "Auto"
TEMPO_HALF = "Half the tempo (shorter note values)"
TEMPO_DOUBLE = "Double the tempo (longer note values)"
TEMPO_CHOICES = (TEMPO_AUTO, TEMPO_HALF, TEMPO_DOUBLE)
