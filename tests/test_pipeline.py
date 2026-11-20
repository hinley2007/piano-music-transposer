"""Tests for the notation stage of the pipeline using made-up analyses."""

from __future__ import annotations

import numpy as np

from config import LAYOUT_EASY_PIANO, LAYOUT_LEAD_SHEET, LAYOUT_PIANO_VOCAL, MODE_PIANO, MODE_SONG
from exporters import score_to_musicxml_bytes
from music_types import NoteEvent
from pipeline import Analysis, NotationOptions, render_score

BEAT = 0.5  # 120 BPM


def make_analysis(mode: str, parts: dict[str, list[NoteEvent]]) -> Analysis:
    beats = np.arange(0, 40) * BEAT
    strength = np.where(np.arange(beats.size) % 4 == 0, 0.9, 0.1)
    return Analysis(
        mode=mode,
        parts=parts,
        pedals=[],
        beat_times=beats,
        downbeat_times=beats[::4],
        duration=float(beats[-1]),
        tuning_semitones=0.0,
        chroma=np.zeros((12, 0)),
        chroma_times=np.zeros(0),
        downbeat_strength=strength,
    )


def c_major_piano() -> list[NoteEvent]:
    notes = []
    melody = [72, 74, 76, 77, 79, 77, 76, 74, 72, 76, 79, 84, 79, 76, 72, 72]
    for index, pitch in enumerate(melody):
        start = index * BEAT
        notes.append(NoteEvent(start, start + BEAT * 0.9, pitch, 90, "piano"))
        if index % 2 == 0:
            notes.append(NoteEvent(start, start + BEAT * 1.8, 48 if index % 4 == 0 else 43, 70, "piano"))
    return notes


def test_piano_score_has_right_key_meter_and_notes() -> None:
    result = render_score(make_analysis(MODE_PIANO, {"piano": c_major_piano()}), NotationOptions())
    assert result.key.display_name == "C major"
    assert result.time_signature == "4/4"
    assert round(result.bpm) == 120
    assert result.note_count == 24
    hands = {n.staff for n in result.quantized["piano"] if n.pitch < 50}
    assert hands == {1}  # bass notes in the left hand


def test_transposes_by_semitones_and_respells_the_key() -> None:
    analysis = make_analysis(MODE_PIANO, {"piano": c_major_piano()})
    result = render_score(analysis, NotationOptions(transpose_semitones=2))
    assert result.key.display_name == "D major"
    assert min(n.pitch for n in result.quantized["piano"]) == 45
    xml = score_to_musicxml_bytes(result.score).decode("utf-8")
    assert "<fifths>2</fifths>" in xml


def test_transposes_to_a_chosen_key() -> None:
    analysis = make_analysis(MODE_PIANO, {"piano": c_major_piano()})
    result = render_score(analysis, NotationOptions(target_tonic="B-"))
    assert result.key.display_name == "B♭ major"
    assert result.original_key.display_name == "C major"


def test_every_detected_piano_note_is_written_even_when_very_short() -> None:
    """Regression for the La Campanella report: staccato notes were thrown away."""
    notes = []
    for index in range(64):  # 16ths at 120 BPM, each held for only 30 ms
        start = index * BEAT / 2
        notes.append(NoteEvent(start, start + 0.03, 72 + index % 7, 60 + index % 20, "piano"))
        if index % 4 == 0:
            notes.append(NoteEvent(start, start + 0.03, 48, 70, "piano"))
    result = render_score(make_analysis(MODE_PIANO, {"piano": notes}), NotationOptions())
    assert result.detected_count == len(notes)
    assert result.written_count == len(notes)


def test_chord_names_are_not_counted_as_written_notes() -> None:
    notes = c_major_piano()
    analysis = make_analysis(MODE_PIANO, {"piano": notes})
    # A little harmony for the chord finder: C major throughout.
    times = np.arange(0, 8.0, 0.05)
    chroma = np.zeros((12, times.size))
    chroma[[0, 4, 7], :] = 1.0
    analysis.chroma, analysis.chroma_times = chroma, times
    result = render_score(analysis, NotationOptions(show_chords=True))
    assert result.chords  # chord names were added...
    assert result.written_count == len(notes)  # ...but only real notes are counted


def test_sensitivity_is_applied_when_writing_not_when_analysing() -> None:
    loud = [NoteEvent(i * BEAT, i * BEAT + 0.4, 60 + i % 5, 90, "piano") for i in range(16)]
    faint = [NoteEvent(i * BEAT, i * BEAT + 0.4, 84, 20, "piano") for i in range(0, 16, 4)]
    analysis = make_analysis(MODE_PIANO, {"piano": loud + faint})
    everything = render_score(analysis, NotationOptions(sensitivity="Detailed (more notes)"))
    clean = render_score(analysis, NotationOptions(sensitivity="Clean (fewer notes)"))
    assert everything.written_count == 20
    assert clean.written_count == 16
    assert len(analysis.parts["piano"]) == 20  # the analysis itself is untouched


def test_six_eight_is_recognised_from_eighth_note_beats() -> None:
    eighth = 0.3
    notes = []
    for index in range(96):  # 16 bars: melody on every eighth, chords on 1 and 4
        start = index * eighth
        notes.append(NoteEvent(start, start + 0.05, 80 + index % 4, 70, "piano"))
        notes.append(NoteEvent(start + eighth / 2, start + eighth / 2 + 0.05, 92, 70, "piano"))
        if index % 6 == 0:
            notes += [NoteEvent(start, start + 0.05, p, 70, "piano") for p in (44, 56, 63)]
        elif index % 6 == 3:
            notes += [NoteEvent(start, start + 0.05, p, 70, "piano") for p in (59, 63, 68)]
    beats = np.arange(0, 100) * eighth
    analysis = Analysis(
        mode=MODE_PIANO,
        parts={"piano": notes},
        pedals=[],
        beat_times=beats,
        downbeat_times=beats[::6],
        duration=float(beats[-1]),
        tuning_semitones=0.0,
        chroma=np.zeros((12, 0)),
        chroma_times=np.zeros(0),
        downbeat_strength=np.where(np.arange(beats.size) % 6 == 0, 0.9, 0.1),
    )
    result = render_score(analysis, NotationOptions())
    assert result.time_signature == "6/8"
    assert result.tempo_text == "♩. = 67"
    assert result.written_count == len(notes)
    # The low chords belong to the left hand, the figure above to the right.
    assert all(n.staff == 1 for n in result.quantized["piano"] if n.pitch <= 68)
    assert all(n.staff == 0 for n in result.quantized["piano"] if n.pitch >= 80)


def song_parts() -> dict[str, list[NoteEvent]]:
    melody = [NoteEvent(i * BEAT, i * BEAT + 0.45, p, 90, "melody") for i, p in enumerate([67, 69, 71, 72] * 4)]
    bass = [NoteEvent(i * 4 * BEAT, i * 4 * BEAT + 1.9, 36, 90, "bass") for i in range(4)]
    chords = [
        NoteEvent(i * BEAT, i * BEAT + 0.45, p, 70, "accompaniment")
        for i in range(16)
        for p in (60, 64, 67)
    ]
    return {"melody": melody, "bass": bass, "accompaniment": chords}


def test_song_layouts() -> None:
    analysis = make_analysis(MODE_SONG, song_parts())
    easy = render_score(analysis, NotationOptions(layout=LAYOUT_EASY_PIANO, show_chords=False))
    assert [p.partName for p in easy.score.parts] == ["Piano", "Piano"]  # one grand staff
    lead = render_score(analysis, NotationOptions(layout=LAYOUT_LEAD_SHEET, show_chords=False))
    assert [p.partName for p in lead.score.parts] == ["Melody"]
    full = render_score(analysis, NotationOptions(layout=LAYOUT_PIANO_VOCAL, show_chords=False))
    assert [p.partName for p in full.score.parts] == ["Melody", "Piano", "Piano"]
    assert [n.pitch for n in lead.lines["melody"][:4]] == [67, 69, 71, 72]
