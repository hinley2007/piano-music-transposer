"""The Streamlit app: piano only, and the sheet settings work (no ML models are run)."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from config import MODE_PIANO
from music_types import NoteEvent
from pipeline import Analysis

APP = str(Path(__file__).resolve().parents[1] / "app.py")
BEAT = 0.5  # 120 BPM


def wav_bytes(seconds: float = 3.0, rate: int = 22050) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    samples = (0.2 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    return buffer.getvalue()


def c_major_analysis() -> Analysis:
    """What the slow analysis would return for a short C major piece."""
    notes = []
    for index, pitch in enumerate([72, 74, 76, 77, 79, 77, 76, 74, 72, 76, 79, 84, 79, 76, 72, 72]):
        start = index * BEAT
        notes.append(NoteEvent(start, start + BEAT * 0.9, pitch, 90, "piano"))
        if index % 2 == 0:
            notes.append(NoteEvent(start, start + BEAT * 1.8, 48, 70, "piano"))
    beats = np.arange(0, 40) * BEAT
    return Analysis(
        mode=MODE_PIANO,
        parts={"piano": notes},
        pedals=[],
        beat_times=beats,
        downbeat_times=beats[::4],
        duration=float(beats[-1]),
        tuning_semitones=0.0,
        chroma=np.zeros((12, 0)),
        chroma_times=np.zeros(0),
        downbeat_strength=np.where(np.arange(beats.size) % 4 == 0, 0.9, 0.1),
    )


@pytest.fixture()
def app_with_sheet() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.file_uploader[0].upload("My_Etude in C.wav", wav_bytes(), "audio/wav")
    at.run()
    assert [button.label for button in at.button] == ["Transcribe"]
    upload = at.session_state["upload"]
    at.session_state["analysis"] = c_major_analysis()
    at.session_state["analysis_key"] = (upload["hash"], None, None)
    at.run()
    assert not at.exception
    return at


def metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def test_the_app_is_piano_only() -> None:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert at.title[0].value.endswith("Piano Music Transposer")
    assert not at.radio  # no "What is in the recording?" choice any more


def test_results_show_the_sheet_and_its_settings(app_with_sheet: AppTest) -> None:
    at = app_with_sheet
    assert not at.button  # the sheet is up to date: no Transcribe button
    assert metric(at, "Key") == "C major"
    assert metric(at, "Time signature") == "4/4"
    assert any("All 24 notes heard" in caption.value for caption in at.caption)
    assert at.text_input(key="title").value == "My Etude in C"
    labels = [download.proto.label for download in at.get("download_button")]
    assert {"MusicXML", "MIDI"} <= set(labels)


def test_transpose_lists_keys_and_survives_other_changes(app_with_sheet: AppTest) -> None:
    at = app_with_sheet
    transpose = at.selectbox(key="transpose")
    assert "C major (original key)" in transpose.options
    assert "D major (+2)" in transpose.options
    assert "B♭ major (−2)" in transpose.options

    transpose.set_value(2)
    at.run()
    assert metric(at, "Key") == "D major"
    assert any("Transposed from C major." in caption.value for caption in at.caption)

    at.radio(key="notes").set_value("Detailed (more notes)")
    at.run()
    assert at.selectbox(key="transpose").value == 2
    assert metric(at, "Key") == "D major"
