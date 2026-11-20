"""Tests for score export and optional MuseScore handling."""

from __future__ import annotations

from pathlib import Path

import pytest
from music21 import note, stream

import exporters
from exporters import ExportError, export_musicxml, export_pdf


def make_export_score() -> stream.Score:
    score = stream.Score()
    part = stream.Part()
    for pitch_name in ("C4", "D4", "E4", "G4"):
        part.append(note.Note(pitch_name, quarterLength=1.0))
    score.insert(0, part)
    return score


def test_exports_generated_score_to_musicxml(tmp_path: Path) -> None:
    destination = tmp_path / "test_score.musicxml"
    result = export_musicxml(make_export_score(), destination)
    assert result == destination
    assert destination.is_file()
    assert "<score-partwise" in destination.read_text(encoding="utf-8")


def test_preview_renders_in_fresh_threads() -> None:
    """Streamlit renders in a new thread each time; Verovio must keep its fonts."""
    import threading

    from exporters import render_svg_pages, score_to_musicxml_bytes

    xml = score_to_musicxml_bytes(make_export_score())
    results: list[int] = []

    def render() -> None:
        results.append(len(render_svg_pages(xml)))

    for _ in range(2):
        worker = threading.Thread(target=render)
        worker.start()
        worker.join()
    assert results == [1, 1]


def test_handles_missing_musescore_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = export_musicxml(make_export_score(), tmp_path / "source.musicxml")
    monkeypatch.setattr(exporters, "find_musescore_executable", lambda: None)
    with pytest.raises(ExportError, match="MuseScore was not found"):
        export_pdf(source, tmp_path / "score.pdf")

