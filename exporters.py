"""MusicXML, MIDI, PDF (MuseScore) and SVG preview (Verovio) export helpers."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Sequence

from music21 import stream

from config import MUSESCORE_ENV_VAR
from music_types import NoteEvent

LOGGER = logging.getLogger(__name__)


class ExportError(RuntimeError):
    """Raised when a score cannot be exported."""


# ---------------------------------------------------------------------------
# In-memory exports
# ---------------------------------------------------------------------------


def score_to_musicxml_bytes(score: stream.Score) -> bytes:
    """Serialize a score to uncompressed MusicXML."""
    import contextlib
    import io

    from music21.musicxml.m21ToXml import GeneralObjectExporter

    try:
        # music21 prints a notice to stderr for every beam group it has to
        # tidy up in a dense bar; they are harmless and flood the terminal.
        with contextlib.redirect_stderr(io.StringIO()):
            data = GeneralObjectExporter(score).parse()
    except Exception as exc:
        LOGGER.exception("MusicXML export failed")
        raise ExportError("The MusicXML sheet music could not be created.") from exc
    if isinstance(data, str):
        data = data.encode("utf-8")
    return data


def score_to_midi_bytes(score: stream.Score) -> bytes:
    """Serialize the notated (quantized) score to a MIDI file."""
    from music21.midi.translate import streamToMidiFile

    try:
        midi_file = streamToMidiFile(score)
        return midi_file.writestr()
    except Exception as exc:
        LOGGER.exception("MIDI export failed")
        raise ExportError("The MIDI file could not be created.") from exc


GM_PROGRAMS = {"melody": 73, "bass": 33, "accompaniment": 0, "piano": 0, "instrument": 0}


def performance_midi_bytes(parts: dict[str, Sequence[NoteEvent]]) -> bytes:
    """MIDI with the original (unquantized) timing - useful in a DAW."""
    import io

    import pretty_midi

    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    for name, notes in parts.items():
        track = pretty_midi.Instrument(program=GM_PROGRAMS.get(name, 0), name=name.title())
        for item in notes:
            if item.end > item.start:
                track.notes.append(
                    pretty_midi.Note(
                        velocity=int(item.velocity),
                        pitch=int(item.pitch),
                        start=float(item.start),
                        end=float(item.end),
                    )
                )
        midi.instruments.append(track)
    buffer = io.BytesIO()
    midi.write(buffer)
    return buffer.getvalue()


_VEROVIO_LOCK = threading.Lock()


def render_svg_pages(musicxml: bytes, *, page_width: int = 2100, scale: int = 40) -> list[str]:
    """Engrave MusicXML into SVG pages with Verovio (no external program needed)."""
    try:
        import verovio
    except ImportError as exc:  # pragma: no cover - install problem
        raise ExportError("Verovio is not installed, so the preview is unavailable.") from exc
    # Verovio keeps its music fonts in shared state that is lost when the thread
    # that loaded them ends (Streamlit runs each rerun in a new thread), so the
    # fonts are loaded explicitly for every render.
    resources = os.path.join(os.path.dirname(verovio.__file__), "data")
    with _VEROVIO_LOCK:
        toolkit = verovio.toolkit(False)
        if not toolkit.setResourcePath(resources):
            raise ExportError("Verovio's music fonts could not be loaded.")
        toolkit.setOptions(
            {
                "pageWidth": page_width,
                "pageHeight": 2970,
                "scale": scale,
                "adjustPageHeight": True,
                "breaks": "auto",
                "footer": "none",
                "header": "auto",
            }
        )
        if not toolkit.loadData(musicxml.decode("utf-8")):
            raise ExportError("The preview could not be drawn.")
        return [toolkit.renderToSVG(page) for page in range(1, toolkit.getPageCount() + 1)]


# ---------------------------------------------------------------------------
# File exports (kept for scripts and tests)
# ---------------------------------------------------------------------------


def export_midi(score: stream.Score, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(score_to_midi_bytes(score))
    return output_path


def export_musicxml(score: stream.Score, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = score_to_musicxml_bytes(score)
    if not data:
        raise ExportError("The MusicXML export is empty.")
    output_path.write_bytes(data)
    return output_path


def _is_executable_file(path: Path) -> bool:
    if not path.is_file():
        return False
    return os.name == "nt" or os.access(path, os.X_OK)


def find_musescore_executable() -> Path | None:
    """Find MuseScore from MUSESCORE_PATH, PATH, or common install locations."""
    configured = os.getenv(MUSESCORE_ENV_VAR)
    if configured:
        candidate = Path(configured).expanduser()
        if _is_executable_file(candidate):
            return candidate
        LOGGER.warning("%s does not point to an executable: %s", MUSESCORE_ENV_VAR, candidate)

    for command in (
        "musescore4",
        "mscore4",
        "MuseScore4",
        "musescore3",
        "mscore3",
        "MuseScore3",
        "musescore",
        "mscore",
    ):
        located = shutil.which(command)
        if located:
            return Path(located)

    candidates: list[Path] = []
    system = platform.system()
    if system == "Windows":
        for variable in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
            base = os.getenv(variable)
            if base:
                base_path = Path(base)
                candidates.extend(
                    [
                        base_path / "MuseScore 4" / "bin" / "MuseScore4.exe",
                        base_path / "MuseScore 3" / "bin" / "MuseScore3.exe",
                        base_path / "Programs" / "MuseScore 4" / "bin" / "MuseScore4.exe",
                    ]
                )
    elif system == "Darwin":
        candidates.extend(
            [
                Path("/Applications/MuseScore 4.app/Contents/MacOS/mscore"),
                Path("/Applications/MuseScore 3.app/Contents/MacOS/mscore"),
            ]
        )
    return next((path for path in candidates if _is_executable_file(path)), None)


def export_pdf(
    musicxml_path: Path,
    output_path: Path,
    musescore_executable: Path | None = None,
) -> Path:
    """Use MuseScore's command-line interface to render MusicXML as PDF."""
    musicxml_path = Path(musicxml_path)
    output_path = Path(output_path)
    executable = (
        Path(musescore_executable)
        if musescore_executable is not None
        else find_musescore_executable()
    )
    if executable is None or not _is_executable_file(executable):
        raise ExportError(
            "MuseScore was not found. Install MuseScore or set MUSESCORE_PATH to "
            "its executable to enable PDF export."
        )
    if not musicxml_path.is_file():
        raise ExportError("The MusicXML source file for PDF export is missing.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            [str(executable), "-o", str(output_path), str(musicxml_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        LOGGER.exception("MuseScore could not be started")
        raise ExportError("MuseScore could not complete the PDF conversion.") from exc

    if completed.returncode != 0:
        LOGGER.error(
            "MuseScore PDF export failed (%s): %s",
            completed.returncode,
            completed.stderr.strip(),
        )
        raise ExportError(
            "MuseScore reported an error while creating the PDF. Check the log for details."
        )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise ExportError("MuseScore finished but did not create a usable PDF.")
    return output_path
