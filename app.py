"""Piano Music Transposer: turn a piano recording into accurate sheet music."""

from __future__ import annotations

import hashlib
import io
import logging
import math
import re
import tempfile
import time
from pathlib import Path
from typing import Callable

import streamlit as st

from audio_utils import (
    AudioConversionError,
    AudioValidationError,
    get_duration_seconds,
    validate_audio_file,
)
from config import (
    ALLOWED_AUDIO_EXTENSIONS,
    DEFAULT_SENSITIVITY,
    MODE_PIANO,
    SENSITIVITY_PRESETS,
    TEMPO_AUTO,
    TEMPO_CHOICES,
    TEMPO_DOUBLE,
    TEMPO_HALF,
    TIME_SIGNATURE_CHOICES,
    TUNING_CORRECTION_THRESHOLD,
)
from exporters import (
    ExportError,
    export_pdf,
    find_musescore_executable,
    render_svg_pages,
    score_to_midi_bytes,
    score_to_musicxml_bytes,
)
from keys import KeyEstimate
from pipeline import (
    Analysis,
    AnalysisOptions,
    NotationOptions,
    RenderResult,
    analyze_audio,
    quiet_library_warnings,
    render_score,
)
from synth import render_notes, to_wav_bytes
from transcription import NoNotesDetectedError, TranscriptionError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
quiet_library_warnings()
LOGGER = logging.getLogger(__name__)

APP_NAME = "Piano Music Transposer"

FRIENDLY_ERRORS = (
    AudioValidationError,
    AudioConversionError,
    NoNotesDetectedError,
    TranscriptionError,
    ExportError,
)

# "Transpose" lists each of the 12 keys once: up to 6 semitones up or 5 down.
TRANSPOSE_STEPS = tuple(range(6, -6, -1))
NOTE_LABELS = {name: name.split(" (")[0] for name in SENSITIVITY_PRESETS}
DETAILED = min(SENSITIVITY_PRESETS, key=lambda name: SENSITIVITY_PRESETS[name]["relative_velocity"])
TEMPO_LABELS = {
    TEMPO_AUTO: "Auto",
    TEMPO_HALF: "Half (shorter notes)",
    TEMPO_DOUBLE: "Double (longer notes)",
}
# The sidebar widgets keep the sheet settings in st.session_state under these keys.
SETTING_DEFAULTS = {
    "transpose": 0,
    "notes": DEFAULT_SENSITIVITY,
    "time_signature": "Auto",
    "tempo": TEMPO_AUTO,
    "simple_rhythm": False,
}
MAX_CACHED_RENDERS = 4

TIPS = """
- Solo piano recorded close and clean works best. Room echo, applause or other
  instruments cause mistakes.
- Notes missing? Choose **Detailed** under *Notes to write*. A note that is still
  missing was not heard (usually a very soft note inside a loud chord).
- Cadenzas and long trills have no steady beat, so they are written out note by
  note. **Simpler rhythm** makes them easier to read.
- To fix a small mistake, download the MusicXML and open it in MuseScore (free).
"""

st.set_page_config(page_title=APP_NAME, page_icon="\U0001f3b9", layout="wide")


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------


def _work_folder() -> Path:
    folder = Path(tempfile.gettempdir()) / "piano_music_transposer"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _store_upload(name: str, data: bytes) -> dict:
    digest = hashlib.sha1(data).hexdigest()[:16]
    current = st.session_state.get("upload")
    if current and current["hash"] == digest:
        return current
    safe_name = validate_audio_file(name, len(data))
    path = _work_folder() / f"{digest}{Path(safe_name).suffix}"
    if not path.exists():
        path.write_bytes(data)
    try:
        duration = get_duration_seconds(path)
    except Exception:
        duration = 0.0
    upload = {
        "hash": digest,
        "name": safe_name,
        "title": _default_title(Path(name).stem),  # from the name as the user wrote it
        "path": path,
        "duration": duration,
        "bytes": data,
    }
    st.session_state["upload"] = upload
    for key in ("analysis", "analysis_key", "renders", "pdfs"):
        st.session_state.pop(key, None)
    return upload


def _audio_format(name: str) -> str:
    suffix = Path(name).suffix.lower()
    return {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
    }.get(suffix, "audio/mpeg")


def _format_time(seconds: float) -> str:
    minutes, rest = divmod(int(round(seconds)), 60)
    return f"{minutes}:{rest:02d}"


def _setting(name: str):
    """Current value of a sheet setting (stored by its sidebar widget)."""
    return st.session_state.get(name, SETTING_DEFAULTS[name])


def _default_title(stem: str) -> str:
    title = re.sub(r"[\x00-\x1f]+", "", stem).replace("_", " ")
    return re.sub(r"\s+", " ", title).strip()[:60] or "Transcription"


def _download_name(title: str, key: KeyEstimate | None) -> str:
    """A file name that works everywhere, e.g. 'La Campanella (Bb minor)'."""
    name = f"{title} ({key.display_name})" if key else title
    name = name.replace("♭", "b").replace("♯", "#")
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "-", name).strip(" .-")
    return name or "sheet music"


def _transpose_label(steps: int, original_key: KeyEstimate | None) -> str:
    amount = f"+{steps}" if steps > 0 else f"−{-steps}"
    if original_key is None:
        return "Original key" if steps == 0 else f"{amount} semitones"
    if steps == 0:
        return f"{original_key.display_name} (original key)"
    return f"{original_key.transposed(steps).display_name} ({amount})"


# ---------------------------------------------------------------------------
# Transcription (slow, once per recording) and rendering (fast, per setting)
# ---------------------------------------------------------------------------


def _transcribe(upload: dict, analysis_key: tuple) -> None:
    _, start_seconds, end_seconds = analysis_key
    progress = st.progress(0.0, text="Starting...")
    started = time.perf_counter()
    try:
        analysis = analyze_audio(
            upload["path"],
            AnalysisOptions(mode=MODE_PIANO, start_seconds=start_seconds, end_seconds=end_seconds),
            lambda fraction, message: progress.progress(min(1.0, max(0.0, fraction)), text=message),
        )
    except FRIENDLY_ERRORS as exc:
        progress.empty()
        LOGGER.warning("Transcription stopped: %s", exc)
        st.error(str(exc))
        return
    except Exception:
        progress.empty()
        LOGGER.exception("Unexpected transcription failure")
        st.error("Something went wrong while transcribing. Details are in the terminal window.")
        return
    LOGGER.info(
        "Transcribed %s in %.0f s, steps: %s",
        upload["name"],
        time.perf_counter() - started,
        analysis.timings,
    )
    st.session_state["analysis"] = analysis
    st.session_state["analysis_key"] = analysis_key
    st.session_state["renders"] = {}
    st.session_state["pdfs"] = {}
    st.rerun()


def _on_white(svg: str) -> str:
    """Give the engraved page a white background (readable in dark mode too)."""
    opening = svg.find("<svg")
    if opening < 0:
        return svg
    tag_end = svg.find(">", opening) + 1
    background = '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>'
    return svg[:tag_end] + background + svg[tag_end:]


def _listen_audio(midi_bytes: bytes) -> bytes:
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    notes = [
        (n.start, n.end, n.pitch, n.velocity)
        for instrument in midi.instruments
        for n in instrument.notes
    ]
    return to_wav_bytes(render_notes(notes))


def _build_render(analysis: Analysis, options: NotationOptions) -> dict:
    started = time.perf_counter()
    result = render_score(analysis, options)
    musicxml = score_to_musicxml_bytes(result.score)
    midi = score_to_midi_bytes(result.score)
    try:
        pages = [_on_white(svg) for svg in render_svg_pages(musicxml)]
        preview_error = None
    except ExportError as exc:
        pages, preview_error = [], str(exc)
    bundle = {
        "result": result,
        "musicxml": musicxml,
        "midi": midi,
        "pages": pages,
        "preview_error": preview_error,
        "listen": _listen_audio(midi),
    }
    LOGGER.info("Wrote the sheet music in %.1f s", time.perf_counter() - started)
    return bundle


def _render(analysis: Analysis, options: NotationOptions) -> dict | None:
    """The sheet music for these settings. The last few are kept, so going back is instant."""
    renders = st.session_state.setdefault("renders", {})
    pdfs = st.session_state.setdefault("pdfs", {})
    key = repr(options)
    bundle = renders.pop(key, None)
    if bundle is None:
        with st.spinner("Writing the sheet music..."):
            try:
                bundle = _build_render(analysis, options)
            except FRIENDLY_ERRORS as exc:
                st.error(str(exc))
                return None
            except Exception:
                LOGGER.exception("Unexpected notation failure")
                st.error("The sheet music could not be written. Details are in the terminal window.")
                return None
    renders[key] = bundle  # most recently used last
    while len(renders) > MAX_CACHED_RENDERS:
        oldest = next(iter(renders))
        renders.pop(oldest)
        pdfs.pop(oldest, None)
    return bundle


def _pdf_maker(musicxml: bytes, cache: dict, cache_key: str, musescore: Path) -> Callable[[], bytes]:
    """Engrave the PDF with MuseScore only when the button is clicked (it takes a few seconds)."""

    def make() -> bytes:
        if cache_key not in cache:
            try:
                folder = Path(tempfile.mkdtemp(dir=_work_folder()))
                xml_path = folder / "score.musicxml"
                xml_path.write_bytes(musicxml)
                cache[cache_key] = export_pdf(xml_path, folder / "score.pdf", musescore).read_bytes()
            except Exception:
                LOGGER.exception("PDF export failed")
                raise
        return cache[cache_key]

    return make


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------


def _choose_part(upload: dict) -> tuple[float | None, float | None]:
    """Optionally transcribe only a section of a long recording."""
    total = math.ceil(float(upload["duration"]))
    if total <= 1:
        return None, None
    with st.expander("Transcribe only part of the recording"):
        start, end = st.slider(
            "Part to transcribe",
            min_value=0,
            max_value=total,
            value=(0, total),
            step=1,
            format="%d s",
            key="part",
        )
        st.caption(f"From {_format_time(start)} to {_format_time(min(end, upload['duration']))}")
    return (float(start) if start > 0 else None), (float(end) if end < total else None)


def _sheet_settings(default_title: str, original_key: KeyEstimate | None) -> None:
    """Sidebar controls.

    Their values are read with ``_setting`` before the score is written, and the
    widgets are drawn afterwards, once the key of the recording is known.
    """
    with st.sidebar:
        st.header("Sheet music")
        st.selectbox(
            "Transpose",
            TRANSPOSE_STEPS,
            index=TRANSPOSE_STEPS.index(_setting("transpose")),
            format_func=lambda steps: _transpose_label(steps, original_key),
            key="transpose",
            help="Pick a new key. The number is how many semitones the music moves "
            "up (+) or down (−).",
        )
        presets = list(SENSITIVITY_PRESETS)
        st.radio(
            "Notes to write",
            presets,
            index=presets.index(_setting("notes")),
            format_func=NOTE_LABELS.get,
            horizontal=True,
            key="notes",
            help="Balanced writes every note except the very faintest, Detailed writes "
            "all of them, and Clean leaves out quiet notes.",
        )
        st.selectbox(
            "Time signature",
            TIME_SIGNATURE_CHOICES,
            index=TIME_SIGNATURE_CHOICES.index(_setting("time_signature")),
            key="time_signature",
            help="Found automatically. Pick one yourself if the bar lines look shifted.",
        )
        st.selectbox(
            "Tempo",
            TEMPO_CHOICES,
            index=TEMPO_CHOICES.index(_setting("tempo")),
            format_func=TEMPO_LABELS.get,
            key="tempo",
            help="If the notes look twice as fast or slow as they should, halve or "
            "double the tempo.",
        )
        st.toggle(
            "Simpler rhythm",
            value=_setting("simple_rhythm"),
            key="simple_rhythm",
            help="No triplets or 32nd notes: easier to read, a little less exact.",
        )
        st.text_input(
            "Title",
            value=st.session_state.get("title", default_title),
            max_chars=80,
            key="title",
        )
        with st.expander("Tips"):
            st.markdown(TIPS)


def _summary(result: RenderResult, analysis: Analysis, options: NotationOptions) -> str:
    heard, written = result.detected_count, result.written_count
    if written >= heard:
        lines = [f"All {heard:,} notes heard in the recording are written."]
    else:
        lines = [f"{written:,} of the {heard:,} notes heard in the recording are written."]
        if options.sensitivity != DETAILED:
            lines.append("Choose Detailed under Notes to write to include the quietest ones.")
    if result.key.tonic != result.original_key.tonic:
        lines.append(f"Transposed from {result.original_key.display_name}.")
    if abs(analysis.tuning_semitones) >= TUNING_CORRECTION_THRESHOLD:
        lines.append(
            "The recording is not tuned to A = 440 Hz; this was corrected before transcribing."
        )
    return " ".join(lines)


def _show_results(upload: dict, analysis: Analysis, part: tuple) -> None:
    default_title = upload.get("title") or _default_title(Path(upload["name"]).stem)
    title = str(st.session_state.get("title", default_title)).strip() or default_title
    options = NotationOptions(
        title=title,
        time_signature=_setting("time_signature"),
        tempo_choice=_setting("tempo"),
        transpose_semitones=int(_setting("transpose")),
        show_chords=False,
        simplify_rhythm=bool(_setting("simple_rhythm")),
        sensitivity=_setting("notes"),
    )
    bundle = _render(analysis, options)
    result: RenderResult | None = bundle["result"] if bundle else None
    _sheet_settings(default_title, result.original_key if result else None)
    if bundle is None or result is None:
        return

    st.divider()
    cards = st.columns(5)
    cards[0].metric("Key", result.key.display_name, border=True)
    cards[1].metric("Time signature", result.time_signature, border=True)
    cards[2].metric("Tempo", result.tempo_text, border=True)
    cards[3].metric("Bars", f"{result.bar_count:,}", border=True)
    cards[4].metric("Notes", f"{result.written_count:,}", border=True)
    st.caption(_summary(result, analysis, options))
    for message in result.messages:
        st.warning(message)

    name = _download_name(title, result.key if options.transpose_semitones else None)
    musescore = find_musescore_executable()
    with st.container(horizontal=True):
        if musescore is not None:
            st.download_button(
                "PDF",
                data=_pdf_maker(
                    bundle["musicxml"],
                    st.session_state.setdefault("pdfs", {}),
                    repr(options),
                    musescore,
                ),
                file_name=f"{name}.pdf",
                mime="application/pdf",
                type="primary",
                icon=":material/download:",
                on_click="ignore",
                help="Engraved with MuseScore when you click (takes a few seconds).",
            )
        st.download_button(
            "MusicXML",
            data=bundle["musicxml"],
            file_name=f"{name}.musicxml",
            mime="application/vnd.recordare.musicxml+xml",
            icon=":material/download:",
            on_click="ignore",
            help="Open and edit it in MuseScore, Finale, Sibelius or Dorico.",
        )
        st.download_button(
            "MIDI",
            data=bundle["midi"],
            file_name=f"{name}.mid",
            mime="audio/midi",
            icon=":material/download:",
            on_click="ignore",
            help="The written notes, for playback or a piano-learning app.",
        )
    if musescore is None:
        st.caption("To download a PDF, install MuseScore 4 (free) and restart the app.")

    sheet, original = st.columns(2)
    sheet.caption("Sheet music, played back")
    sheet.audio(bundle["listen"], format="audio/wav")
    original.caption("Original recording")
    start, end = part
    original.audio(
        upload["bytes"],
        format=_audio_format(upload["name"]),
        start_time=start or 0,
        end_time=end,
    )

    if bundle["preview_error"]:
        st.warning(bundle["preview_error"])
    _, page_column, _ = st.columns([1, 4, 1])
    for page in bundle["pages"]:
        page_column.image(page, width="stretch")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("\U0001f3b9 " + APP_NAME)
st.write(
    "Turn a piano recording into accurate sheet music. Listen to it, transpose it, "
    "and download it as PDF, MusicXML or MIDI."
)

uploaded = st.file_uploader(
    "Upload a piano recording",
    type=sorted(extension.lstrip(".") for extension in ALLOWED_AUDIO_EXTENSIONS),
    help="Solo piano works best.",
)
upload = None
if uploaded is not None:
    try:
        upload = _store_upload(uploaded.name, uploaded.getvalue())
    except AudioValidationError as exc:
        st.error(str(exc))
if upload is None:
    st.stop()

player = st.container()
start_seconds, end_seconds = _choose_part(upload)
analysis_key = (upload["hash"], start_seconds, end_seconds)
analysis = st.session_state.get("analysis")
stored_key = st.session_state.get("analysis_key")
has_sheet = (
    analysis is not None
    and isinstance(stored_key, tuple)
    and len(stored_key) == len(analysis_key)
    and stored_key[0] == upload["hash"]
)

if not (has_sheet and stored_key == analysis_key):
    player.audio(upload["bytes"], format=_audio_format(upload["name"]))
    if has_sheet:
        st.info("The part to transcribe has changed. Press Transcribe to update the sheet music.")
    if st.button("Transcribe", type="primary", icon=":material/music_note:"):
        _transcribe(upload, analysis_key)

if has_sheet:
    _show_results(upload, analysis, stored_key[1:])
