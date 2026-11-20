"""Split a song into stems (vocals, drums, bass, other) with Demucs."""

from __future__ import annotations

import logging
import threading
from typing import Callable

import numpy as np

LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 44100
FAST_MODEL = "htdemucs"
QUALITY_MODEL = "htdemucs_ft"

_LOCK = threading.Lock()
_SEPARATORS: dict[str, object] = {}


class SeparationError(RuntimeError):
    """Raised when stems cannot be separated."""


def _get_separator(model_name: str):
    with _LOCK:
        if model_name not in _SEPARATORS:
            try:
                from demucs.api import Separator
            except ImportError as exc:  # pragma: no cover - install problem
                raise SeparationError(
                    "Demucs is not installed. Run 'pip install -r requirements.txt'."
                ) from exc
            # shifts=0: no random time offset, so the same file always gives the
            # same result (the random shift improves quality only marginally).
            _SEPARATORS[model_name] = Separator(
                model=model_name, device="cpu", shifts=0, overlap=0.25, progress=False
            )
        return _SEPARATORS[model_name]


def separate_stems(
    audio: np.ndarray,
    *,
    high_quality: bool = False,
    progress: Callable[[float], None] | None = None,
) -> dict[str, np.ndarray]:
    """Separate 44.1 kHz stereo audio (samples, 2) into named stems."""
    import torch

    model_name = QUALITY_MODEL if high_quality else FAST_MODEL
    separator = _get_separator(model_name)

    def report(info: dict) -> None:
        if progress is None or info.get("state") != "end":
            return
        models = max(1, int(info.get("models", 1)))
        length = max(1, int(info.get("audio_length", 1)))
        within = min(1.0, (int(info.get("segment_offset", 0)) + 1) / length)
        overall = (int(info.get("model_idx_in_bag", 0)) + within) / models
        progress(min(1.0, overall))

    separator.update_parameter(callback=report)
    waveform = torch.from_numpy(np.ascontiguousarray(audio.T, dtype=np.float32))
    try:
        with torch.inference_mode():
            _, stems = separator.separate_tensor(waveform, SAMPLE_RATE)
    except Exception as exc:
        LOGGER.exception("Demucs separation failed")
        raise SeparationError("The song could not be split into instruments.") from exc
    return {name: tensor.numpy().T.copy() for name, tensor in stems.items()}
