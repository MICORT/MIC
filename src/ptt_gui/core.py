"""
ptt_gui.core — Shared audio recording and transcription logic.

Used by both the CLI (ptt.py) and the GTK4 GUI (ptt_app.py).

Speech recognition uses faster-whisper (CTranslate2 backend, no PyTorch).
Model: whisper base, language: Polish (pl).
"""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
import threading
import wave
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE: int = 16000
CHANNELS: int = 1
BLOCK_SIZE: int = 1024

# Default whisper model size — "base" is a good speed/quality balance for CPU
DEFAULT_WHISPER_MODEL: str = "base"
# Language hint for Whisper — speeds up recognition and improves accuracy
DEFAULT_LANGUAGE: str = "pl"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_size: str = DEFAULT_WHISPER_MODEL,
    language: str = DEFAULT_LANGUAGE,
) -> WhisperModel:
    """Load a faster-whisper model.

    Downloads the model from HuggingFace Hub on first run (~145 MB for base),
    then caches it at ~/.cache/huggingface/hub/.

    Parameters
    ----------
    model_size:
        Whisper model size: "tiny", "base", "small", "medium", "large-v3", …
    language:
        Language hint stored on the model object for use in transcribe().

    Returns
    -------
    WhisperModel
        A loaded faster-whisper model instance with a `language` attribute set.
    """
    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",  # fastest CPU inference
    )
    # Attach language preference so callers don't need to pass it separately
    model.language = language  # type: ignore[attr-defined]
    return model


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe(model: WhisperModel, audio_data: np.ndarray) -> str:
    """Transcribe *audio_data* (int16, mono, 16 kHz) using Whisper.

    Writes audio to a temporary WAV file, then feeds it to faster-whisper.
    Returns the recognised text, or an empty string if nothing was heard.

    Parameters
    ----------
    model:
        A WhisperModel loaded via load_model().  Should have a .language attr.
    audio_data:
        Flat int16 numpy array, mono, sampled at SAMPLE_RATE.

    Returns
    -------
    str
        Transcribed text, stripped of leading/trailing whitespace.
    """
    if len(audio_data) == 0:
        return ""

    language = getattr(model, "language", DEFAULT_LANGUAGE)

    # Write audio to a temp WAV file — faster-whisper reads from file path
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)          # int16 = 2 bytes per sample
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())

    try:
        segments, _info = model.transcribe(
            tmp_path,
            language=language,
            beam_size=1,            # fastest decoding
            vad_filter=True,        # skip silent parts automatically
            vad_parameters={
                "min_silence_duration_ms": 300,
                "speech_pad_ms": 200,
            },
        )
        text = " ".join(seg.text for seg in segments).strip()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return text


# ---------------------------------------------------------------------------
# wtype integration
# ---------------------------------------------------------------------------

def wtype_text(text: str) -> bool:
    """Type *text* into the previously active window.

    Strategy: copy to X11 clipboard via xclip, then simulate Ctrl+V via xdotool.
    Falls back to wtype if xdotool is unavailable.
    Returns True on success, False on failure.
    """
    # Method 1: xclip + xdotool (works on GNOME Wayland with XWayland)
    try:
        subprocess.run(
            ["xclip", "-selection", "clipboard"],
            input=text.encode("utf-8"), check=True, timeout=3,
        )
        subprocess.run(
            ["xdotool", "key", "--clearmodifiers", "ctrl+v"],
            check=True, timeout=3,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass

    # Method 2: wtype (native Wayland — may not work on all compositors)
    try:
        subprocess.run(["wtype", "--", text], check=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass

    return False


def copy_to_clipboard_xdg(text: str) -> bool:
    """Copy *text* to clipboard via wl-copy (Wayland) or xclip (X11)."""
    for cmd in (["wl-copy"], ["xclip", "-selection", "clipboard"]):
        try:
            subprocess.run(cmd, input=text.encode(), check=True, timeout=3)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
    return False


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class Recorder:
    """Thread-safe audio recorder.

    Usage::

        recorder = Recorder(on_chunk=callback)
        recorder.start()
        # ... wait ...
        audio = recorder.stop()   # np.ndarray int16
    """

    def __init__(self, on_chunk: Optional[Callable[[np.ndarray], None]] = None):
        """
        Parameters
        ----------
        on_chunk:
            Optional callback invoked with each raw audio chunk (int16 array).
            Called from the PortAudio thread — keep it lightweight.
        """
        self._frames: list[np.ndarray] = []
        self._recording: bool = False
        self._stream: Optional[sd.InputStream] = None
        self._on_chunk = on_chunk
        self._lock = threading.Lock()
        # Event set once the stream is fully opened — stop() waits on this
        self._stream_ready = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self, device: Optional[int] = None, samplerate: int = SAMPLE_RATE) -> None:
        """Start recording.  Safe to call multiple times (stops previous).

        May be called from a background thread.  Sets _stream_ready when the
        PortAudio stream is open so that stop() can safely close it.
        """
        self.stop()
        with self._lock:
            self._frames = []
            self._recording = True
            self._stream_ready.clear()
            self._stream = sd.InputStream(
                samplerate=samplerate,
                channels=CHANNELS,
                dtype="int16",
                blocksize=BLOCK_SIZE,
                device=device,
                callback=self._callback,
            )
            self._stream.start()
            self._stream_ready.set()

    def stop(self) -> np.ndarray:
        """Stop recording and return the captured audio as int16 mono array.

        Waits (briefly) for the stream to be ready before closing, so it is
        safe to call immediately after start() even from another thread.
        """
        self._recording = False
        # Wait at most 2 s for the stream to be opened before attempting close
        self._stream_ready.wait(timeout=2.0)
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                finally:
                    self._stream = None
            frames = list(self._frames)
        if frames:
            return np.concatenate(frames, axis=0).flatten()
        return np.array([], dtype=np.int16)

    # ------------------------------------------------------------------
    # PortAudio callback (runs in audio thread)
    # ------------------------------------------------------------------

    def _callback(
        self,
        indata: np.ndarray,
        frame_count: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if self._recording:
            chunk = indata.copy()
            self._frames.append(chunk)
            if self._on_chunk is not None:
                try:
                    self._on_chunk(chunk.flatten())
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Level meter helper
# ---------------------------------------------------------------------------

def rms_level(chunk: np.ndarray) -> float:
    """Return RMS amplitude in [0, 1] for an int16 audio chunk."""
    if len(chunk) == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
    return min(rms / 32768.0, 1.0)
