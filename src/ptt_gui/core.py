"""
ptt_gui.core — Shared audio recording and transcription logic.

Used by both the CLI (ptt.py) and the GTK4 GUI (ptt_app.py).
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import wave
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import vosk

SAMPLE_RATE: int = 16000
CHANNELS: int = 1
BLOCK_SIZE: int = 1024
DEFAULT_MODEL_PATH: str = os.path.expanduser("~/stt-models/polish")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(path: str = DEFAULT_MODEL_PATH) -> vosk.Model:
    """Load a VOSK model from *path*.  Raises FileNotFoundError if not found."""
    vosk.SetLogLevel(-1)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"VOSK model not found: {path}")
    return vosk.Model(path)


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe(model: vosk.Model, audio_data: np.ndarray) -> str:
    """Transcribe *audio_data* (int16, mono, 16 kHz) using *model*.

    Returns the recognised text, or an empty string if nothing was heard.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())
    buf.seek(0)

    wf = wave.open(buf, "rb")
    rec = vosk.KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    while True:
        data = wf.readframes(4000)
        if not data:
            break
        rec.AcceptWaveform(data)
    result = json.loads(rec.FinalResult())
    return result.get("text", "").strip()


# ---------------------------------------------------------------------------
# wtype integration
# ---------------------------------------------------------------------------

def wtype_text(text: str) -> bool:
    """Type *text* into the previously active window via wtype.

    Returns True on success, False on failure.
    """
    try:
        subprocess.run(["wtype", "--", text], check=True, timeout=5)
        return True
    except FileNotFoundError:
        return False
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self, device: Optional[int] = None, samplerate: int = SAMPLE_RATE) -> None:
        """Start recording.  Safe to call multiple times (stops previous)."""
        self.stop()
        self._frames = []
        self._recording = True
        self._stream = sd.InputStream(
            samplerate=samplerate,
            channels=CHANNELS,
            dtype="int16",
            blocksize=BLOCK_SIZE,
            device=device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        """Stop recording and return the captured audio as int16 mono array."""
        self._recording = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            finally:
                self._stream = None
        if self._frames:
            return np.concatenate(self._frames, axis=0).flatten()
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
