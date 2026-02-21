"""
Tests for ptt_gui.core.transcribe() and related helpers.

All VOSK model calls are mocked so tests run without the actual model files.
"""

import io
import json
import wave
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from ptt_gui.core import transcribe, rms_level, copy_to_clipboard_xdg, wtype_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_audio(duration_s: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    """Return a silent int16 mono array of *duration_s* seconds."""
    samples = int(duration_s * sample_rate)
    return np.zeros(samples, dtype=np.int16)


def _make_sine_audio(freq: float = 440.0, duration_s: float = 0.5, sample_rate: int = 16000) -> np.ndarray:
    """Return a sine wave int16 mono array."""
    t = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False)
    wave_data = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)
    return wave_data


# ---------------------------------------------------------------------------
# transcribe()
# ---------------------------------------------------------------------------

class TestTranscribe:
    """Tests for core.transcribe() using a mocked VOSK model."""

    def _make_recogniser(self, result_text: str) -> MagicMock:
        rec = MagicMock()
        rec.AcceptWaveform.return_value = True
        rec.FinalResult.return_value = json.dumps({"text": result_text})
        return rec

    @patch("ptt_gui.core.vosk.KaldiRecognizer")
    def test_returns_recognised_text(self, mock_recogniser_cls):
        """transcribe() should return the text from VOSK FinalResult."""
        mock_rec = self._make_recogniser("hello world")
        mock_recogniser_cls.return_value = mock_rec

        model = MagicMock()
        audio = _make_audio(1.0)
        result = transcribe(model, audio)

        assert result == "hello world"

    @patch("ptt_gui.core.vosk.KaldiRecognizer")
    def test_returns_empty_string_when_no_speech(self, mock_recogniser_cls):
        """transcribe() should return '' when VOSK returns empty text."""
        mock_rec = self._make_recogniser("")
        mock_recogniser_cls.return_value = mock_rec

        model = MagicMock()
        audio = _make_audio(0.5)
        result = transcribe(model, audio)

        assert result == ""

    @patch("ptt_gui.core.vosk.KaldiRecognizer")
    def test_strips_whitespace(self, mock_recogniser_cls):
        """transcribe() should strip leading/trailing whitespace."""
        mock_rec = self._make_recogniser("  tekst z polskim  ")
        mock_recogniser_cls.return_value = mock_rec

        model = MagicMock()
        audio = _make_audio(1.0)
        result = transcribe(model, audio)

        assert result == "tekst z polskim"

    @patch("ptt_gui.core.vosk.KaldiRecognizer")
    def test_writes_valid_wav_to_recogniser(self, mock_recogniser_cls):
        """transcribe() should call AcceptWaveform at least once with bytes."""
        mock_rec = self._make_recogniser("test")
        mock_recogniser_cls.return_value = mock_rec

        model = MagicMock()
        audio = _make_sine_audio(440, 0.5)
        transcribe(model, audio)

        assert mock_rec.AcceptWaveform.called
        call_args = mock_rec.AcceptWaveform.call_args_list
        for c in call_args:
            data = c[0][0]
            assert isinstance(data, (bytes, bytearray))

    @patch("ptt_gui.core.vosk.KaldiRecognizer")
    def test_handles_missing_text_key(self, mock_recogniser_cls):
        """transcribe() should handle JSON without 'text' key gracefully."""
        mock_rec = MagicMock()
        mock_rec.AcceptWaveform.return_value = True
        mock_rec.FinalResult.return_value = json.dumps({"result": []})
        mock_recogniser_cls.return_value = mock_rec

        model = MagicMock()
        audio = _make_audio(0.5)
        result = transcribe(model, audio)

        assert result == ""

    @patch("ptt_gui.core.vosk.KaldiRecognizer")
    def test_creates_recogniser_with_correct_samplerate(self, mock_recogniser_cls):
        """transcribe() should create KaldiRecognizer with 16000 Hz."""
        mock_rec = self._make_recogniser("")
        mock_recogniser_cls.return_value = mock_rec

        model = MagicMock()
        audio = _make_audio(0.5)
        transcribe(model, audio)

        # First arg to KaldiRecognizer constructor is the model, second is sample rate
        args = mock_recogniser_cls.call_args[0]
        assert args[1] == 16000


# ---------------------------------------------------------------------------
# rms_level()
# ---------------------------------------------------------------------------

class TestRmsLevel:
    def test_silent_signal_returns_zero(self):
        audio = np.zeros(1024, dtype=np.int16)
        assert rms_level(audio) == 0.0

    def test_max_signal_returns_one(self):
        # Full-scale int16 signal → RMS ≈ 0.707 → clipped to ≤ 1.0
        audio = np.full(1024, 32767, dtype=np.int16)
        level = rms_level(audio)
        assert 0.5 <= level <= 1.0

    def test_empty_array_returns_zero(self):
        audio = np.array([], dtype=np.int16)
        assert rms_level(audio) == 0.0

    def test_level_between_zero_and_one(self):
        audio = (np.random.rand(1024) * 16000).astype(np.int16)
        level = rms_level(audio)
        assert 0.0 <= level <= 1.0

    def test_louder_signal_has_higher_rms(self):
        quiet = (np.random.rand(1024) * 1000).astype(np.int16)
        loud = (np.random.rand(1024) * 20000).astype(np.int16)
        assert rms_level(loud) > rms_level(quiet)


# ---------------------------------------------------------------------------
# wtype_text()
# ---------------------------------------------------------------------------

class TestWtypeText:
    @patch("ptt_gui.core.subprocess.run")
    def test_calls_wtype_with_correct_args(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        result = wtype_text("hello")
        mock_run.assert_called_once_with(["wtype", "--", "hello"], check=True, timeout=5)
        assert result is True

    @patch("ptt_gui.core.subprocess.run", side_effect=FileNotFoundError)
    def test_returns_false_when_wtype_not_found(self, _mock_run):
        result = wtype_text("text")
        assert result is False

    @patch("ptt_gui.core.subprocess.run")
    def test_returns_true_on_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        assert wtype_text("dobry tekst") is True

    @patch("ptt_gui.core.subprocess.run")
    def test_handles_empty_string(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        result = wtype_text("")
        assert result is True
        mock_run.assert_called_once_with(["wtype", "--", ""], check=True, timeout=5)

    @patch("ptt_gui.core.subprocess.run")
    def test_handles_unicode_text(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        text = "zażółć gęślą jaźń"
        result = wtype_text(text)
        assert result is True
        mock_run.assert_called_once_with(["wtype", "--", text], check=True, timeout=5)
