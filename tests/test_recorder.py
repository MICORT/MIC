"""
Tests for ptt_gui.core.Recorder.

sounddevice is mocked so no actual audio hardware is required.
"""

import time
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest

from ptt_gui.core import Recorder, SAMPLE_RATE, CHANNELS, BLOCK_SIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(n: int = BLOCK_SIZE) -> np.ndarray:
    """Return a random int16 chunk shaped (n, CHANNELS)."""
    return (np.random.rand(n, CHANNELS) * 16000).astype(np.int16)


def _make_recorder_with_mock_stream(stream_mock=None):
    """Return a Recorder and a mock sd.InputStream."""
    if stream_mock is None:
        stream_mock = MagicMock()
    with patch("ptt_gui.core.sd.InputStream", return_value=stream_mock):
        rec = Recorder()
        return rec, stream_mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRecorder:
    def test_initial_state(self):
        rec = Recorder()
        assert not rec.is_recording
        assert rec.stop().tolist() == []

    @patch("ptt_gui.core.sd.InputStream")
    def test_start_creates_input_stream(self, mock_cls):
        mock_stream = MagicMock()
        mock_cls.return_value = mock_stream

        rec = Recorder()
        rec.start()

        mock_cls.assert_called_once()
        mock_stream.start.assert_called_once()
        assert rec.is_recording

    @patch("ptt_gui.core.sd.InputStream")
    def test_stop_returns_empty_when_no_frames(self, mock_cls):
        mock_cls.return_value = MagicMock()
        rec = Recorder()
        rec.start()
        audio = rec.stop()

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.int16
        assert len(audio) == 0

    @patch("ptt_gui.core.sd.InputStream")
    def test_stop_closes_stream(self, mock_cls):
        mock_stream = MagicMock()
        mock_cls.return_value = mock_stream

        rec = Recorder()
        rec.start()
        rec.stop()

        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        assert not rec.is_recording

    @patch("ptt_gui.core.sd.InputStream")
    def test_callback_accumulates_frames(self, mock_cls):
        mock_cls.return_value = MagicMock()
        rec = Recorder()
        rec.start()

        # Simulate audio callback delivering 3 chunks
        chunks = [_make_chunk() for _ in range(3)]
        for chunk in chunks:
            rec._callback(chunk, BLOCK_SIZE, None, MagicMock())

        audio = rec.stop()
        expected_len = BLOCK_SIZE * 3
        assert len(audio) == expected_len

    @patch("ptt_gui.core.sd.InputStream")
    def test_callback_not_called_when_stopped(self, mock_cls):
        mock_cls.return_value = MagicMock()
        rec = Recorder()
        # Don't call start — _recording is False

        chunk = _make_chunk()
        rec._callback(chunk, BLOCK_SIZE, None, MagicMock())

        audio = rec.stop()
        assert len(audio) == 0

    @patch("ptt_gui.core.sd.InputStream")
    def test_on_chunk_callback_is_called(self, mock_cls):
        mock_cls.return_value = MagicMock()
        received = []
        rec = Recorder(on_chunk=lambda c: received.append(c.copy()))
        rec.start()

        chunk = _make_chunk()
        rec._callback(chunk, BLOCK_SIZE, None, MagicMock())
        rec.stop()

        assert len(received) == 1
        assert received[0].dtype == np.int16

    @patch("ptt_gui.core.sd.InputStream")
    def test_start_twice_resets_frames(self, mock_cls):
        mock_cls.return_value = MagicMock()
        rec = Recorder()
        rec.start()

        chunk = _make_chunk(512)
        rec._callback(chunk, 512, None, MagicMock())

        rec.start()   # second start should clear frames
        audio = rec.stop()
        assert len(audio) == 0

    @patch("ptt_gui.core.sd.InputStream")
    def test_stop_is_safe_when_not_started(self, mock_cls):
        mock_cls.return_value = MagicMock()
        rec = Recorder()
        audio = rec.stop()    # should not raise
        assert isinstance(audio, np.ndarray)

    @patch("ptt_gui.core.sd.InputStream")
    def test_audio_concatenated_correctly(self, mock_cls):
        mock_cls.return_value = MagicMock()
        rec = Recorder()
        rec.start()

        # Feed known values
        chunk1 = np.ones((4, 1), dtype=np.int16) * 100
        chunk2 = np.ones((4, 1), dtype=np.int16) * 200
        rec._callback(chunk1, 4, None, MagicMock())
        rec._callback(chunk2, 4, None, MagicMock())

        audio = rec.stop()
        assert len(audio) == 8
        assert list(audio[:4]) == [100] * 4
        assert list(audio[4:]) == [200] * 4
