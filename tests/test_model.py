"""
Tests for ptt_gui.core.load_model().

The actual model directory is NOT required — we use mocking.
Tests also verify FileNotFoundError for missing paths.
"""

import os
from unittest.mock import patch, MagicMock

import pytest

from ptt_gui.core import load_model, DEFAULT_MODEL_PATH


class TestLoadModel:
    def test_raises_when_model_path_missing(self, tmp_path):
        """load_model() should raise FileNotFoundError for non-existent dir."""
        missing = str(tmp_path / "nonexistent_model")
        with pytest.raises(FileNotFoundError, match="VOSK model not found"):
            load_model(missing)

    @patch("ptt_gui.core.vosk.SetLogLevel")
    @patch("ptt_gui.core.vosk.Model")
    def test_returns_model_for_valid_path(self, mock_model_cls, mock_set_log, tmp_path):
        """load_model() should call vosk.Model and return its result."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        fake_model = MagicMock()
        mock_model_cls.return_value = fake_model

        result = load_model(str(model_dir))

        mock_model_cls.assert_called_once_with(str(model_dir))
        assert result is fake_model

    @patch("ptt_gui.core.vosk.SetLogLevel")
    @patch("ptt_gui.core.vosk.Model")
    def test_suppresses_vosk_log_output(self, mock_model_cls, mock_set_log, tmp_path):
        """load_model() should call vosk.SetLogLevel(-1) to silence logging."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        mock_model_cls.return_value = MagicMock()

        load_model(str(model_dir))

        mock_set_log.assert_called_once_with(-1)

    def test_default_model_path_is_defined(self):
        """DEFAULT_MODEL_PATH should be a non-empty expanduser'd path."""
        assert DEFAULT_MODEL_PATH
        assert "~" not in DEFAULT_MODEL_PATH   # should already be expanded
        assert "polish" in DEFAULT_MODEL_PATH.lower() or os.path.sep in DEFAULT_MODEL_PATH
