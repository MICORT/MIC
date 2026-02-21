"""
Tests for wtype_text() and copy_to_clipboard_xdg() integration helpers.
"""

import subprocess
from unittest.mock import patch, MagicMock, call

import pytest

from ptt_gui.core import wtype_text, copy_to_clipboard_xdg


class TestWtypeIntegration:
    @patch("ptt_gui.core.subprocess.run")
    def test_special_characters_passed_through(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        text = "--option value"
        wtype_text(text)
        # The "--" separator must appear before the text to prevent wtype
        # treating the text as flags
        args = mock_run.call_args[0][0]
        assert args[1] == "--"
        assert args[2] == text

    @patch("ptt_gui.core.subprocess.run")
    def test_timeout_kills_hanging_wtype(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="wtype", timeout=5)
        result = wtype_text("test")
        assert result is False

    @patch("ptt_gui.core.subprocess.run")
    def test_calledprocesserror_returns_false(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd="wtype")
        result = wtype_text("test")
        assert result is False

    @patch("ptt_gui.core.subprocess.run")
    def test_multiline_text(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        text = "pierwsza linia\ndruga linia"
        result = wtype_text(text)
        assert result is True


class TestCopyToClipboard:
    @patch("ptt_gui.core.subprocess.run")
    def test_uses_wl_copy_first(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        result = copy_to_clipboard_xdg("test text")
        assert result is True
        # First attempt should be wl-copy
        first_call = mock_run.call_args_list[0]
        assert "wl-copy" in first_call[0][0]

    @patch("ptt_gui.core.subprocess.run")
    def test_falls_back_to_xclip(self, mock_run):
        """When wl-copy fails, should fall back to xclip."""
        def side_effect(cmd, **kwargs):
            if "wl-copy" in cmd:
                raise FileNotFoundError
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect
        result = copy_to_clipboard_xdg("test")
        assert result is True
        calls = [c[0][0] for c in mock_run.call_args_list]
        assert any("xclip" in " ".join(c) for c in calls)

    @patch("ptt_gui.core.subprocess.run", side_effect=FileNotFoundError)
    def test_returns_false_when_nothing_available(self, _mock_run):
        result = copy_to_clipboard_xdg("test")
        assert result is False

    @patch("ptt_gui.core.subprocess.run")
    def test_passes_text_as_stdin_bytes(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        text = "tekst do schowka"
        copy_to_clipboard_xdg(text)
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs.get("input") == text.encode()
