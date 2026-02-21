"""
ptt_gui.app — GTK4/Adwaita Push-to-Talk Speech-to-Text desktop application.

Architecture
------------
- GLib.idle_add / GLib.timeout_add are used to ferry results from worker
  threads (audio recording, VOSK transcription) back to the GTK main loop.
- Recorder runs on a background thread via threading.Thread.
- Transcription runs on a GLib.idle callback after recording stops so the
  UI updates immediately (spinner, status label) before the heavy work.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime
from typing import Optional

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import Adw, GLib, Gtk, Gdk, GdkPixbuf, Gio  # noqa: E402

import numpy as np  # noqa: E402

# Local core (same package)
from ptt_gui.core import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    Recorder,
    copy_to_clipboard_xdg,
    load_model,
    rms_level,
    transcribe,
    wtype_text,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_ID = "pl.tomw.PushToTalk"
APP_NAME = "Push-to-Talk STT"
VERSION = "1.0.0"

WAVEFORM_BARS = 40          # number of bars in the visualiser
WAVEFORM_UPDATE_MS = 50     # redraw interval while recording
HISTORY_MAX = 200           # max entries in history list

DARK_CSS = """
window, .main-window {
    background-color: #1e1e2e;
}

.mic-button {
    border-radius: 80px;
    min-width: 160px;
    min-height: 160px;
    font-size: 16px;
    font-weight: bold;
    background: linear-gradient(135deg, #313244, #45475a);
    color: #cdd6f4;
    border: 3px solid #45475a;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    transition: all 200ms ease;
}

.mic-button:hover {
    background: linear-gradient(135deg, #45475a, #585b70);
    border-color: #89b4fa;
    box-shadow: 0 8px 40px rgba(137,180,250,0.25);
}

.mic-button.recording {
    background: linear-gradient(135deg, #f38ba8, #e64553);
    border-color: #f38ba8;
    box-shadow: 0 0 0 0 rgba(243,139,168,0.6);
}

.mic-button.transcribing {
    background: linear-gradient(135deg, #fab387, #e5a066);
    border-color: #fab387;
}

.status-label {
    font-size: 13px;
    color: #a6adc8;
    font-weight: 500;
    letter-spacing: 1px;
}

.status-label.recording {
    color: #f38ba8;
    font-weight: 700;
}

.status-label.transcribing {
    color: #fab387;
}

.status-label.ready {
    color: #a6e3a1;
}

.history-row {
    background-color: #313244;
    border-radius: 8px;
    margin: 2px 4px;
    padding: 8px 12px;
}

.history-row:hover {
    background-color: #45475a;
}

.history-text {
    color: #cdd6f4;
    font-size: 14px;
}

.history-time {
    color: #6c7086;
    font-size: 11px;
}

.copy-btn {
    border-radius: 6px;
    padding: 2px 8px;
    color: #89b4fa;
    border: 1px solid #89b4fa;
    font-size: 11px;
    background: transparent;
}

.copy-btn:hover {
    background: rgba(137,180,250,0.15);
}

.type-btn {
    border-radius: 6px;
    padding: 2px 8px;
    color: #a6e3a1;
    border: 1px solid #a6e3a1;
    font-size: 11px;
    background: transparent;
}

.type-btn:hover {
    background: rgba(166,227,161,0.15);
}

.sidebar-title {
    color: #cdd6f4;
    font-weight: 700;
    font-size: 15px;
}

.clear-btn {
    color: #f38ba8;
}

.hint-label {
    color: #585b70;
    font-size: 12px;
    font-style: italic;
}

.settings-group {
    background-color: #181825;
    border-radius: 12px;
}

headerbar {
    background-color: #181825;
    border-bottom: 1px solid #313244;
    color: #cdd6f4;
}

.waveform-container {
    background-color: #181825;
    border-radius: 12px;
    border: 1px solid #313244;
    padding: 8px;
}

.waveform-box {
    min-height: 80px;
}

.wf-bar {
    background-color: #89b4fa;
    border-radius: 2px;
    min-width: 4px;
    transition: all 80ms ease;
}

.wf-bar-idle {
    background-color: #45475a;
}

.wf-bar-high {
    background-color: #a6e3a1;
}
"""


# ---------------------------------------------------------------------------
# Waveform drawing area
# ---------------------------------------------------------------------------

class WaveformWidget(Gtk.Box):
    """CSS-based bar-graph waveform (no cairo dependency).
    Feed RMS values via push_level()."""

    def __init__(self) -> None:
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=2)
        self.add_css_class("waveform-box")
        self.set_halign(Gtk.Align.CENTER)
        self.set_valign(Gtk.Align.CENTER)
        self.set_size_request(-1, 80)

        self._levels: list[float] = [0.0] * WAVEFORM_BARS
        self._bars: list[Gtk.Box] = []
        self._active: bool = False

        for _ in range(WAVEFORM_BARS):
            bar = Gtk.Box()
            bar.add_css_class("wf-bar")
            bar.set_size_request(4, 3)
            bar.set_valign(Gtk.Align.CENTER)
            self.append(bar)
            self._bars.append(bar)

    def push_level(self, rms: float) -> None:
        self._levels.append(min(rms * 3.5, 1.0))
        if len(self._levels) > WAVEFORM_BARS:
            self._levels.pop(0)
        GLib.idle_add(self._update_bars)

    def set_active(self, active: bool) -> None:
        self._active = active
        if not active:
            self._levels = [0.0] * WAVEFORM_BARS
        GLib.idle_add(self._update_bars)

    def _update_bars(self) -> None:
        if self._active:
            for i, bar in enumerate(self._bars):
                lvl = self._levels[i] if i < len(self._levels) else 0.0
                h = max(3, int(lvl * 72))
                bar.set_size_request(4, h)
                bar.remove_css_class("wf-bar-idle")
                if lvl > 0.6:
                    bar.add_css_class("wf-bar-high")
                else:
                    bar.remove_css_class("wf-bar-high")
        else:
            # Static idle state — flat bars, no animation
            for bar in self._bars:
                bar.set_size_request(4, 3)
                bar.add_css_class("wf-bar-idle")
                bar.remove_css_class("wf-bar-high")
        return GLib.SOURCE_REMOVE


# ---------------------------------------------------------------------------
# History row widget
# ---------------------------------------------------------------------------

class HistoryRow(Gtk.Box):
    def __init__(self, text: str, timestamp: str, on_copy, on_type) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.add_css_class("history-row")
        self.set_margin_top(2)
        self.set_margin_bottom(2)
        self.set_margin_start(4)
        self.set_margin_end(4)

        # Top row: text + buttons
        top = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

        label = Gtk.Label(label=text, xalign=0.0, wrap=True, wrap_mode=2)
        label.add_css_class("history-text")
        label.set_hexpand(True)
        label.set_selectable(True)
        top.append(label)

        btn_copy = Gtk.Button(label="Copy")
        btn_copy.add_css_class("copy-btn")
        btn_copy.connect("clicked", lambda _: on_copy(text))
        top.append(btn_copy)

        btn_type = Gtk.Button(label="Type")
        btn_type.add_css_class("type-btn")
        btn_type.connect("clicked", lambda _: on_type(text))
        top.append(btn_type)

        self.append(top)

        # Bottom: timestamp
        time_label = Gtk.Label(label=timestamp, xalign=0.0)
        time_label.add_css_class("history-time")
        self.append(time_label)


# ---------------------------------------------------------------------------
# Settings dialog
# ---------------------------------------------------------------------------

class SettingsDialog(Adw.Dialog):
    def __init__(self, parent: "PTTWindow") -> None:
        super().__init__(title="Ustawienia")
        self._parent = parent
        self.set_content_width(420)
        self.set_content_height(480)

        toolbar_view = Adw.ToolbarView()
        header = Adw.HeaderBar()
        toolbar_view.add_top_bar(header)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        content.set_margin_top(16)
        content.set_margin_bottom(16)
        content.set_margin_start(16)
        content.set_margin_end(16)

        # --- Model path ---
        model_group = Adw.PreferencesGroup(title="Model VOSK")
        model_row = Adw.EntryRow(title="Ścieżka do modelu")
        model_row.set_text(parent.model_path)
        model_row.connect("changed", lambda r: setattr(parent, "_pending_model_path", r.get_text()))
        model_group.add(model_row)
        content.append(model_group)

        # --- Trigger key ---
        key_group = Adw.PreferencesGroup(title="Skrót klawiszowy")
        key_hint = Gtk.Label(label="Aktywacja: przytrzymaj i puść przycisk mikrofonu myszą.\nSkrót globalny: Space (w oknie aplikacji).")
        key_hint.set_wrap(True)
        key_hint.add_css_class("hint-label")
        key_group.add(key_hint)
        content.append(key_group)

        # --- Output mode ---
        mode_group = Adw.PreferencesGroup(title="Tryb wyjścia")

        self._mode_row = Adw.ComboRow(title="Co zrobić z tekstem")
        mode_model = Gtk.StringList.new(["Tylko historia (clipboard)", "Wpisz w aktywne okno (wtype)"])
        self._mode_row.set_model(mode_model)
        self._mode_row.set_selected(1 if parent.auto_type else 0)  # reflects current state
        self._mode_row.connect("notify::selected", self._on_mode_changed)
        mode_group.add(self._mode_row)
        content.append(mode_group)

        # --- Microphone sensitivity ---
        sens_group = Adw.PreferencesGroup(title="Mikrofon")
        sens_row = Adw.SpinRow.new_with_range(0.5, 5.0, 0.1)
        sens_row.set_title("Wzmocnienie sygnału (gain)")
        sens_row.set_value(parent.gain)
        sens_row.connect("changed", lambda r: setattr(parent, "gain", r.get_value()))
        sens_group.add(sens_row)
        content.append(sens_group)

        # --- Min duration ---
        dur_group = Adw.PreferencesGroup(title="Transkrypcja")
        dur_row = Adw.SpinRow.new_with_range(0.1, 2.0, 0.1)
        dur_row.set_title("Min. czas nagrania (s)")
        dur_row.set_value(parent.min_duration)
        dur_row.connect("changed", lambda r: setattr(parent, "min_duration", r.get_value()))
        dur_group.add(dur_row)
        content.append(dur_group)

        # Apply button
        apply_btn = Gtk.Button(label="Zastosuj i uruchom ponownie model")
        apply_btn.add_css_class("suggested-action")
        apply_btn.connect("clicked", self._on_apply)
        content.append(apply_btn)

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_child(content)
        scrolled.set_vexpand(True)
        toolbar_view.set_content(scrolled)
        self.set_child(toolbar_view)

    def _on_mode_changed(self, row, _param) -> None:
        self._parent.auto_type = (row.get_selected() == 1)

    def _on_apply(self, _btn) -> None:
        pending = getattr(self._parent, "_pending_model_path", None)
        if pending:
            self._parent.model_path = pending
        self._parent.reload_model()
        self.close()


# ---------------------------------------------------------------------------
# About dialog
# ---------------------------------------------------------------------------

def show_about(parent) -> None:
    dialog = Adw.AboutDialog(
        application_name=APP_NAME,
        application_icon="audio-input-microphone",
        version=VERSION,
        developer_name="PTT STT Project",
        comments="Push-to-Talk mowa na tekst (Polski / VOSK)",
        license_type=Gtk.License.MIT_X11,
    )
    dialog.present(parent)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class PTTWindow(Adw.ApplicationWindow):
    def __init__(self, app: Adw.Application) -> None:
        super().__init__(application=app)
        self.set_title(APP_NAME)
        self.set_default_size(780, 620)
        self.set_resizable(True)

        # --- State ---
        self.model_path: str = DEFAULT_MODEL_PATH
        self.auto_type: bool = True
        self.gain: float = 1.0
        self.min_duration: float = 0.3
        self._model: Optional[object] = None
        self._model_loading: bool = False
        self._recorder = Recorder(on_chunk=self._on_audio_chunk)
        self._recording: bool = False
        self._history_items: list[str] = []

        # --- CSS ---
        css_provider = Gtk.CssProvider()
        css_provider.load_from_string(DARK_CSS)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

        # --- Build UI ---
        self._build_ui()

        # --- Load model in background ---
        self._load_model_async()

        # No idle animation — waveform only updates when recording

        # --- Keyboard shortcut: Space = push-to-talk ---
        key_ctrl = Gtk.EventControllerKey()
        key_ctrl.connect("key-pressed", self._on_key_pressed)
        key_ctrl.connect("key-released", self._on_key_released)
        self.add_controller(key_ctrl)

    # ======================================================================
    # UI construction
    # ======================================================================

    def _build_ui(self) -> None:
        toolbar_view = Adw.ToolbarView()

        # --- Header bar ---
        header = Adw.HeaderBar()
        header.set_centering_policy(Adw.CenteringPolicy.STRICT)

        title_widget = Adw.WindowTitle(title=APP_NAME, subtitle="Naciśnij i przytrzymaj mikrofon")
        header.set_title_widget(title_widget)

        # Settings button
        settings_btn = Gtk.Button(icon_name="preferences-system-symbolic")
        settings_btn.set_tooltip_text("Ustawienia")
        settings_btn.connect("clicked", self._on_settings)
        header.pack_end(settings_btn)

        # About button
        about_btn = Gtk.Button(icon_name="help-about-symbolic")
        about_btn.set_tooltip_text("O aplikacji")
        about_btn.connect("clicked", lambda _: show_about(self))
        header.pack_end(about_btn)

        toolbar_view.add_top_bar(header)

        # --- Main layout: left panel + right history ---
        main_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)

        # LEFT PANEL
        left = self._build_left_panel()
        main_box.append(left)

        # Separator
        sep = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        main_box.append(sep)

        # RIGHT PANEL (history)
        right = self._build_right_panel()
        main_box.append(right)

        toolbar_view.set_content(main_box)
        self.set_content(toolbar_view)

    def _build_left_panel(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=24)
        box.set_margin_top(32)
        box.set_margin_bottom(24)
        box.set_margin_start(32)
        box.set_margin_end(32)
        box.set_hexpand(True)

        # --- Status label ---
        self._status_label = Gtk.Label(label="ŁADOWANIE MODELU...")
        self._status_label.add_css_class("status-label")
        box.append(self._status_label)

        # --- Mic button ---
        self._mic_btn = Gtk.Button()
        self._mic_btn.add_css_class("mic-button")
        self._mic_btn.set_sensitive(False)   # disabled until model loads

        mic_inner = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        mic_inner.set_halign(Gtk.Align.CENTER)
        mic_inner.set_valign(Gtk.Align.CENTER)

        # Mic icon — use a large symbolic icon
        mic_icon = Gtk.Image.new_from_icon_name("audio-input-microphone-symbolic")
        mic_icon.set_pixel_size(64)
        mic_inner.append(mic_icon)

        self._mic_label = Gtk.Label(label="Mów")
        mic_inner.append(self._mic_label)

        self._mic_btn.set_child(mic_inner)
        self._mic_btn.set_halign(Gtk.Align.CENTER)

        # Press/release gesture for hold-to-record
        gesture = Gtk.GestureClick()
        gesture.set_button(0)   # all buttons
        gesture.connect("pressed", self._on_mic_press)
        gesture.connect("released", self._on_mic_release)
        self._mic_btn.add_controller(gesture)

        box.append(self._mic_btn)

        # --- Spinner (transcribing) ---
        self._spinner = Gtk.Spinner()
        self._spinner.set_size_request(32, 32)
        self._spinner.set_visible(False)
        box.append(self._spinner)

        # --- Waveform ---
        wf_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        wf_container.add_css_class("waveform-container")
        wf_container.set_margin_top(4)

        self._waveform = WaveformWidget()
        wf_container.append(self._waveform)
        box.append(wf_container)

        # --- Mode toggle ---
        mode_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        mode_box.set_halign(Gtk.Align.CENTER)

        mode_label = Gtk.Label(label="Auto-type:")
        mode_label.add_css_class("hint-label")
        mode_box.append(mode_label)

        self._auto_type_switch = Gtk.Switch()
        self._auto_type_switch.set_active(True)
        self._auto_type_switch.connect("state-set", self._on_auto_type_toggle)
        mode_box.append(self._auto_type_switch)

        mode_hint = Gtk.Label(label="(wtype w aktywne okno)")
        mode_hint.add_css_class("hint-label")
        mode_box.append(mode_hint)

        box.append(mode_box)

        # --- Hint ---
        hint = Gtk.Label(label="Przytrzymaj mikrofon lub [Space] aby nagrać")
        hint.add_css_class("hint-label")
        box.append(hint)

        return box

    def _build_right_panel(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        box.set_size_request(320, -1)
        box.set_margin_top(16)
        box.set_margin_bottom(16)
        box.set_margin_start(12)
        box.set_margin_end(12)

        # Header row
        header_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        header_box.set_margin_bottom(8)

        history_title = Gtk.Label(label="Historia transkrypcji")
        history_title.add_css_class("sidebar-title")
        history_title.set_hexpand(True)
        history_title.set_xalign(0.0)
        header_box.append(history_title)

        clear_btn = Gtk.Button(label="Wyczyść")
        clear_btn.add_css_class("clear-btn")
        clear_btn.add_css_class("flat")
        clear_btn.connect("clicked", self._on_clear_history)
        header_box.append(clear_btn)

        box.append(header_box)

        # Scrollable list
        scroll = Gtk.ScrolledWindow()
        scroll.set_vexpand(True)
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        self._history_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self._history_box.set_margin_top(4)
        scroll.set_child(self._history_box)
        box.append(scroll)

        # Empty state label
        self._empty_label = Gtk.Label(label="Brak transkrypcji.\nNaciśnij mikrofon i mów po polsku.")
        self._empty_label.set_wrap(True)
        self._empty_label.set_justify(Gtk.Justification.CENTER)
        self._empty_label.add_css_class("hint-label")
        self._empty_label.set_vexpand(True)
        self._empty_label.set_valign(Gtk.Align.CENTER)
        self._history_box.append(self._empty_label)

        return box

    # ======================================================================
    # Model loading
    # ======================================================================

    def _load_model_async(self) -> None:
        self._model_loading = True
        self._set_status("ŁADOWANIE MODELU...", "")
        thread = threading.Thread(target=self._load_model_worker, daemon=True)
        thread.start()

    def _load_model_worker(self) -> None:
        try:
            model = load_model(self.model_path)
            GLib.idle_add(self._on_model_loaded, model)
        except Exception as exc:
            GLib.idle_add(self._on_model_error, str(exc))

    def _on_model_loaded(self, model) -> None:
        self._model = model
        self._model_loading = False
        self._mic_btn.set_sensitive(True)
        self._set_status("GOTOWY", "ready")
        return GLib.SOURCE_REMOVE

    def _on_model_error(self, msg: str) -> None:
        self._model_loading = False
        self._set_status(f"BŁĄD: {msg}", "")
        self._show_toast(f"Nie można załadować modelu: {msg}")
        return GLib.SOURCE_REMOVE

    def reload_model(self) -> None:
        self._model = None
        self._mic_btn.set_sensitive(False)
        self._load_model_async()

    # ======================================================================
    # Recording logic
    # ======================================================================

    def _start_recording(self) -> None:
        if self._recording or self._model is None:
            return
        self._recording = True
        self._waveform.set_active(True)
        self._mic_btn.add_css_class("recording")
        self._set_status("NAGRYWANIE...", "recording")
        self._mic_label.set_label("Puść")
        # Open the PortAudio stream in a thread to avoid blocking the GTK loop
        threading.Thread(target=self._recorder.start, daemon=True).start()

    def _stop_recording(self) -> None:
        if not self._recording:
            return
        self._recording = False
        audio = self._recorder.stop()
        self._waveform.set_active(False)
        self._mic_btn.remove_css_class("recording")
        self._set_status("PRZETWARZANIE...", "transcribing")
        self._mic_btn.add_css_class("transcribing")
        self._mic_label.set_label("Mów")
        self._spinner.set_visible(True)
        self._spinner.start()

        duration = len(audio) / 16000
        if duration < self.min_duration:
            self._reset_to_ready()
            self._show_toast(f"Za krótkie nagranie ({duration:.1f}s)")
            return

        # Transcribe in background thread
        model_ref = self._model
        thread = threading.Thread(
            target=self._transcribe_worker, args=(model_ref, audio), daemon=True
        )
        thread.start()

    def _transcribe_worker(self, model, audio: "np.ndarray") -> None:
        try:
            text = transcribe(model, audio)
            GLib.idle_add(self._on_transcription_done, text)
        except Exception as exc:
            GLib.idle_add(self._on_transcription_error, str(exc))

    def _on_transcription_done(self, text: str) -> None:
        self._reset_to_ready()
        if text:
            self._add_to_history(text)
            # Always copy to GTK clipboard so the result is always accessible
            clipboard = Gdk.Display.get_default().get_clipboard()
            clipboard.set(text)
            if self.auto_type:
                # Short delay so the PTT window can lose focus before wtype fires
                threading.Thread(
                    target=self._wtype_with_delay, args=(text,), daemon=True
                ).start()
            else:
                self._show_toast("Skopiowano do schowka")
        else:
            self._show_toast("Nie rozpoznano mowy")
        return GLib.SOURCE_REMOVE

    def _wtype_with_delay(self, text: str) -> None:
        """Wait briefly so the PTT window loses focus, then type via wtype."""
        time.sleep(0.25)
        ok = wtype_text(text)
        if not ok:
            GLib.idle_add(self._show_toast, "Błąd wtype — skopiowano do schowka")

    def _on_transcription_error(self, msg: str) -> None:
        self._reset_to_ready()
        self._show_toast(f"Błąd transkrypcji: {msg}")
        return GLib.SOURCE_REMOVE

    def _reset_to_ready(self) -> None:
        self._mic_btn.remove_css_class("recording")
        self._mic_btn.remove_css_class("transcribing")
        self._spinner.stop()
        self._spinner.set_visible(False)
        self._set_status("GOTOWY", "ready")
        self._mic_label.set_label("Mów")

    # ======================================================================
    # Audio chunk callback (audio thread)
    # ======================================================================

    def _on_audio_chunk(self, chunk: "np.ndarray") -> None:
        level = rms_level(chunk) * self.gain
        self._waveform.push_level(level)

    # ======================================================================
    # History management
    # ======================================================================

    def _add_to_history(self, text: str) -> None:
        # Remove empty state label
        if self._empty_label.get_parent() is not None:
            self._history_box.remove(self._empty_label)

        ts = datetime.now().strftime("%H:%M:%S")
        row = HistoryRow(text, ts, self._copy_text, self._type_text)
        self._history_box.prepend(row)
        self._history_items.insert(0, text)

        # Trim excess
        children = []
        child = self._history_box.get_first_child()
        while child:
            children.append(child)
            child = child.get_next_sibling()

        while len(children) > HISTORY_MAX:
            self._history_box.remove(children[-1])
            children.pop()

    def _on_clear_history(self, _btn) -> None:
        self._history_items.clear()
        child = self._history_box.get_first_child()
        while child:
            nxt = child.get_next_sibling()
            self._history_box.remove(child)
            child = nxt
        self._history_box.append(self._empty_label)

    def _copy_text(self, text: str) -> None:
        # GTK clipboard (Wayland-native)
        clipboard = Gdk.Display.get_default().get_clipboard()
        clipboard.set(text)
        self._show_toast("Skopiowano do schowka")

    def _type_text(self, text: str) -> None:
        threading.Thread(target=self._type_worker, args=(text,), daemon=True).start()

    def _type_worker(self, text: str) -> None:
        ok = wtype_text(text)
        if not ok:
            GLib.idle_add(self._show_toast, "Błąd wtype — czy wtype jest zainstalowany?")
            return GLib.SOURCE_REMOVE

    # ======================================================================
    # Status helpers
    # ======================================================================

    def _set_status(self, label: str, style: str) -> None:
        self._status_label.set_label(label)
        for cls in ("ready", "recording", "transcribing"):
            self._status_label.remove_css_class(cls)
        if style:
            self._status_label.add_css_class(style)

    def _show_toast(self, message: str) -> None:
        """Show an Adwaita toast notification."""
        toast = Adw.Toast(title=message)
        toast.set_timeout(3)
        # Find the ToastOverlay — we wrap the content in one
        overlay = self._get_toast_overlay()
        if overlay:
            overlay.add_toast(toast)
        return GLib.SOURCE_REMOVE

    def _get_toast_overlay(self) -> Optional[Adw.ToastOverlay]:
        return getattr(self, "_toast_overlay", None)

    # ======================================================================
    # Event handlers
    # ======================================================================

    def _on_mic_press(self, gesture, n_press, x, y) -> None:
        self._start_recording()

    def _on_mic_release(self, gesture, n_press, x, y) -> None:
        self._stop_recording()

    def _on_key_pressed(self, ctrl, keyval, keycode, state) -> bool:
        if keyval == Gdk.KEY_space and not self._recording:
            self._start_recording()
            return True
        return False

    def _on_key_released(self, ctrl, keyval, keycode, state) -> None:
        if keyval == Gdk.KEY_space and self._recording:
            self._stop_recording()

    def _on_auto_type_toggle(self, switch, state) -> bool:
        self.auto_type = state
        return False

    def _on_settings(self, _btn) -> None:
        dialog = SettingsDialog(self)
        dialog.present(self)



# ---------------------------------------------------------------------------
# Wrap window in ToastOverlay (must be done before show)
# ---------------------------------------------------------------------------

class PTTWindowWithToast(PTTWindow):
    def __init__(self, app: Adw.Application) -> None:
        super().__init__(app)
        # Re-wrap the content in a ToastOverlay
        content = self.get_content()
        self.set_content(None)
        overlay = Adw.ToastOverlay()
        overlay.set_child(content)
        self.set_content(overlay)
        self._toast_overlay = overlay


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

class PTTApplication(Adw.Application):
    def __init__(self) -> None:
        super().__init__(
            application_id=APP_ID,
            flags=Gio.ApplicationFlags.DEFAULT_FLAGS,
        )
        self._window: Optional[PTTWindowWithToast] = None

    def do_activate(self) -> None:
        if self._window is None:
            self._window = PTTWindowWithToast(self)
            # Force dark colour scheme
            Adw.StyleManager.get_default().set_color_scheme(Adw.ColorScheme.FORCE_DARK)
        self._window.present()

    def do_startup(self) -> None:
        Adw.Application.do_startup(self)

        # Quit action
        quit_action = Gio.SimpleAction.new("quit", None)
        quit_action.connect("activate", lambda *_: self.quit())
        self.add_action(quit_action)
        self.set_accels_for_action("app.quit", ["<Ctrl>q"])


def main(argv: Optional[list[str]] = None) -> int:
    app = PTTApplication()
    return app.run(argv if argv is not None else sys.argv)


if __name__ == "__main__":
    sys.exit(main())
