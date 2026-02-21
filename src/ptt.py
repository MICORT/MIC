#!/usr/bin/env python3
"""
Push-to-Talk Speech-to-Text (Polish / VOSK)

Modes:
    terminal   - Hold key in terminal, text prints to stdout
    type       - Hold key in terminal, text is TYPED into last active window via wtype

Usage:
    ptt                    - Terminal mode, hold SPACE, continuous
    ptt -m type            - Type mode: text goes to your active window (terminal, browser, etc.)
    ptt -k m               - Use 'm' key instead of space
    ptt -o file.txt        - Also append output to file

Controls:
    Hold key  = recording
    Release   = stop + transcribe + print/type text
    Ctrl+C    = exit
"""

import sys
import os
import io
import json
import wave
import argparse
import termios
import tty
import select
import subprocess
import time
import numpy as np
import sounddevice as sd
import vosk

MODEL_PATH = os.path.expanduser("~/stt-models/polish")
SAMPLE_RATE = 16000
CHANNELS = 1
RELEASE_TIMEOUT = 0.25

# ── colours ──────────────────────────────────────────────────────────
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"
CLR = "\033[2K\r"


def load_model():
    vosk.SetLogLevel(-1)
    sys.stderr.write(f"{CYAN}Loading model...{RESET}\n")
    model = vosk.Model(MODEL_PATH)
    sys.stderr.write(f"{GREEN}Model ready.{RESET}\n")
    return model


def transcribe(model, audio_data: np.ndarray) -> str:
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
        if len(data) == 0:
            break
        rec.AcceptWaveform(data)
    result = json.loads(rec.FinalResult())
    return result.get("text", "").strip()


def wtype_text(text: str):
    """Type text into the previously active window using wtype."""
    try:
        subprocess.run(["wtype", "--", text], check=True, timeout=5)
    except FileNotFoundError:
        sys.stderr.write(f"{RED}wtype not found! Install: sudo apt install wtype{RESET}\n")
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"{RED}wtype error: {e}{RESET}\n")


class Recorder:
    def __init__(self):
        self.frames = []
        self.recording = False
        self.stream = None

    def start(self):
        self.frames = []
        self.recording = True
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=1024,
            callback=self._callback,
        )
        self.stream.start()

    def _callback(self, indata, frame_count, time_info, status):
        if self.recording:
            self.frames.append(indata.copy())

    def stop(self) -> np.ndarray:
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if self.frames:
            return np.concatenate(self.frames, axis=0).flatten()
        return np.array([], dtype=np.int16)


def is_key_pressed(trigger_char: str, timeout: float = 0.05) -> bool:
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if rlist:
        ch = sys.stdin.read(1)
        while select.select([sys.stdin], [], [], 0.01)[0]:
            sys.stdin.read(1)
        if trigger_char == " ":
            return ch == " "
        return ch.lower() == trigger_char.lower()
    return False


def run(args):
    model = load_model()
    recorder = Recorder()
    trigger = args.key
    trigger_label = "SPACE" if trigger == " " else trigger.upper()
    mode = args.mode

    mode_desc = {
        "terminal": "text prints here",
        "type": "text TYPED into active window (wtype)",
    }

    sys.stderr.write(
        f"\n{BOLD}{YELLOW}Push-to-Talk STT{RESET}  [{mode}: {mode_desc[mode]}]\n"
        f"Hold {BOLD}{trigger_label}{RESET} to record, release to transcribe.\n"
        f"Press {BOLD}Ctrl+C{RESET} to exit.\n\n"
    )

    old_settings = termios.tcgetattr(sys.stdin)

    try:
        tty.setcbreak(sys.stdin.fileno())

        while True:
            sys.stderr.write(f"{CLR}{CYAN}[ Ready ] Hold {trigger_label} to speak...{RESET}")
            sys.stderr.flush()

            while not is_key_pressed(trigger):
                pass

            recorder.start()
            sys.stderr.write(f"{CLR}{RED}{BOLD}[ REC ] Recording... release {trigger_label} to stop{RESET}")
            sys.stderr.flush()

            last_key_time = time.monotonic()
            while True:
                rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch == "\x03":
                        raise KeyboardInterrupt
                    while select.select([sys.stdin], [], [], 0.01)[0]:
                        sys.stdin.read(1)
                    last_key_time = time.monotonic()
                else:
                    if time.monotonic() - last_key_time > RELEASE_TIMEOUT:
                        break

            audio = recorder.stop()
            duration = len(audio) / SAMPLE_RATE

            if duration < 0.3:
                sys.stderr.write(f"{CLR}{YELLOW}[ Skip ] Too short ({duration:.1f}s){RESET}\n")
                continue

            sys.stderr.write(f"{CLR}{YELLOW}[ ... ] Transcribing {duration:.1f}s...{RESET}")
            sys.stderr.flush()

            text = transcribe(model, audio)

            if text:
                sys.stderr.write(f"{CLR}{GREEN}[ OK ]{RESET} {text}\n")

                if mode == "type":
                    # Restore terminal before wtype so focus returns
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                    time.sleep(0.1)
                    wtype_text(text)
                    tty.setcbreak(sys.stdin.fileno())
                else:
                    print(text, flush=True)

                if args.output:
                    with open(args.output, "a", encoding="utf-8") as f:
                        f.write(text + "\n")
            else:
                sys.stderr.write(f"{CLR}{YELLOW}[ ... ] No speech detected{RESET}\n")

    except KeyboardInterrupt:
        sys.stderr.write(f"\n{CYAN}Bye!{RESET}\n")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        recorder.stop()


def main():
    parser = argparse.ArgumentParser(description="Push-to-Talk STT (Polish)")
    parser.add_argument("-m", "--mode", choices=["terminal", "type"], default="terminal",
                        help="terminal = print text here; type = wtype into active window")
    parser.add_argument("-k", "--key", default=" ",
                        help="Trigger key (default: space)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Append transcriptions to file")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
