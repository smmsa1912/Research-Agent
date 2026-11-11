"""Small, PowerShell-friendly terminal progress indicator."""

from __future__ import annotations

import shutil
import sys
import threading
import time
from typing import TextIO


class TerminalSpinner:
    """Show a single-line spinner without allowing status text to wrap.

    Long progress labels are clipped to the current terminal width. This is
    important on Windows PowerShell: carriage return only returns to the
    beginning of the current visual line, so an overlong spinner can wrap and
    leave fragments behind on every animation frame.

    When stdout is redirected (and is not a TTY), progress changes are emitted
    as ordinary log lines instead of using cursor-control characters.
    """

    FRAMES = ("|", "/", "-", "\\")

    def __init__(self, initial_status: str = "Working", *, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stdout
        self.status = initial_status
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._interactive = bool(getattr(self.stream, "isatty", lambda: False)())

    def _line_width(self) -> int:
        try:
            columns = shutil.get_terminal_size(fallback=(80, 24)).columns
        except (OSError, ValueError):
            columns = 80
        # Leave the final column unused so the terminal never auto-wraps.
        return max(20, columns - 1)

    def update(self, status: str) -> None:
        with self._lock:
            self.status = status
            if not self._interactive:
                print(f"  -> {status}", file=self.stream, flush=True)

    def _animate(self) -> None:
        index = 0
        started = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                status = self.status
                elapsed = int(time.monotonic() - started)
                width = self._line_width()
                frame = self.FRAMES[index % len(self.FRAMES)]
                line = f"{frame} {status}... ({elapsed}s)"
                # Fixed-width output both removes stale characters and avoids
                # wrapping in narrow PowerShell windows.
                line = line[:width].ljust(width)
                # Clear the whole terminal row before drawing; this avoids
                # leftover characters when PowerShell redraws a shorter status.
                self.stream.write("\r\x1b[2K" + line)
                self.stream.flush()
            index += 1
            self._stop.wait(0.12)

    def __enter__(self) -> "TerminalSpinner":
        if self._interactive:
            self._thread = threading.Thread(target=self._animate, daemon=True)
            self._thread.start()
        else:
            print(f"  -> {self.status}", file=self.stream, flush=True)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            width = self._line_width()
            self.stream.write("\r\x1b[2K")
            self.stream.flush()
        if exc_type is None:
            print("  Done.\n", file=self.stream, flush=True)
        else:
            print("  Stopped because an error occurred.\n", file=self.stream, flush=True)
