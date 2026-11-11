from __future__ import annotations

import io

from app.terminal import TerminalSpinner


class FakeTTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_non_tty_prints_progress_as_log_lines() -> None:
    output = io.StringIO()
    with TerminalSpinner("Starting", stream=output) as spinner:
        spinner.update("Running tool: web_search")
    text = output.getvalue()
    assert "Starting" in text
    assert "Running tool: web_search" in text
    assert "Done." in text
    assert "\\r" not in text


def test_tty_spinner_clears_line_and_finishes_cleanly() -> None:
    output = FakeTTY()
    spinner = TerminalSpinner("Starting", stream=output)
    with spinner:
        spinner.update("Running tool: " + ("very_long_tool_name_" * 10))
    text = output.getvalue()
    assert "\\r" not in text  # actual carriage returns are used, not printable escapes
    assert "\r" in text
    assert "Done." in text
    # Every animation frame is clipped to the terminal width minus one.
    import re

    frame_lines = [part for part in text.split("\r") if part and not part.isspace()]
    frame_lines = [re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line) for line in frame_lines]
    assert all(len(line) <= 79 for line in frame_lines if "Done." not in line)
