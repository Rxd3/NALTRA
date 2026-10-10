"""Shared fixtures."""

from __future__ import annotations

import builtins
import io

import pytest


@pytest.fixture
def posix_text_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Write text files as on Linux/macOS, where newline=None leaves "\\n" untranslated.

    Patches io.open (behind Path.open/write_text) and builtins.open; os.linesep cannot be
    patched because the io module fixes the platform newline at compile time.
    """
    real_open = io.open

    def posix_open(
        file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *a, **kw
    ):
        if newline is None and "w" in mode and "b" not in mode:
            newline = "\n"
        return real_open(file, mode, buffering, encoding, errors, newline, *a, **kw)

    monkeypatch.setattr(io, "open", posix_open)
    monkeypatch.setattr(builtins, "open", posix_open)
