"""
The bot should mimic human movements and leave no trace behind.

"""

from __future__ import annotations

import sys
import time
from typing import Protocol

VALID_KEYS = ("w", "a", "s", "d", "q", "r", "e")


class InputBackend(Protocol):
    def key_down(self, key: str) -> None: ...

    def key_up(self, key: str) -> None: ...


class RecordingBackend:
    """Records events; used by tests and by the simulator."""

    def __init__(self) -> None:
        self.pressed: set[str] = set()
        self.events: list[tuple[str, str]] = []  # (kind, key)

    def key_down(self, key: str) -> None:
        assert key in VALID_KEYS, key
        assert key not in self.pressed, f"{key} pressed twice without release"
        self.pressed.add(key)
        self.events.append(("down", key))

    def key_up(self, key: str) -> None:
        assert key in self.pressed, f"{key} released while not pressed"
        self.pressed.discard(key)
        self.events.append(("up", key))

    def taps(self, key: str) -> int:
        return sum(1 for k, x in self.events if k == "down" and x == key)

    def clear_events(self) -> None:
        self.events.clear()


class NullBackend:
    """Shadow mode: accepts commands and does nothing."""

    def key_down(self, key: str) -> None:  # noqa: D102
        pass

    def key_up(self, key: str) -> None:  # noqa: D102
        pass


# DirectInput scan codes (set 1)
SCAN_CODES = {"w": 0x11, "a": 0x1E, "s": 0x1F, "d": 0x20, "q": 0x10, "r": 0x13, "e": 0x12}


class WindowsSendInputBackend:  # pragma: no cover - Windows only
    def __init__(self, require_window_title: str | None = None) -> None:
        if sys.platform != "win32":
            raise RuntimeError("WindowsSendInputBackend only works on Windows")
        import ctypes
        from ctypes import wintypes

        self._ct = ctypes
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._title = require_window_title

        ULONG_PTR = ctypes.c_size_t

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR),
            ]

        class _U(ctypes.Union):
            _fields_ = [("ki", KEYBDINPUT), ("pad", ctypes.c_byte * 32)]

        class INPUT(ctypes.Structure):
            _anonymous_ = ("u",)
            _fields_ = [("type", wintypes.DWORD), ("u", _U)]

        self._KEYBDINPUT, self._INPUT = KEYBDINPUT, INPUT
        self.pressed: set[str] = set()

    def game_has_focus(self) -> bool:
        if not self._title:
            return True
        hwnd = self._user32.GetForegroundWindow()
        n = self._user32.GetWindowTextLengthW(hwnd)
        buf = self._ct.create_unicode_buffer(n + 1)
        self._user32.GetWindowTextW(hwnd, buf, n + 1)
        return self._title.lower() in buf.value.lower()

    def _send(self, key: str, up: bool) -> None:
        KEYEVENTF_SCANCODE, KEYEVENTF_KEYUP = 0x0008, 0x0002
        flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if up else 0)
        inp = self._INPUT(type=1)
        inp.ki = self._KEYBDINPUT(0, SCAN_CODES[key], flags, 0, 0)
        self._user32.SendInput(1, self._ct.byref(inp), self._ct.sizeof(inp))

    def key_down(self, key: str) -> None:
        if not self.game_has_focus():
            return
        self._send(key, up=False)
        self.pressed.add(key)

    def key_up(self, key: str) -> None:
        self._send(key, up=True)  # always allow releases, even without focus
        self.pressed.discard(key)


def monotonic() -> float:
    return time.perf_counter()
