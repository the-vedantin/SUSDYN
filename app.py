"""
app.py — Vahan entry point

Run from SUSDYN directory:
    python app.py
"""
import sys

from PyQt6.QtWidgets import QApplication

from gui.main_window import launch
from gui.wheel_guard import install_wheel_guard


_APP = None   # keeps the QApplication alive: an unreferenced one is garbage-
              # collected and launch() would silently build a second, unguarded one.


def make_app() -> QApplication:
    """The QApplication exactly as the running app has it: created (or reused)
    and with the app-wide wheel guard installed (mouse wheel over an unfocused
    number box / dropdown / slider scrolls the panel, never the value).
    `launch()` picks this instance up via QApplication.instance()."""
    global _APP
    _APP = QApplication.instance() or QApplication(sys.argv)
    install_wheel_guard(_APP)
    return _APP


if __name__ == '__main__':
    make_app()
    launch()
