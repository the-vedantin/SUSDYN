"""
gui/wheel_guard.py — app-wide "scroll the panel, not the value" guard.

Problem: rolling the mouse wheel over a number box / dropdown / slider that
happens to sit under the cursor while scrolling a long panel silently changes
that value (the user changed the rack width this way).  Only the hardpoint
coordinate boxes (`_NoScrollSpin` / `_NoScrollCombo` in gui/panels.py)
protected themselves; everything else built from a plain QSpinBox /
QDoubleSpinBox / QComboBox / QSlider did not.

Fix: ONE QObject event filter on the QApplication (installed in app.py).
For a QEvent.Wheel whose receiver is a QAbstractSpinBox / QComboBox /
QAbstractSlider (but NOT a QScrollBar — those are the scroll mechanism
itself) that does NOT have keyboard focus:

  * the value widget never sees the wheel (filter returns True), and
  * the wheel is re-sent to the viewport of the nearest enclosing
    QAbstractScrollArea so the panel scrolls.  Explicit forwarding is
    required: Qt 6's QApplication::notify only propagates SPONTANEOUS wheel
    events up the parent chain — `event.ignore(); return True` would work
    for a real mouse but a synthesised/sent event would just be swallowed,
    and forwarding is deterministic for both.  If that scroll area is at its
    limit (event not accepted) the next enclosing one is tried.

A widget the user has clicked (has focus) is left alone, so the wheel still
steps the value exactly as before.  Keyboard/typing is untouched.

Focus policy: Qt gives WheelFocus widgets focus on the first wheel notch
BEFORE any event filter runs, so the filter also downgrades WheelFocus ->
StrongFocus on the value widgets the first time it sees them (Polish / Show /
Enter — all precede a wheel).  Hovering + wheel therefore never focuses a box;
a click still does.  NoFocus widgets are left as they are.

The existing `_NoScrollSpin` / `_NoScrollCombo` classes are not double-handled:
unfocused, the filter forwards before their wheelEvent runs; focused, the
filter passes the event through and they behave as before.
"""
from PyQt6.QtCore import QObject, QEvent, Qt, QPointF
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import (QApplication, QAbstractSpinBox, QComboBox,
                             QAbstractSlider, QScrollBar, QAbstractScrollArea,
                             QWidget)

_VALUE_WIDGETS = (QAbstractSpinBox, QComboBox, QAbstractSlider)
_POLICY_EVENTS = (QEvent.Type.Polish, QEvent.Type.Show, QEvent.Type.Enter)


def _is_value_widget(obj) -> bool:
    return isinstance(obj, _VALUE_WIDGETS) and not isinstance(obj, QScrollBar)


def _enclosing_scroll_areas(w: QWidget):
    """Yield QAbstractScrollArea ancestors of `w`, nearest first."""
    p = w.parentWidget()
    while p is not None:
        if isinstance(p, QAbstractScrollArea):
            yield p
        p = p.parentWidget()


def _clone_wheel_for(target: QWidget, ev: QWheelEvent) -> QWheelEvent:
    gpos = ev.globalPosition()
    local = QPointF(target.mapFromGlobal(gpos.toPoint()))
    out = QWheelEvent(local, gpos, ev.pixelDelta(), ev.angleDelta(),
                      ev.buttons(), ev.modifiers(), ev.phase(),
                      ev.inverted())
    return out


class WheelGuard(QObject):
    """See module docstring.  Install with `install_wheel_guard(app)`."""

    def eventFilter(self, obj, event):
        et = event.type()
        if et in _POLICY_EVENTS:
            if _is_value_widget(obj) and obj.focusPolicy() == Qt.FocusPolicy.WheelFocus:
                obj.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            return False
        if et != QEvent.Type.Wheel or not _is_value_widget(obj):
            return False
        if obj.hasFocus():
            return False                      # user clicked it: wheel steps the value
        # Not focused: never touch the value; scroll the panel instead.
        event.ignore()
        for sa in _enclosing_scroll_areas(obj):
            fwd = _clone_wheel_for(sa.viewport(), event)
            fwd.ignore()
            QApplication.sendEvent(sa.viewport(), fwd)
            if fwd.isAccepted():
                break                         # this scroll area moved
        return True                           # value widget never sees it


_GUARD = None


def install_wheel_guard(app: QApplication) -> WheelGuard:
    """Install the guard once on `app`; returns the (kept-alive) filter."""
    global _GUARD
    if _GUARD is None:
        _GUARD = WheelGuard(app)              # parented to the app -> lives as long as it
        app.installEventFilter(_GUARD)
    return _GUARD
