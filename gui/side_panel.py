"""Side control panels: never silently clipped, never forced wide.

History:
- 2026-11-14 ("the side panels just clip and hide half the shit"): panels
  were fixed-width scroll areas with the sideways bar off, so the right part
  of rows sat off-screen.  The first fix made the panel's MINIMUM = its widest
  row, which stopped the clipping but
- 2026-11-15 ("can't shrink side panels past a certain point, takes 1/4 of
  the screen each side, BAD for viewing graphs"): cost the graphs their area.

Now:
- a panel can be dragged as narrow as MIN_W; if its rows are wider than the
  panel a sideways scroll bar appears (nothing hidden without a bar showing);
- drag the divider all the way to the edge, or DOUBLE-CLICK the divider, and
  the panel collapses to nothing; double-click again (or drag) to bring it
  back at its last width;
- each page's divider positions are remembered between runs (QSettings).
"""
from __future__ import annotations

from PyQt6.QtCore import QSettings, QSize, Qt
from PyQt6.QtWidgets import (QLayout, QScrollArea, QSplitter,
                             QSplitterHandle, QWidget)

MIN_W = 120          # narrowest a side panel can be dragged (px) before collapse
HANDLE_W = 8         # divider width: wide enough to grab and double-click


class SideScroll(QScrollArea):
    def __init__(self, inner: QWidget, width: int = 300, parent=None):
        super().__init__(parent)
        self._pref_w = int(width)
        self.setWidget(inner)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        # narrower than the rows -> sideways bar, so nothing is hidden silently
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    def minimumSizeHint(self) -> QSize:
        return QSize(MIN_W, super().minimumSizeHint().height())

    def sizeHint(self) -> QSize:
        return QSize(self._pref_w, super().sizeHint().height())


class _Handle(QSplitterHandle):
    """Divider that collapses / restores the neighbouring side panel on
    double-click."""

    def mouseDoubleClickEvent(self, ev):
        sp = self.splitter()
        if isinstance(sp, SideSplitter):
            sp.toggle_side_at_handle(sp.indexOf(self))
        else:
            super().mouseDoubleClickEvent(ev)


class SideSplitter(QSplitter):
    """Horizontal splitter whose side panels (indices in `sides`) can be
    dragged narrow, collapsed to nothing and restored; the other children
    (graphs / 3-D view) cannot collapse.  `key` names the QSettings entry
    that remembers the sizes."""

    def __init__(self, sides, key: str = '', parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._sides = set(int(i) for i in sides)
        self._key = key
        self._last = {}            # side index -> width before collapse
        self.setHandleWidth(HANDLE_W)
        self.setChildrenCollapsible(True)
        self.splitterMoved.connect(self._remember)
        self.setToolTip('')

    def createHandle(self):
        h = _Handle(self.orientation(), self)
        h.setToolTip('Drag to resize the side panel.\n'
                      'Double-click (or drag to the edge) to hide / show it.')
        return h

    def finish(self, sizes):
        """Call after adding the children: set collapsibility, then the
        remembered sizes (or `sizes` the first time)."""
        for i in range(self.count()):
            self.setCollapsible(i, i in self._sides)
        saved = None
        if self._key:
            try:
                v = QSettings('Vahan', 'Vahan').value('split/' + self._key)
                if v:
                    saved = [int(x) for x in v]
            except Exception:
                saved = None
        if saved and len(saved) == self.count() and \
                any(saved[i] > 0 for i in range(self.count()) if i not in self._sides):
            self.setSizes(saved)
            for i in self._sides:
                if saved[i] > 0:
                    self._last[i] = saved[i]
        else:
            self.setSizes(list(sizes))
        for i in self._sides:
            self._last.setdefault(i, max(MIN_W, int(sizes[i])))

    def _side_for_handle(self, h: int):
        # handle h sits between widget h-1 and widget h
        if (h - 1) in self._sides:
            return h - 1
        if h in self._sides:
            return h
        return None

    def toggle_side_at_handle(self, h: int):
        i = self._side_for_handle(h)
        if i is not None:
            self.toggle_side(i)

    def toggle_side(self, i: int):
        sizes = self.sizes()
        main = [j for j in range(self.count()) if j not in self._sides]
        if not main:
            return
        m = main[0]
        if sizes[i] > 0:                       # collapse
            self._last[i] = sizes[i]
            sizes[m] += sizes[i]
            sizes[i] = 0
        else:                                  # restore at last width
            w = max(MIN_W, int(self._last.get(i, 300)))
            w = min(w, max(MIN_W, sizes[m] - 200))
            sizes[m] -= w
            sizes[i] = w
        self.setSizes(sizes)
        self._remember()

    def _remember(self, *_):
        sizes = self.sizes()
        for i in self._sides:
            if sizes[i] > 0:
                self._last[i] = sizes[i]
        if self._key:
            try:
                QSettings('Vahan', 'Vahan').setValue('split/' + self._key, sizes)
            except Exception:
                pass


def side_split(side: QWidget, main, side_width: int = 300,
               key: str = '') -> SideSplitter:
    """[side | main] in a SideSplitter; main (a widget or a layout) takes all
    the extra width.  The side starts at side_width, can be dragged narrower
    (sideways scroll) or collapsed by double-clicking the divider."""
    if isinstance(main, QLayout):
        host = QWidget()
        host.setLayout(main)
        main = host
    sp = SideSplitter([0], key)
    sp.addWidget(side)
    sp.addWidget(main)
    sp.setStretchFactor(0, 0)
    sp.setStretchFactor(1, 1)
    sp.finish([int(side_width), 10000])
    return sp
