"""
gui/plot_dialog.py — Reusable popup for showing a matplotlib Figure
with the right-click copy/save menu (matches the one on CurvesCanvas).
"""
from __future__ import annotations
import io

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QMenu, QApplication,
                              QFileDialog, QSizePolicy)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QAction

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


def whites_to_black(fig, threshold=0.85):
    """Light-theme export: every near-white artist (the dark theme's white RL / reference
    curves, markers, fills, text) would vanish on a white page, so turn it black.  Returns a
    callable that restores the original colours (user 2026-11-14: "RL curve should become
    black so it's actually visible")."""
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.text import Text
    from matplotlib.patches import Patch
    from matplotlib.collections import Collection
    undo = []

    def _white(c):
        try:
            r, g, b, a = to_rgba(c)
        except (ValueError, TypeError):
            return False
        return a > 0 and min(r, g, b) >= threshold

    def _swap(get, set_, c_new='#111111'):
        try:
            c = get()
        except Exception:
            return
        if isinstance(c, str) or (hasattr(c, '__len__') and len(c) in (3, 4) and not hasattr(c[0], '__len__')):
            if _white(c):
                undo.append((set_, c)); set_(c_new)
        elif hasattr(c, '__len__') and len(c):          # array of colours (collections)
            import numpy as np
            arr = np.asarray(c, float)
            if arr.ndim == 2 and arr.shape[1] == 4:
                m = (arr[:, 3] > 0) & (arr[:, :3].min(axis=1) >= threshold)
                if m.any():
                    new = arr.copy(); new[m, :3] = 0.067
                    undo.append((set_, arr)); set_(new)

    for ax in fig.axes:
        for ln in ax.get_lines():
            _swap(ln.get_color, ln.set_color)
            _swap(ln.get_markerfacecolor, ln.set_markerfacecolor)
            _swap(ln.get_markeredgecolor, ln.set_markeredgecolor)
        for p in ax.patches:
            if isinstance(p, Patch):
                _swap(p.get_facecolor, p.set_facecolor); _swap(p.get_edgecolor, p.set_edgecolor)
        for col in ax.collections:
            if isinstance(col, Collection):
                _swap(col.get_facecolor, col.set_facecolor); _swap(col.get_edgecolor, col.set_edgecolor)
        for t in ax.texts:
            _swap(t.get_color, t.set_color)
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                _swap(t.get_color, t.set_color)
            for h in getattr(leg, 'legend_handles', None) or getattr(leg, 'legendHandles', []) or []:
                if isinstance(h, Line2D):
                    _swap(h.get_color, h.set_color)
                    _swap(h.get_markerfacecolor, h.set_markerfacecolor)
                elif isinstance(h, (Patch,)):
                    _swap(h.get_facecolor, h.set_facecolor)
            fr = leg.get_frame()
            # dark legend box on a white page: make it white with a grey edge
            fc, ec = fr.get_facecolor(), fr.get_edgecolor()
            undo.append((fr.set_facecolor, fc)); undo.append((fr.set_edgecolor, ec))
            fr.set_facecolor('white'); fr.set_edgecolor('#999999')
        for t in (ax.title, ax.xaxis.label, ax.yaxis.label):
            if isinstance(t, Text):
                _swap(t.get_color, t.set_color)
    for t in fig.texts:
        _swap(t.get_color, t.set_color)

    def restore():
        for set_, c in reversed(undo):
            try:
                set_(c)
            except Exception:
                pass
    return restore


class _ExportCanvas(FigureCanvas):
    """FigureCanvas with right-click 'Copy / Save image' menu.

    Reused by every analysis-plot dialog and matches the menu on the
    main CurvesCanvas so behaviour is consistent across the GUI.
    """

    def __init__(self, fig):
        super().__init__(fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        a = QAction('Copy graph (as displayed)', self)
        a.triggered.connect(self._copy_as_displayed)
        menu.addAction(a)

        b = QAction('Copy graph (light theme, for documents)', self)
        b.triggered.connect(self._copy_light)
        menu.addAction(b)

        menu.addSeparator()

        c = QAction('Save graph as image (as displayed)…', self)
        c.triggered.connect(lambda: self._save(light=False))
        menu.addAction(c)

        d = QAction('Save graph as image (light theme)…', self)
        d.triggered.connect(lambda: self._save(light=True))
        menu.addAction(d)

        menu.exec(event.globalPos())

    def _fig_to_bytes(self, light=False, fmt='png', dpi=200):
        buf = io.BytesIO()
        if light:
            self._with_light(lambda: self.figure.savefig(
                buf, format=fmt, dpi=dpi, bbox_inches='tight', facecolor='white'))
        else:
            self.figure.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight',
                                facecolor=self.figure.get_facecolor())
        buf.seek(0)
        return buf.getvalue()

    def _with_light(self, fn):
        _restore_whites = whites_to_black(self.figure)
        try:
            self._with_light_inner(fn)
        finally:
            _restore_whites()

    def _with_light_inner(self, fn):
        original_fig = self.figure.get_facecolor()
        per_axes = []
        for ax in self.figure.axes:
            per_axes.append({
                'fc': ax.get_facecolor(),
                'spines': {k: s.get_edgecolor() for k, s in ax.spines.items()},
                'xc': ax.xaxis.label.get_color(),
                'yc': ax.yaxis.label.get_color(),
            })
        # Apply light
        self.figure.set_facecolor('white')
        for ax in self.figure.axes:
            ax.set_facecolor('white')
            for s in ax.spines.values():
                s.set_edgecolor('#666')
            ax.tick_params(colors='#333')
            ax.xaxis.label.set_color('#222')
            ax.yaxis.label.set_color('#222')
            ax.grid(True, color='#DDD', lw=0.6)
            if ax.get_title():
                ax.title.set_color('#111')
        suptitle_color = (self.figure._suptitle.get_color()
                          if self.figure._suptitle else None)
        if self.figure._suptitle:
            self.figure._suptitle.set_color('#111')
        try:
            fn()
        finally:
            # Restore
            self.figure.set_facecolor(original_fig)
            for ax, orig in zip(self.figure.axes, per_axes):
                ax.set_facecolor(orig['fc'])
                for k, s in ax.spines.items():
                    s.set_edgecolor(orig['spines'].get(k, '#222'))
                ax.tick_params(colors='#777')
                ax.xaxis.label.set_color(orig['xc'])
                ax.yaxis.label.set_color(orig['yc'])
                ax.grid(True, color='#1a1a1a', lw=0.5)
            if self.figure._suptitle and suptitle_color:
                self.figure._suptitle.set_color(suptitle_color)
            self.draw_idle()

    def _copy_as_displayed(self):
        QApplication.clipboard().setImage(QImage.fromData(self._fig_to_bytes(False), 'PNG'))

    def _copy_light(self):
        QApplication.clipboard().setImage(QImage.fromData(self._fig_to_bytes(True), 'PNG'))

    def _save(self, light=False):
        default = 'graph_light.png' if light else 'graph.png'
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save graph', default, 'PNG (*.png);;PDF (*.pdf);;SVG (*.svg)')
        if not path:
            return
        fmt = 'png'
        if path.lower().endswith('.pdf'): fmt = 'pdf'
        elif path.lower().endswith('.svg'): fmt = 'svg'
        with open(path, 'wb') as f:
            f.write(self._fig_to_bytes(light=light, fmt=fmt, dpi=300))


class PlotDialog(QDialog):
    """Reusable popup wrapping a matplotlib Figure with copy/save menu.

    Use:
        dlg = PlotDialog(parent, fig, title='My Plot')
        dlg.show()    # non-blocking
    """

    def __init__(self, parent, fig, title='Plot'):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1000, 700)
        lay = QVBoxLayout(self)
        self.canvas = ReadableCanvas(fig)   # hover-to-read + copy on every popup plot
        lay.addWidget(self.canvas)


class ReadableCanvas(_ExportCanvas):
    """THE graph canvas for every page (user 2026-09-26: "no hover to read, no
    copy graph ... i need reading features for all graphs"): hover anywhere to
    read every curve's value at the cursor (gui.main_window.HoverAnnotator,
    the reader the main curves already use) + right-click copy / save (as
    displayed or light theme).  Drop-in for FigureCanvasQTAgg(fig)."""

    def __init__(self, fig):
        super().__init__(fig)
        from gui.main_window import HoverAnnotator      # lazy: main_window imports this module
        self.hover = HoverAnnotator(self)
