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
        self.canvas = _ExportCanvas(fig)
        lay.addWidget(self.canvas)
