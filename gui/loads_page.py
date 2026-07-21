"""Loads — its own full-window page (like the Laptime / Design City pages).

Left:  inputs (load case, corner, resultant/components).
Right: a LIVE, hoverable 3-D Load view (a second View3D driven from the one
       solved model) on top, and the full every-point load table below.

Selecting a corner + case isolates that corner in the embedded 3-D view, draws
the force vectors, and lets you hover any arrow to read its load — exactly like
the HTML loads viewer, but in-page (no redirect to the suspension GUI).  The
'All corners' entry exits isolation in one click.  Inputs -> table -> picture
all read the SAME solved model (ONE MODEL).
"""
import numpy as np
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLabel, QComboBox,
                             QRadioButton, QButtonGroup, QPushButton, QTableWidget,
                             QTableWidgetItem, QHeaderView, QAbstractItemView, QSplitter)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from gui.wheel_package import CASES, _load_items

_CAT_COLOR = {'CHASSIS': QColor(235, 235, 245), 'UPRIGHT': QColor(240, 190, 60),
              'ROCKER': QColor(180, 195, 235), 'ARB': QColor(180, 195, 235),
              'TYRE': QColor(240, 190, 60)}

# corner selector entries ('All' -> no isolation)
_CORNERS = ['All corners', 'FL', 'FR', 'RL', 'RR']


class LoadsPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self._main = main
        self._v3d = None            # lazily built on first show (needs GL context)
        self.setStyleSheet('QWidget{background:#0b0b0d;color:#e6e6e6;}'
                           'QComboBox,QPushButton{background:#1e1e24;border:1px solid #444;'
                           'border-radius:4px;padding:4px 8px;}'
                           'QPushButton:hover{background:#2a2a32;}')
        root = QHBoxLayout(self); root.setContentsMargins(12, 12, 12, 12); root.setSpacing(14)

        # ── left: inputs ──
        left = QVBoxLayout(); left.setSpacing(8)
        def hdr(t):
            l = QLabel(t); l.setStyleSheet('font-weight:bold;color:#e0a83a;font-size:13px'); return l
        left.addWidget(hdr('LOADS'))
        left.addWidget(QLabel('Load case'))
        self._case = QComboBox(); self._case.addItems([c[0] for c in CASES])
        # Default to a case with BOTH cornering and braking so every load shows at
        # once: pure cornering has no brake torque (caliper reads as "missing"),
        # pure braking has little ARB load.  Prefer lat&lon both non-zero, else any
        # braking case.
        _combined = [i for i, c in enumerate(CASES) if abs(c[1]) > 1e-6 and abs(c[2]) > 1e-6]
        _braking = [i for i, c in enumerate(CASES) if abs(c[2]) > 1e-6]
        if _combined:
            self._case.setCurrentIndex(_combined[0])
        elif _braking:
            self._case.setCurrentIndex(_braking[0])
        left.addWidget(self._case)
        left.addWidget(QLabel('Corner'))
        self._corner = QComboBox(); self._corner.addItems(_CORNERS); left.addWidget(self._corner)
        left.addWidget(QLabel('3D vectors'))
        self._res = QRadioButton('Resultant'); self._comp = QRadioButton('Components (X/Y/Z)')
        self._res.setChecked(True)
        bg = QButtonGroup(self); bg.addButton(self._res); bg.addButton(self._comp)
        left.addWidget(self._res); left.addWidget(self._comp)
        self._note = QLabel('Hover any force arrow in the 3-D view to read its load.\n\n'
                            'CHASSIS = reaction into the frame pickups (tension pulls it '
                            'outboard).\nUPRIGHT = ball joints, wheel bearings (radial + axial), '
                            'caliper, tyre patch.\nROCKER/ARB = pushrod / spring / drop-link '
                            '(axial) + the rocker pivot (the only moment).')
        self._note.setWordWrap(True); self._note.setStyleSheet('color:#999;font-size:11px')
        left.addWidget(self._note); left.addStretch(1)
        lw = QWidget(); lw.setLayout(left); lw.setFixedWidth(270); root.addWidget(lw)

        # ── right: LIVE 3-D view (top) + outputs table (bottom) ──
        self._split = QSplitter(Qt.Orientation.Vertical)
        self._view_host = QWidget()
        self._view_host_lay = QVBoxLayout(self._view_host)
        self._view_host_lay.setContentsMargins(0, 0, 0, 0)
        self._view_ph = QLabel('3-D Load view — select a corner and load case.')
        self._view_ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._view_ph.setStyleSheet('color:#666;background:#000;')
        self._view_host_lay.addWidget(self._view_ph)
        self._split.addWidget(self._view_host)

        self._tbl = QTableWidget(0, 5)
        self._tbl.setHorizontalHeaderLabels(['Category', 'Point / member', 'Magnitude (N)',
                                             'Direction (lat / fore-aft / vert)', 'Type'])
        self._tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._tbl.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._tbl.setStyleSheet('QTableWidget{background:#111;gridline-color:#333;}'
                                'QHeaderView::section{background:#1e1e24;color:#e0a83a;'
                                'padding:5px;border:0;}')
        self._tbl.setSortingEnabled(True)
        self._split.addWidget(self._tbl)
        self._split.setStretchFactor(0, 3)
        self._split.setStretchFactor(1, 2)
        root.addWidget(self._split, stretch=1)

        for w in (self._case, self._corner):
            w.currentTextChanged.connect(self._on_input)
        self._res.toggled.connect(self._on_input)
        self.refresh()

    # ── corner selection -> isolation arg (None for 'All corners') ──
    def _corner_arg(self):
        c = self._corner.currentText()
        return None if c.startswith('All') else c

    def _case_g(self):
        return next((c[1], c[2]) for c in CASES if c[0] == self._case.currentText())

    def _ensure_view(self):
        """Build the embedded View3D on first use (its GL canvas needs a running
        QApplication + a shown window, so we defer it out of __init__)."""
        if self._v3d is not None:
            return
        try:
            from gui.view3d import View3D
            self._v3d = View3D()
            self._view_host_lay.removeWidget(self._view_ph)
            self._view_ph.hide()
            self._view_host_lay.addWidget(self._v3d.native)
        except Exception:
            self._v3d = None

    def _on_input(self, *_):
        self.refresh()

    def refresh(self, *_):
        self._refresh_table()
        self._refresh_3d()

    def _refresh_table(self):
        try:
            lat, lon = self._case_g()
            items = _load_items(self._main, lat, lon, only_corner=self._corner_arg())
        except Exception as e:
            self._tbl.setRowCount(1)
            self._tbl.setItem(0, 0, QTableWidgetItem(f'Error: {e}'))
            return
        self._tbl.setSortingEnabled(False)
        self._tbl.setRowCount(len(items))
        for r, (p, v, col, lab) in enumerate(items):
            parts = lab.split(' · ')
            cat = parts[0].split(' ', 1)[-1] if ' ' in parts[0] else parts[0]
            point = parts[1] if len(parts) > 1 else ''
            mag = float(np.linalg.norm(v))
            typ = ''
            if 'tension' in lab:
                typ = 'tension'
            elif 'compression' in lab:
                typ = 'compression'
            elif 'AXIAL' in lab or 'axial' in lab:
                typ = 'axial'
            elif 'RADIAL' in lab:
                typ = 'radial'
            elif 'moment' in lab:
                typ = 'moment reaction'
            cells = [cat, point, f'{mag:,.0f}',
                     f'{v[0]:+,.0f} / {v[1]:+,.0f} / {v[2]:+,.0f}', typ]
            magitem = QTableWidgetItem()
            magitem.setData(Qt.ItemDataRole.DisplayRole, round(mag))   # numeric sort
            for cndx, txt in enumerate(cells):
                it = magitem if cndx == 2 else QTableWidgetItem(str(txt))
                if cndx == 0:
                    it.setForeground(_CAT_COLOR.get(cat, QColor(220, 220, 220)))
                self._tbl.setItem(r, cndx, it)
        self._tbl.setSortingEnabled(True)
        self._tbl.sortItems(2, Qt.SortOrder.DescendingOrder)

    def _refresh_3d(self):
        """Drive the embedded 3-D view into Load mode for this corner + case."""
        self._ensure_view()
        if self._v3d is None:
            return
        try:
            lat, lon = self._case_g()
            vec = 'components' if self._comp.isChecked() else 'resultant'
            self._main.build_load_view(self._v3d, self._corner_arg(), lat, lon, vec)
        except Exception:
            pass
