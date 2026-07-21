"""Loads — its own full-window page (like the Laptime / Design City pages).

Inputs (load case, corner, vector mode) + outputs (a table of EVERY load in the
wheel package, grouped CHASSIS / UPRIGHT / ROCKER-ARB) on one page.  Selecting a
corner + case also drives the live 3-D Load view (isolated, force vectors,
hover) so inputs -> outputs -> picture stay on the one solved model.
"""
import numpy as np
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLabel, QComboBox,
                             QRadioButton, QButtonGroup, QPushButton, QTableWidget,
                             QTableWidgetItem, QHeaderView, QAbstractItemView)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from gui.wheel_package import CASES, _load_items

_CAT_COLOR = {'CHASSIS': QColor(235, 235, 245), 'UPRIGHT': QColor(240, 190, 60),
              'ROCKER': QColor(180, 195, 235), 'ARB': QColor(180, 195, 235),
              'TYRE': QColor(240, 190, 60)}


class LoadsPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self._main = main
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
        self._case = QComboBox(); self._case.addItems([c[0] for c in CASES]); left.addWidget(self._case)
        left.addWidget(QLabel('Corner'))
        self._corner = QComboBox(); self._corner.addItems(['FL', 'FR', 'RL', 'RR']); left.addWidget(self._corner)
        left.addWidget(QLabel('3D vectors'))
        self._res = QRadioButton('Resultant'); self._comp = QRadioButton('Components (X/Y/Z)')
        self._res.setChecked(True)
        bg = QButtonGroup(self); bg.addButton(self._res); bg.addButton(self._comp)
        left.addWidget(self._res); left.addWidget(self._comp)
        self._show3d = QPushButton('Show this corner in the 3-D Load view')
        self._show3d.clicked.connect(self._drive_3d)
        left.addWidget(self._show3d)
        self._note = QLabel('CHASSIS = reaction into the frame pickups (tension pulls it '
                            'outboard).\nUPRIGHT = ball joints, wheel bearings (radial + axial), '
                            'caliper, tyre patch.\nROCKER/ARB = pushrod / spring / drop-link '
                            '(axial) + the rocker pivot (the only moment).')
        self._note.setWordWrap(True); self._note.setStyleSheet('color:#999;font-size:11px')
        left.addWidget(self._note); left.addStretch(1)
        lw = QWidget(); lw.setLayout(left); lw.setFixedWidth(270); root.addWidget(lw)

        # ── right: outputs table ──
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
        root.addWidget(self._tbl, stretch=1)

        for w in (self._case, self._corner):
            w.currentTextChanged.connect(self.refresh)
        self.refresh()

    def _case_g(self):
        return next((c[1], c[2]) for c in CASES if c[0] == self._case.currentText())

    def refresh(self, *_):
        try:
            lat, lon = self._case_g()
            lbl = self._corner.currentText()
            items = _load_items(self._main, lat, lon, only_corner=lbl)
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

    def _drive_3d(self):
        """Put the main 3-D view into Load mode isolated on this corner."""
        m = self._main
        try:
            lat, lon = self._case_g()
            m._dynamics_panel._lat_g.setValue(lat); m._dynamics_panel._lon_g.setValue(lon)
            m._car['view_mode'] = 'load'
            m._car['load_vec_mode'] = 'components' if self._comp.isChecked() else 'resultant'
            m._car['wheel_pkg_corner'] = self._corner.currentText()
            try:
                m.view3d.sync_view_controls(view_mode='load')
            except Exception:
                pass
            m._switch_page(0)          # back to the suspension page to see the 3-D view
            m._update_3d()
        except Exception:
            pass
