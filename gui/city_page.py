# -*- coding: utf-8 -*-
"""city_page.py — Design City gallery.

Grid of candidate designs from design_city.py runs: each card shows ONLY the
graphs (card.png) with a one-line tag. Click a card -> full report dialog
(metrics table + full-size graphs + the candidate's .vahan path) with
'Open in Vahan' to load it into the main window for manual iteration.
"""
import os, json, glob, subprocess, sys
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QScrollArea, QGridLayout, QFrame, QDialog, QTableWidget,
                             QTableWidgetItem, QFileDialog, QComboBox, QSpinBox,
                             QMessageBox)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt


class _Card(QFrame):
    def __init__(self, cdir, meta, on_click):
        super().__init__()
        self._cdir = cdir; self._meta = meta; self._on_click = on_click
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet('QFrame{background:#101018;border:1px solid #333;border-radius:6px;}'
                           'QFrame:hover{border:1px solid #E8A000;}')
        lay = QVBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6)
        png = os.path.join(cdir, 'card.png')
        pic = QLabel()
        if os.path.exists(png):
            pm = QPixmap(png)
            pic.setPixmap(pm.scaledToWidth(430, Qt.TransformationMode.SmoothTransformation))
        else:
            pic.setText('(no card.png)')
        lay.addWidget(pic)
        tag = QLabel('%s   gmax %.2f  margin %+.3f  %s' % (
            meta.get('id', '?'), meta.get('gmax', float('nan')),
            meta.get('margin', float('nan')), meta.get('first_axle', '')))
        tag.setStyleSheet('color:#ccc;font-size:10px;')
        lay.addWidget(tag)

    def mousePressEvent(self, ev):
        self._on_click(self._cdir, self._meta)


class _Detail(QDialog):
    def __init__(self, parent, cdir, meta, main_window):
        super().__init__(parent)
        self._mw = main_window; self._cfg = os.path.join(cdir, 'config.vahan')
        self.setWindowTitle('Design %s — full report' % meta.get('id', '?'))
        self.resize(980, 720)
        lay = QVBoxLayout(self)
        # graphs full size
        sc = QScrollArea(); sc.setWidgetResizable(True)
        inner = QWidget(); il = QVBoxLayout(inner)
        png = os.path.join(cdir, 'card.png')
        if os.path.exists(png):
            lbl = QLabel(); lbl.setPixmap(QPixmap(png).scaledToWidth(
                900, Qt.TransformationMode.SmoothTransformation))
            il.addWidget(lbl)
        # metrics table
        tbl = QTableWidget()
        rows = [(k, v) for k, v in sorted(meta.items())
                if k not in ('genome', 'scores') and not isinstance(v, dict)]
        rows += [('genome.' + k, v) for k, v in sorted(meta.get('genome', {}).items())]
        rows += [('score.' + k, round(v, 3)) for k, v in sorted(meta.get('scores', {}).items())]
        tbl.setRowCount(len(rows)); tbl.setColumnCount(2)
        tbl.setHorizontalHeaderLabels(['metric', 'value'])
        for i, (k, v) in enumerate(rows):
            tbl.setItem(i, 0, QTableWidgetItem(str(k)))
            tbl.setItem(i, 1, QTableWidgetItem(str(round(v, 4)) if isinstance(v, float) else str(v)))
        tbl.setMinimumHeight(320)
        il.addWidget(tbl)
        pth = QLabel('config: ' + self._cfg); pth.setStyleSheet('color:#888;font-size:10px;')
        pth.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        il.addWidget(pth)
        sc.setWidget(inner); lay.addWidget(sc)
        bt = QHBoxLayout()
        openb = QPushButton('Open in Vahan (load this design)')
        openb.clicked.connect(self._open_in_vahan)
        bt.addWidget(openb)
        close = QPushButton('Close'); close.clicked.connect(self.accept)
        bt.addWidget(close)
        lay.addLayout(bt)

    def _open_in_vahan(self):
        try:
            self._mw._load_project_from_path(self._cfg)
            self._mw._switch_page(0)
            self.accept()
        except Exception as e:
            QMessageBox.warning(self, 'Load failed', str(e))


class CityPage(QWidget):
    """Design City — browse population-engine candidates; click to inspect/load."""

    def __init__(self, main_window):
        super().__init__()
        self._mw = main_window
        self._run_dir = self._latest_run()
        v = QVBoxLayout(self)
        top = QHBoxLayout()
        self._dir_lbl = QLabel(self._run_dir or '(no runs found in designs_city/)')
        self._dir_lbl.setStyleSheet('color:#aaa;')
        pick = QPushButton('Pick run…'); pick.clicked.connect(self._pick)
        refresh = QPushButton('Refresh'); refresh.clicked.connect(self.reload)
        self._sort = QComboBox(); self._sort.addItems(
            ['sort: gmax', 'sort: margin', 'sort: feel', 'sort: endurance', 'sort: id'])
        self._sort.currentIndexChanged.connect(self.reload)
        self._pareto_only = QComboBox(); self._pareto_only.addItems(['Pareto archive', 'all candidates'])
        self._pareto_only.currentIndexChanged.connect(self.reload)
        launch = QPushButton('Launch city run…'); launch.clicked.connect(self._launch)
        for w_ in (pick, refresh, self._sort, self._pareto_only, launch):
            top.addWidget(w_)
        top.addWidget(self._dir_lbl); top.addStretch(1)
        v.addLayout(top)
        self._scroll = QScrollArea(); self._scroll.setWidgetResizable(True)
        v.addWidget(self._scroll)
        self._status = QLabel(''); self._status.setStyleSheet('color:#888;')
        v.addWidget(self._status)
        self.reload()

    def _latest_run(self):
        runs = sorted(glob.glob(os.path.join('designs_city', '*')), key=os.path.getmtime)
        return runs[-1] if runs else None

    def _pick(self):
        d = QFileDialog.getExistingDirectory(self, 'Pick a city run dir', 'designs_city')
        if d:
            self._run_dir = d; self._dir_lbl.setText(d); self.reload()

    def _launch(self):
        # non-blocking: spawn the engine as a subprocess; Refresh to see progress
        out = os.path.join('designs_city', 'run_%d' % (len(glob.glob('designs_city/*')) + 1))
        exe = sys.executable
        try:
            subprocess.Popen([exe, 'design_city.py', '--out', out,
                              '--islands', '4', '--pop', '12', '--gens', '4', '--workers', '4'],
                             cwd=os.getcwd())
            self._run_dir = out; self._dir_lbl.setText(out + '  (running — Refresh to update)')
        except Exception as e:
            QMessageBox.warning(self, 'Launch failed', str(e))

    def reload(self):
        inner = QWidget(); grid = QGridLayout(inner)
        cards = []
        if self._run_dir and os.path.isdir(self._run_dir):
            city = {}
            cj = os.path.join(self._run_dir, 'city.json')
            if os.path.exists(cj):
                try: city = json.load(open(cj))
                except Exception: city = {}
            keep = set(city.get('archive', [])) if (self._pareto_only.currentIndex() == 0
                                                    and city.get('archive')) else None
            for mj in glob.glob(os.path.join(self._run_dir, '*', 'metrics.json')):
                try: m = json.load(open(mj))
                except Exception: continue
                if not m.get('ok'): continue
                if keep is not None and m.get('id') not in keep: continue
                cards.append((os.path.dirname(mj), m))
        key = self._sort.currentText().split(': ')[1]
        if key in ('feel', 'endurance'):
            kk = 'endur' if key == 'endurance' else key
            cards.sort(key=lambda cm: cm[1].get('scores', {}).get(kk, -9), reverse=True)
        elif key in ('gmax', 'margin'):
            cards.sort(key=lambda cm: cm[1].get(key, -9), reverse=True)
        else:
            cards.sort(key=lambda cm: cm[1].get('id', ''))
        for i, (cdir, m) in enumerate(cards[:120]):
            grid.addWidget(_Card(cdir, m, self._open_detail), i // 3, i % 3)
        self._scroll.setWidget(inner)
        self._status.setText('%d designs shown from %s' % (len(cards[:120]), self._run_dir or '-'))

    def _open_detail(self, cdir, meta):
        _Detail(self, cdir, meta, self._mw).exec()
