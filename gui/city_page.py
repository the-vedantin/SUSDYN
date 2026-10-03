# -*- coding: utf-8 -*-
"""city_page.py — Design City gallery (Ctrl+3).

Shows one design_city.py run: per axle, the packaging solutions that hold
EVERY parameter within 0.1 % of the baseline, grouped by similarity
(complete-linkage clustering on the chassis-side hardpoint deltas).  One
section per group: the representative solution, the group's spread in mm,
which points moved and how far, then a card per solution with its native-GL
axle view.  Click a card for the full parameter table (baseline / value /
% deviation for all ~110 parameters), the three rendered views and
'Open in Vahan' to load that .vahan into the main window.

Reads the run layout design_city.py writes:
    <run>/run.json, <run>/<axle>/groups.json, <run>/<axle>/<id>/{config.vahan,
    metrics.json, gui_axle.png, gui_iso.png, gui_top.png}
Fonts and text colours follow docs/DESIGN.md (ink / muted / panel / line /
accent red; no yellow text).
"""
import os, json, glob, subprocess, sys
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QScrollArea, QGridLayout, QFrame, QDialog, QTableWidget,
                             QTableWidgetItem, QFileDialog, QComboBox, QSpinBox,
                             QMessageBox, QHeaderView, QSizePolicy)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

# docs/DESIGN.md dark tokens
INK, MUTED, PAPER, PANEL, LINE, ACCENT = '#ECECEE', '#9A9AA2', '#0B0B0D', '#141416', '#2A2A2E', '#E23B48'
FONT = "'Segoe UI', system-ui, -apple-system, Roboto, Arial, sans-serif"
MONO = "'Consolas', 'SF Mono', monospace"
CARD_W = 330
IMG_W = CARD_W - 16


def _eyebrow(text):
    l = QLabel(text.upper())
    l.setStyleSheet(f'color:{ACCENT};font-size:11px;font-weight:650;letter-spacing:0.15em;font-family:{FONT};')
    return l


def _muted(text, size=11):
    l = QLabel(text)
    l.setStyleSheet(f'color:{MUTED};font-size:{size}px;font-family:{FONT};')
    l.setWordWrap(True)
    return l


def _ink(text, size=13, weight=400):
    l = QLabel(text)
    l.setStyleSheet(f'color:{INK};font-size:{size}px;font-weight:{weight};font-family:{FONT};')
    l.setWordWrap(True)
    return l


def _fmt_mm(v):
    return f'{v:.1f} mm'


class _Card(QFrame):
    """One solution: axle view + what it is, in plain words."""

    def __init__(self, sdir, meta, on_click, representative=False):
        super().__init__()
        self._sdir = sdir; self._meta = meta; self._on_click = on_click
        self.setFrameShape(QFrame.Shape.StyledPanel)
        border = ACCENT if representative else LINE
        self.setStyleSheet(f'QFrame{{background:{PANEL};border:1px solid {border};border-radius:6px;}}'
                           f'QFrame:hover{{border:1px solid {INK};}}')
        self.setFixedWidth(CARD_W)
        lay = QVBoxLayout(self); lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(4)
        png = os.path.join(sdir, 'gui_axle.png')
        pic = QLabel(); pic.setStyleSheet('border:none;')
        if os.path.exists(png):
            pic.setPixmap(QPixmap(png).scaledToWidth(IMG_W, Qt.TransformationMode.SmoothTransformation))
        else:
            pic.setText('(render pending — run design_city.py --render <run>)')
            pic.setStyleSheet(f'color:{MUTED};font-size:11px;border:none;')
            pic.setMinimumHeight(120)
        lay.addWidget(pic)
        head = QHBoxLayout()
        title = _ink(('baseline (as designed)' if meta.get('is_baseline') else meta.get('id', '?')), 13, 600)
        title.setStyleSheet(title.styleSheet() + 'border:none;')
        head.addWidget(title)
        if representative:
            rep = _eyebrow('representative'); rep.setStyleSheet(rep.styleSheet() + 'border:none;')
            head.addStretch(1); head.addWidget(rep)
        lay.addLayout(head)
        dev = meta.get('max_deviation_pct', float('nan'))
        mv = meta.get('max_point_move_mm', 0.0)
        fam = ', '.join(meta.get('families', [])) or 'no move'
        sub = _muted(f'largest parameter deviation {dev:.4f} %   ·   largest point move {mv:.1f} mm\n'
                     f'what moved: {fam}')
        sub.setStyleSheet(sub.styleSheet() + 'border:none;')
        lay.addWidget(sub)

    def mousePressEvent(self, ev):
        self._on_click(self._sdir, self._meta)


class _Detail(QDialog):
    """Full report of one solution: three views, gates, every parameter."""

    def __init__(self, parent, sdir, meta, main_window):
        super().__init__(parent)
        self._mw = main_window; self._cfg = os.path.join(sdir, 'config.vahan')
        self.setWindowTitle(f"Design City — {meta.get('axle', '')} {meta.get('id', '?')}")
        self.resize(1100, 800)
        self.setStyleSheet(f'QDialog{{background:{PAPER};}} QLabel{{color:{INK};font-family:{FONT};}}')
        lay = QVBoxLayout(self)
        sc = QScrollArea(); sc.setWidgetResizable(True)
        inner = QWidget(); il = QVBoxLayout(inner)
        il.addWidget(_eyebrow(f"{meta.get('axle', '')} packaging solution"))
        il.addWidget(_ink(('baseline (as designed)' if meta.get('is_baseline') else meta.get('id', '?')), 18, 600))
        gates = meta.get('clash_locks', {}); au = meta.get('audit', {}); hw = meta.get('rocker_hw_gaps_mm', {})
        il.addWidget(_muted(
            f"largest parameter deviation {meta.get('max_deviation_pct', float('nan')):.4f} % "
            f"({meta.get('worst_parameter', '')})   ·   largest point move {meta.get('max_point_move_mm', 0.0):.1f} mm\n"
            f"clash sweep at centre and both locks: {gates.get('negatives', '?')} negatives"
            f"   ·   39-state audit: {au.get('negatives', '?')} negatives, {au.get('warnings', '?')} under 3 mm, "
            f"closest {au.get('min_gap_mm', float('nan')):.2f} mm   ·   rocker hardware closest "
            f"{min(hw.values()) if hw else float('nan'):.1f} mm   ·   reload gate {'pass' if meta.get('reload', {}).get('ok') else 'FAIL'}"
            f"   ·   wheel side byte-identical {meta.get('wheel_side_byte_identical')}", 12))
        # images
        row = QHBoxLayout()
        for name in ('gui_axle', 'gui_iso', 'gui_top'):
            png = os.path.join(sdir, name + '.png')
            if os.path.exists(png):
                lbl = QLabel(); lbl.setPixmap(QPixmap(png).scaledToWidth(340, Qt.TransformationMode.SmoothTransformation))
                row.addWidget(lbl)
        il.addLayout(row)
        # what moved
        deltas = meta.get('chassis_deltas_mm', {})
        moved = sorted(((k, (sum(x * x for x in v)) ** 0.5, v) for k, v in deltas.items()), key=lambda t: -t[1])
        moved = [m for m in moved if m[1] > 0.05]
        if moved:
            il.addWidget(_eyebrow('points moved (chassis side, mm from baseline)'))
            il.addWidget(_muted('\n'.join(f'{k.split(".")[-1]:20s} {d:6.1f} mm   (x {v[0]:+.1f}, y {v[1]:+.1f}, z {v[2]:+.1f})'
                                          for k, d, v in moved), 12))
        rec = meta.get('recipe', {})
        knobs = {k: v for k, v in rec.items() if v not in (0, 0.0, 1.0, False, {}, [0.0, 0.0, 0.0], None)}
        il.addWidget(_muted('recipe: ' + (json.dumps(knobs, default=float) if knobs else 'none (baseline)') +
                            f"   ·   retune: {json.dumps(meta.get('retune', {}), default=float)}", 11))
        # every parameter
        il.addWidget(_eyebrow('every parameter (0.1 % gate)'))
        rows = meta.get('rows', [])
        tbl = QTableWidget(len(rows), 7)
        tbl.setHorizontalHeaderLabels(['parameter', 'unit', 'baseline', 'this design', 'deviation %',
                                       'tolerance (abs)', 'floor active'])
        tbl.setStyleSheet(f'QTableWidget{{background:{PANEL};color:{INK};gridline-color:{LINE};font-family:{MONO};font-size:11px;}}'
                          f'QHeaderView::section{{background:{PANEL};color:{MUTED};border:none;border-bottom:1px solid {INK};'
                          f'font-size:11px;letter-spacing:0.06em;text-transform:uppercase;}}')
        for i, r in enumerate(sorted(rows, key=lambda r: -(r.get('deviation_pct') or 0.0))):
            vals = [r['name'], r['unit'], f"{r['baseline']:.5g}", f"{r['value']:.5g}",
                    f"{r['deviation_pct']:.4f}", f"{r['tolerance_abs']:.4g}", 'yes' if r.get('floor_active') else '']
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if j >= 2:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                tbl.setItem(i, j, it)
        tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        tbl.setMinimumHeight(420)
        il.addWidget(tbl)
        pth = _muted('config: ' + self._cfg, 10)
        pth.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        il.addWidget(pth)
        sc.setWidget(inner); lay.addWidget(sc)
        bt = QHBoxLayout()
        openb = QPushButton('Open in Vahan (load this design)')
        openb.setStyleSheet(f'background:{ACCENT};color:{INK};font-weight:600;padding:6px 12px;border-radius:4px;')
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
    """Design City — the 0.1 %-gated packaging solutions, grouped by similarity."""

    def __init__(self, main_window):
        super().__init__()
        self._mw = main_window
        self._run_dir = self._latest_run()
        self.setStyleSheet(f'QWidget{{background:{PAPER};}} QLabel{{color:{INK};font-family:{FONT};}}'
                           f'QPushButton{{color:{INK};background:{PANEL};border:1px solid {LINE};padding:4px 10px;border-radius:4px;}}'
                           f'QPushButton:hover{{border:1px solid {INK};}}'
                           f'QComboBox,QSpinBox{{color:{INK};background:{PANEL};border:1px solid {LINE};padding:2px 6px;}}')
        v = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(_eyebrow('Design City'))
        self._axle = QComboBox(); self._axle.addItems(['front', 'rear'])
        self._axle.currentIndexChanged.connect(self.reload)
        top.addWidget(_muted('axle', 12)); top.addWidget(self._axle)
        pick = QPushButton('Pick run…'); pick.clicked.connect(self._pick)
        refresh = QPushButton('Refresh'); refresh.clicked.connect(self.reload)
        self._trials = QSpinBox(); self._trials.setRange(10, 5000); self._trials.setValue(300)
        launch = QPushButton('Launch run (both axles)…'); launch.clicked.connect(self._launch)
        for w_ in (pick, refresh):
            top.addWidget(w_)
        top.addWidget(_muted('trials per axle', 12)); top.addWidget(self._trials); top.addWidget(launch)
        self._dir_lbl = _muted(self._run_dir or '(no runs found in designs_city/)', 11)
        top.addWidget(self._dir_lbl); top.addStretch(1)
        v.addLayout(top)
        self._summary = _muted('', 12)
        v.addWidget(self._summary)
        self._scroll = QScrollArea(); self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet(f'QScrollArea{{border:none;background:{PAPER};}}')
        v.addWidget(self._scroll)
        self._status = _muted('', 11)
        v.addWidget(self._status)
        self.reload()

    # ── run selection ────────────────────────────────────────────────────
    @staticmethod
    def _latest_run():
        runs = [d for d in glob.glob(os.path.join('designs_city', '*')) if os.path.exists(os.path.join(d, 'run.json'))]
        def _started(d):
            # the run's own timestamp, not the folder mtime (re-rendering an old run's images
            # touched its folder and made the page open v99 instead of v102, 2026-09-11)
            try:
                import json as _json
                with open(os.path.join(d, 'run.json'), 'r', encoding='utf-8') as fh:
                    return _json.load(fh).get('finished') or _json.load(fh).get('started') or ''
            except Exception:
                return ''
        runs.sort(key=lambda d: (_started(d), os.path.getmtime(os.path.join(d, 'run.json'))))
        return runs[-1] if runs else None

    def _pick(self):
        d = QFileDialog.getExistingDirectory(self, 'Pick a Design City run', 'designs_city')
        if d:
            self._run_dir = d; self._dir_lbl.setText(d); self.reload()

    def _launch(self):
        """Non-blocking: the engine runs as a subprocess (search offscreen,
        then native-GL renders in its own child); Refresh to see progress."""
        n = int(self._trials.value())
        try:
            env = dict(os.environ); env['QT_QPA_PLATFORM'] = 'offscreen'; env['VAHAN_MCP'] = '0'
            subprocess.Popen([sys.executable, 'design_city.py', '--trials', str(n), '--workers', '4'],
                             cwd=os.getcwd(), env=env)
            self._status.setText(f'launched design_city.py with {n} trials per axle — Refresh when run.json appears')
        except Exception as e:
            QMessageBox.warning(self, 'Launch failed', str(e))

    # ── content ──────────────────────────────────────────────────────────
    def _load_run(self):
        if not self._run_dir:
            return None
        rj = os.path.join(self._run_dir, 'run.json')
        if not os.path.exists(rj):
            return None
        try:
            with open(rj, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

    def reload(self):
        inner = QWidget(); inner.setStyleSheet(f'background:{PAPER};')
        col = QVBoxLayout(inner); col.setSpacing(14)
        run = self._load_run()
        axle = self._axle.currentText()
        n_cards = 0
        if run is None:
            col.addWidget(_ink('No Design City run selected.  Launch one (button above) or pick a '
                               'designs_city/<run> folder that contains run.json.', 13))
            self._summary.setText('')
        else:
            ax = run.get('axles', {}).get(axle, {})
            blk = ax.get('blocking_parameter', {}); sm = ax.get('smallest_max_deviation', {})
            self._summary.setText(
                f"config {run.get('config_name', '?')}  (sha {run.get('config_sha256', '')[:12]})   ·   "
                f"{run.get('n_parameters', '?')} parameters held within {run.get('rel_tol', 0.001) * 100:.1f} %   ·   "
                f"{axle}: {ax.get('trials', 0)} tried, {ax.get('kept', 0)} kept, {ax.get('groups', 0)} groups "
                f"(grouping cut {run.get('cluster_mm', '?')} mm)   ·   baseline passes its own gate: {ax.get('baseline_ok')}"
                + (f"   ·   most-failed parameter: {blk.get('name')} ({blk.get('fail_count')} trials)" if blk else '')
                + (f"   ·   smallest rejected deviation {sm.get('max_deviation_pct', 0):.4f} % ({sm.get('worst_parameter')})"
                   if sm and sm.get('stage') != 'kept' else ''))
            axle_dir = os.path.join(self._run_dir, axle)
            metas = {}
            for mj in glob.glob(os.path.join(axle_dir, '*', 'metrics.json')):
                try:
                    with open(mj, 'r', encoding='utf-8') as f:
                        m = json.load(f)
                except Exception:
                    continue
                if m.get('ok'):
                    metas[m['id']] = (os.path.dirname(mj), m)
            groups = []
            gj = os.path.join(axle_dir, 'groups.json')
            if os.path.exists(gj):
                try:
                    with open(gj, 'r', encoding='utf-8') as f:
                        groups = json.load(f).get('groups', [])
                except Exception:
                    groups = []
            if not groups and metas:
                groups = [{'group': 1, 'members': sorted(metas), 'representative': sorted(metas)[0],
                           'spread_mm': float('nan'), 'families': [], 'mean_point_shift_mm': {}}]
            ncols = max(1, (self._scroll.viewport().width() - 40) // (CARD_W + 12)) if self._scroll.viewport().width() > 0 else 3
            for g in groups:
                members = [m for m in g['members'] if m in metas]
                if not members:
                    continue
                sec = QFrame(); sec.setStyleSheet(f'QFrame{{border-top:1px solid {LINE};}}')
                sl = QVBoxLayout(sec); sl.setContentsMargins(0, 10, 0, 0)
                title = f"Group {g.get('group', '?')} — {len(members)} solution{'s' if len(members) != 1 else ''}"
                if g.get('contains_baseline'):
                    title += ' — contains the baseline'
                h = _ink(title, 15, 600); h.setStyleSheet(h.styleSheet() + 'border:none;')
                sl.addWidget(h)
                shifts = g.get('mean_point_shift_mm', {})
                top_shift = ', '.join(f"{k.split('.')[-1]} {v:.0f} mm" for k, v in list(shifts.items())[:5])
                info = _muted(f"representative {g.get('representative')}   ·   spread inside the group "
                              f"{g.get('spread_mm', float('nan')):.1f} mm (largest point-to-point difference between any two members)"
                              f"   ·   moves: {', '.join(g.get('families', [])) or 'none'}"
                              + (f"   ·   mean point shift from baseline: {top_shift}" if top_shift else ''), 11)
                info.setStyleSheet(info.styleSheet() + 'border:none;')
                sl.addWidget(info)
                grid = QGridLayout(); grid.setSpacing(10)
                order = [g.get('representative')] + [m for m in members if m != g.get('representative')]
                for i, mid in enumerate([m for m in order if m in metas]):
                    sdir, m = metas[mid]
                    grid.addWidget(_Card(sdir, m, self._open_detail, representative=(mid == g.get('representative'))),
                                   i // ncols, i % ncols, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
                    n_cards += 1
                grid.setColumnStretch(ncols, 1)
                sl.addLayout(grid)
                col.addWidget(sec)
            if not metas:
                col.addWidget(_ink(f'No {axle} solution passed the 0.1 % gate in this run.'
                                   + (f"  Most-failed parameter: {blk.get('name')}." if blk else ''), 13))
        col.addStretch(1)
        self._scroll.setWidget(inner)
        self._status.setText(f'{n_cards} {axle} solutions shown from {self._run_dir or "-"}')

    def _open_detail(self, sdir, meta):
        _Detail(self, sdir, meta, self._mw).exec()
