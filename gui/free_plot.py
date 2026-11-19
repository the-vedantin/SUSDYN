"""Plot anything against anything (user 2026-11-10: "the plotting flexibility is lacking ...
wheel rate is locked under roll only.  I need a list of every measurable variable and
measured-against variable so I can view anything against anything, under the 3D model, with
adjustable units").

FreePlotPanel: pick a data source (the geometry sweep, the steady-state dynamics sweep or a lap),
an X channel, any number of Y channels and the unit of every quantity; the graph is drawn in the
panel (one axes per Y, or everything on one) and can be opened big or exported as CSV.  Every
channel comes from vahan.channels — the ONE model's own outputs, nothing re-modelled here.

  kin  runs its own kinematic sweep for the chosen motion / range through
       MainWindow._snapshot_sweep_job + _compute_sweep (the same code the preset graphs use) in
       a background thread; the cache is dropped whenever the model changes
  dyn  reads the last Dynamics-panel sweep (MainWindow._dyn_sweep_data) with the handling
       extras attached; "Run sweep" presses the panel's own button
  lap  reads the Lap Time page's last result
"""
from __future__ import annotations

import csv
import math

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QListWidget,
                             QListWidgetItem, QLineEdit, QCheckBox, QPushButton, QDoubleSpinBox,
                             QSplitter, QFileDialog, QDialog, QSizePolicy)
from matplotlib.figure import Figure

from vahan import channels as CH

_CC = {'FL': '#FFD600', 'FR': '#E53935', 'RL': '#FFFFFF', 'RR': '#42A5F5', 'F': '#FFD600', 'R': '#42A5F5'}
_LS = {'FL': '-', 'FR': '--', 'RL': '-.', 'RR': ':', 'F': '-', 'R': '--'}
_CYCLE = [('#FFD600', '-'), ('#42A5F5', '-'), ('#FFFFFF', '-'), ('#E53935', '-'),
          ('#FFD600', '--'), ('#42A5F5', '--'), ('#FFFFFF', '--'), ('#E53935', '--'),
          ('#FFD600', ':'), ('#42A5F5', ':'), ('#FFFFFF', ':'), ('#E53935', ':')]
SOURCES = [('kin', 'Geometry sweep (kinematics)'),
           ('dyn', 'Steady-state dynamics sweep'),
           ('lap', 'Lap simulation')]
_DEFAULT_Y = {'kin': {'camber', 'toe'}, 'dyn': {'Fz'}, 'lap': {'lap:v'}}
_BTN = ('QPushButton { background:#15151a; color:#FFD600; padding:2px 8px; border:1px solid #FFD600;'
        ' border-radius:4px; font-size:11px; } QPushButton:hover { background:#2a2410; }')


class FreePlotPanel(QWidget):
    def __init__(self, main, sources=('kin', 'dyn', 'lap'), parent=None):
        super().__init__(parent)
        self._main = main
        self._sources = [s for s in SOURCES if s[0] in sources]
        self._channels: list = []
        self._checked = {k: set(v) for k, v in _DEFAULT_Y.items()}
        self._units: dict = {}           # family -> chosen unit
        self._kin_cache: dict = {}       # (motion, lo, hi) -> data dict
        self._kin_worker = None
        self._kin_pending = None
        self._last_rows: list = []
        self._unit_combos: dict = {}
        self._build()
        main._free_plot_panels = getattr(main, '_free_plot_panels', [])
        main._free_plot_panels.append(self)
        self._rebuild_channels()

    # ── UI ─────────────────────────────────────────────────────────────────
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(3)

        top = QHBoxLayout(); top.setSpacing(6)
        top.addWidget(QLabel('Data:'))
        self._source = QComboBox()
        for key, label in self._sources:
            self._source.addItem(label, key)
        self._source.currentIndexChanged.connect(self._on_source_changed)
        top.addWidget(self._source)
        self._sweep_lbl = QLabel('Sweep:')
        top.addWidget(self._sweep_lbl)
        self._motion = QComboBox()
        for m, label, _f, _r in CH.KIN_MOTIONS:
            self._motion.addItem(m, m)
        self._motion.setToolTip('What the geometry sweep moves.  X and Y can be anything the sweep '
                                'produces; this is the input that is varied.')
        self._motion.currentIndexChanged.connect(self._on_motion_changed)
        top.addWidget(self._motion)
        self._lo = QDoubleSpinBox(); self._hi = QDoubleSpinBox()
        for sb in (self._lo, self._hi):
            sb.setRange(-1000.0, 1000.0); sb.setDecimals(1); sb.setSingleStep(5.0)
            sb.setFixedWidth(74)
            sb.editingFinished.connect(self._schedule)
        self._from_lbl = QLabel('from'); self._to_lbl = QLabel('to')
        top.addWidget(self._from_lbl); top.addWidget(self._lo); top.addWidget(self._to_lbl); top.addWidget(self._hi)
        self._run_btn = QPushButton('Run sweep')
        self._run_btn.setStyleSheet(_BTN)
        self._run_btn.setToolTip('Runs the Dynamics panel sweep with its current settings (right sidebar).')
        self._run_btn.clicked.connect(self._run_dyn_sweep)
        top.addWidget(self._run_btn)
        self._hint = QLabel('')
        self._hint.setStyleSheet('color:#8a8a92; font-size:10px;')
        top.addWidget(self._hint, 1)
        root.addLayout(top)

        row2 = QHBoxLayout(); row2.setSpacing(6)
        row2.addWidget(QLabel('X:'))
        self._x = QComboBox()
        self._x.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._x.currentIndexChanged.connect(self._on_x_changed)
        row2.addWidget(self._x)
        row2.addWidget(QLabel('Layout:'))
        self._layout = QComboBox()
        self._layout.addItem('one graph per variable', 'grid')
        self._layout.addItem('everything on one graph', 'overlay')
        self._layout.currentIndexChanged.connect(self._schedule)
        row2.addWidget(self._layout)
        row2.addWidget(QLabel('Corners:'))
        self._corner_cbs = {}
        for c in CH.CORNERS:
            cb = QCheckBox(c); cb.setChecked(True); cb.toggled.connect(self._schedule)
            self._corner_cbs[c] = cb; row2.addWidget(cb)
        row2.addStretch(1)
        self._csv_btn = QPushButton('CSV'); self._csv_btn.setStyleSheet(_BTN)
        self._csv_btn.setToolTip('Save every curve on the graph (in the units shown) as a CSV file.')
        self._csv_btn.clicked.connect(self._export_csv)
        row2.addWidget(self._csv_btn)
        self._big_btn = QPushButton('Open big ↗'); self._big_btn.setStyleSheet(_BTN)
        self._big_btn.clicked.connect(self._open_big)
        row2.addWidget(self._big_btn)
        root.addLayout(row2)

        self._units_row = QHBoxLayout(); self._units_row.setSpacing(6)
        self._units_row.addWidget(QLabel('Units:'))
        self._units_row.addStretch(1)
        root.addLayout(self._units_row)

        body = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget(); lv = QVBoxLayout(left); lv.setContentsMargins(0, 0, 0, 0); lv.setSpacing(2)
        self._search = QLineEdit(); self._search.setPlaceholderText('find a variable…')
        self._search.textChanged.connect(self._filter_list)
        lv.addWidget(self._search)
        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.itemChanged.connect(self._on_item_changed)
        lv.addWidget(self._list, 1)
        bl = QHBoxLayout()
        for label, fn in (('None', self._none),):
            b = QPushButton(label); b.setFixedWidth(50); b.clicked.connect(fn); bl.addWidget(b)
        bl.addStretch(1)
        lv.addLayout(bl)
        left.setMinimumWidth(200)
        body.addWidget(left)
        from gui.plot_dialog import ReadableCanvas
        self._fig = Figure(facecolor='#000000')
        self._canvas = ReadableCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        body.addWidget(self._canvas)
        body.setStretchFactor(0, 0); body.setStretchFactor(1, 1)
        body.setSizes([240, 900])
        root.addWidget(body, 1)

        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.setInterval(120)
        self._timer.timeout.connect(self._draw)
        self._on_source_changed()

    # ── state ──────────────────────────────────────────────────────────────
    @property
    def source(self) -> str:
        return self._source.currentData() or 'kin'

    def set_source(self, key: str):
        i = self._source.findData(key)
        if i >= 0:
            self._source.setCurrentIndex(i)

    def corners(self) -> list:
        return [c for c, cb in self._corner_cbs.items() if cb.isChecked()]

    def x_channel(self):
        key = self._x.currentData()
        for ch in self._channels:
            if ch.key == key:
                return ch
        return None

    def set_x(self, key: str):
        i = self._x.findData(key)
        if i >= 0:
            self._x.setCurrentIndex(i)

    def selected_y(self) -> list:
        keys = self._checked.get(self.source, set())
        return [ch for ch in self._channels if ch.key in keys]

    def set_y(self, keys):
        self._checked[self.source] = set(keys)
        self._fill_list()
        self._rebuild_units()
        self._schedule()

    def unit_for(self, family: str) -> str:
        return self._units.get(family, CH.base_unit(family))

    def set_unit(self, family: str, unit: str):
        if unit in CH.units_for(family):
            self._units[family] = unit
            cb = self._unit_combos.get(family)
            if cb is not None:
                i = cb.findText(unit)
                if i >= 0:
                    cb.blockSignals(True); cb.setCurrentIndex(i); cb.blockSignals(False)
            self._schedule()

    def kin_motion(self) -> str:
        return self._motion.currentData() or 'heave'

    def kin_range(self) -> tuple:
        return float(self._lo.value()), float(self._hi.value())

    # ── channel lists ──────────────────────────────────────────────────────
    def _rebuild_channels(self):
        src = self.source
        main = self._main
        if src == 'kin':
            gp = getattr(main, '_graph_panel', None)
            flav = set(getattr(gp, '_topology_flavors', None) or {'standard'})
            self._kin_flav = flav
            self._channels = CH.kin_channels(flav)
        elif src == 'dyn':
            self._channels = CH.dyn_channels(self._dyn_sweep(attach=True))
        else:
            lp = getattr(main, '_laptime_page', None)
            self._channels = CH.lap_channels(getattr(lp, '_result', None))
        # X combo
        cur = self._x.currentData()
        self._x.blockSignals(True)
        self._x.clear()
        for ch in self._channels:
            self._x.addItem(f'{ch.label}  [{ch.unit}]', ch.key)
        default = cur if any(ch.key == cur for ch in self._channels) else self._default_x()
        i = self._x.findData(default)
        self._x.setCurrentIndex(max(i, 0))
        self._x.blockSignals(False)
        self._fill_list()
        self._rebuild_units()

    def _default_x(self) -> str:
        src = self.source
        if src == 'kin':
            return f'motion:{self.kin_motion()}'
        if src == 'dyn':
            sw = self._dyn_sweep(attach=False) or {}
            for k in ('lateral_g', 'longitudinal_g', 'speed_kph'):
                if k in sw:
                    return k
            return 'lateral_g'
        return 'lap:s'

    def _fill_list(self):
        keys = self._checked.get(self.source, set())
        self._list.blockSignals(True)
        self._list.clear()
        prev = None
        for ch in self._channels:
            if ch.category != prev:
                hdr = QListWidgetItem(f'  {ch.category}')
                hdr.setFlags(Qt.ItemFlag.NoItemFlags)
                hdr.setForeground(QColor('#aaaaaa'))
                f = hdr.font(); f.setBold(True); hdr.setFont(f)
                self._list.addItem(hdr)
                prev = ch.category
            it = QListWidgetItem(f'{ch.label}  [{ch.unit}]')
            it.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            it.setCheckState(Qt.CheckState.Checked if ch.key in keys else Qt.CheckState.Unchecked)
            it.setData(Qt.ItemDataRole.UserRole, ch.key)
            if ch.note:
                it.setToolTip(ch.note)
            self._list.addItem(it)
        self._list.blockSignals(False)
        self._filter_list(self._search.text())

    def _filter_list(self, text=''):
        t = (text or '').strip().lower()
        for i in range(self._list.count()):
            it = self._list.item(i)
            if it.data(Qt.ItemDataRole.UserRole) is None:
                it.setHidden(False)
                continue
            it.setHidden(bool(t) and t not in it.text().lower())

    def _on_item_changed(self, item):
        key = item.data(Qt.ItemDataRole.UserRole)
        if key is None:
            return
        s = self._checked.setdefault(self.source, set())
        if item.checkState() == Qt.CheckState.Checked:
            s.add(key)
        else:
            s.discard(key)
        self._rebuild_units()
        self._schedule()

    def _none(self):
        self._checked[self.source] = set()
        self._fill_list(); self._rebuild_units(); self._schedule()

    def _rebuild_units(self):
        """One unit combo per quantity family on the graph (X and every chosen Y)."""
        fams = []
        xch = self.x_channel()
        if xch is not None:
            fams.append(xch.family)
        for ch in self.selected_y():
            if ch.family not in fams:
                fams.append(ch.family)
        # clear old combos (keep the leading label and trailing stretch)
        while self._units_row.count() > 2:
            it = self._units_row.takeAt(1)
            w = it.widget()
            if w is not None:
                # hide + unparent NOW: a widget waiting for deleteLater still paints
                # at its default 640x480 over the panel until the event loop runs
                w.hide(); w.setParent(None); w.deleteLater()
        self._unit_combos = {}
        pos = 1
        for fam in fams:
            lbl = QLabel(fam.replace('_', ' '))
            lbl.setStyleSheet('color:#8a8a92; font-size:10px;')
            cb = QComboBox()
            for u in CH.units_for(fam):
                cb.addItem(u)
            cb.setCurrentText(self.unit_for(fam))
            cb.currentTextChanged.connect(lambda u, fam=fam: self._on_unit(fam, u))
            self._units_row.insertWidget(pos, lbl); self._units_row.insertWidget(pos + 1, cb)
            pos += 2
            self._unit_combos[fam] = cb

    def _on_unit(self, fam, unit):
        self._units[fam] = unit
        self._schedule()

    # ── events ─────────────────────────────────────────────────────────────
    def _on_source_changed(self, *_):
        src = self.source
        kin = src == 'kin'
        for w in (self._sweep_lbl, self._motion, self._from_lbl, self._lo, self._to_lbl, self._hi):
            w.setVisible(kin)
        self._run_btn.setVisible(src == 'dyn')
        if kin:
            self._seed_range()
        self._hint.setText({'kin': 'X and Y: any geometry output of the swept motion; the sweep itself is the first X.',
                            'dyn': 'The last sweep from the Dynamics panel (right sidebar sets the range and the g / speed axis).',
                            'lap': 'The last lap from the Lap Time page.'}[src])
        self._rebuild_channels()
        self._schedule()

    def _on_motion_changed(self, *_):
        self._seed_range()
        self._rebuild_channels()
        self.set_x(f'motion:{self.kin_motion()}')
        self._schedule()

    def _seed_range(self):
        m = self.kin_motion()
        lo, hi = CH.KIN_MOTION_DEFAULTS.get(m, (-50.0, 50.0))
        mp = getattr(self._main, '_motion_panel', None)
        try:
            if mp is not None and mp.motion == m:
                lo, hi = float(mp.min_val), float(mp.max_val)
        except Exception:
            pass
        unit = '°' if m in ('roll', 'steer') else ' mm'
        for sb in (self._lo, self._hi):
            sb.blockSignals(True); sb.setSuffix(unit)
        self._lo.setValue(lo); self._hi.setValue(hi)
        for sb in (self._lo, self._hi):
            sb.blockSignals(False)

    def _on_x_changed(self, *_):
        self._rebuild_units()
        self._schedule()

    def _schedule(self, *_):
        self._timer.start()

    def showEvent(self, ev):
        super().showEvent(ev)
        self._schedule()

    def on_source_data(self, source: str):
        """MainWindow._notify_free_plot: new data for `source` (kin = the model changed)."""
        if source == 'kin':
            self._kin_cache.clear()
        if self.source == source:
            if source != 'kin':
                self._rebuild_channels()
            else:
                # a different topology (project load) changes which geometry channels exist
                gp = getattr(self._main, '_graph_panel', None)
                flav = set(getattr(gp, '_topology_flavors', None) or {'standard'})
                if flav != getattr(self, '_kin_flav', None):
                    self._rebuild_channels()
            if self.isVisible():
                self._schedule()

    # ── data ───────────────────────────────────────────────────────────────
    def _dyn_sweep(self, attach=True):
        sw = getattr(self._main, '_dyn_sweep_data', None)
        if not isinstance(sw, dict):
            return None
        if attach:
            try:
                self._main._attach_wheel_rate(sw, ['wheel_rate'])
                self._main._attach_dyn_extras(sw, sorted(self._main._DYN_EXTRA_GRAPHS))
            except Exception:
                pass
        return sw

    def _run_dyn_sweep(self):
        dp = getattr(self._main, '_dynamics_panel', None)
        if dp is not None and hasattr(dp, '_on_sweep'):
            dp._on_sweep()

    def _lap_data(self):
        lp = getattr(self._main, '_laptime_page', None)
        res = getattr(lp, '_result', None)
        if res is None:
            return None
        return {'res': res, 'track': getattr(lp, '_track', None)}

    def _kin_key(self):
        lo, hi = self.kin_range()
        return (self.kin_motion(), round(lo, 3), round(hi, 3))

    def _kin_data(self):
        """Cached sweep for the chosen motion / range, or None (and a background sweep started)."""
        key = self._kin_key()
        d = self._kin_cache.get(key)
        if d is not None:
            return d
        self._start_kin_sweep(key)
        return None

    def _kin_extras(self) -> dict:
        """The dynamics car's rates for the derived kin channels (wheel rate, ride rate, ...)."""
        ex = {}
        try:
            veh = self._main._build_dynamics_solver()._veh
            ex['spring_rate_Nmm'] = {'F': float(veh.spring_rate_front_Npm) / 1000.0,
                                     'R': float(veh.spring_rate_rear_Npm) / 1000.0}
            ex['tire_rate_Nmm'] = float(veh.tire_rate_Npm) / 1000.0
            ex['track_m'] = {'F': float(veh.front_track_m), 'R': float(veh.rear_track_m)}
            from vahan.ride_solve import sprung_corner_masses_kg
            mf, mr = sprung_corner_masses_kg(veh)
            ex['corner_mass_kg'] = {'F': float(mf), 'R': float(mr)}
        except Exception:
            pass
        return ex

    def compute_kin_sync(self, motion=None, lo=None, hi=None) -> dict:
        """Synchronous sweep (regression net / programmatic): snapshot -> _compute_sweep on the
        calling thread, cached under the panel's key."""
        m = motion or self.kin_motion()
        if lo is None or hi is None:
            lo, hi = self.kin_range()
        job = self._main._snapshot_sweep_job(motion=m, lo=float(lo), hi=float(hi), bump_gen=False)
        res = self._main._compute_sweep(job)
        d = dict(res); d.update(self._kin_extras())
        self._kin_cache[(m, round(float(lo), 3), round(float(hi), 3))] = d
        return d

    def _start_kin_sweep(self, key):
        if self._kin_worker is not None and self._kin_worker.isRunning():
            self._kin_pending = key
            return
        try:
            from gui.main_window import _KinSweepWorker
            m, lo, hi = key
            job = self._main._snapshot_sweep_job(motion=m, lo=lo, hi=hi, bump_gen=False)
            ex = self._kin_extras()
            w = _KinSweepWorker(self._main, job)

            def _done(res, key=key, ex=ex):
                self._kin_worker = None
                d = dict(res); d.update(ex)
                self._kin_cache[key] = d
                self._after_kin()

            def _fail(msg):
                self._kin_worker = None
                self._hint.setText(f'sweep failed: {msg}')
                self._after_kin()

            w.done.connect(_done); w.failed.connect(_fail)
            self._kin_worker = w
            w.start()
        except Exception as e:
            self._kin_worker = None
            self._hint.setText(f'sweep: {e}')

    def _after_kin(self):
        p = self._kin_pending
        self._kin_pending = None
        if p is not None and p not in self._kin_cache:
            self._start_kin_sweep(p)
        self._schedule()

    def data_or_message(self):
        src = self.source
        if src == 'kin':
            d = self._kin_data()
            return d, ('sweeping…' if d is None else '')
        if src == 'dyn':
            sw = self._dyn_sweep(attach=True)
            if sw is None:
                return None, 'No dynamics sweep yet — press "Run sweep" (or the Sweep button in the Dynamics panel on the right).'
            if any(ch.key.startswith('steer:') for ch in self.selected_y()) and 'steering_effort' not in sw:
                try:
                    self._main._attach_steering_effort(sw, ['steering_torque'])
                except Exception:
                    pass
            return sw, ''
        d = self._lap_data()
        return d, ('' if d is not None else 'No lap yet — run a lap on the Lap Time page.')

    # ── drawing ────────────────────────────────────────────────────────────
    def rows(self, data=None) -> list:
        """[(y channel, series name, x shown, y shown, x unit, y unit)] for the current picks."""
        if data is None:
            data, _ = self.data_or_message()
        xch = self.x_channel()
        if data is None or xch is None:
            return []
        ux = self.unit_for(xch.family)
        xs = xch.series(data)
        out = []
        for ych in self.selected_y():
            uy = self.unit_for(ych.family)
            for name, x, y in CH.pair_series(xs, ych.series(data), self.corners()):
                out.append((ych, name, CH.convert(x, xch.family, ux), CH.convert(y, ych.family, uy), ux, uy))
        return out

    def _draw(self):
        self.draw_into(self._fig, big=False)
        self._canvas.draw_idle()

    def draw_into(self, fig, big=False) -> list:
        fig.clf()
        fig.set_facecolor('#000000')
        xch = self.x_channel()
        data, msg = self.data_or_message()
        rows = self.rows(data) if data is not None else []
        self._last_rows = rows
        if not rows:
            txt = msg or ('Pick an X and tick at least one variable on the left.' if not self.selected_y()
                          else 'Nothing to draw for this pick (the data does not carry it, or no corner is ticked).')
            fig.text(0.5, 0.5, txt, ha='center', va='center', color='#9a9aa2', fontsize=10, wrap=True)
            return []
        ux = self.unit_for(xch.family)
        xlabel = f'{xch.label} ({ux})'
        axes = []
        if self._layout.currentData() == 'overlay':
            ax = fig.add_subplot(1, 1, 1)
            self._style(ax)
            units = []
            for i, (ych, name, x, y, _ux, uy) in enumerate(rows):
                col, ls = self._style_for(name, i)
                lab = ych.label + (f' {name}' if name else '')
                ax.plot(x, y, color=col, ls=ls, lw=1.5, label=lab)
                if uy not in units:
                    units.append(uy)
            ax.set_xlabel(xlabel, color='#888888', fontsize=9 if big else 8)
            ax.set_ylabel(', '.join(units), color='#888888', fontsize=9 if big else 8)
            self._zero_lines(ax, x)
            ax.legend(fontsize=8 if big else 7, facecolor='#101014', edgecolor='#3a3a40', labelcolor='#e0e0e4', framealpha=0.85)
            axes.append(ax)
        else:
            ys = [ych for ych in self.selected_y() if any(r[0] is ych for r in rows)]
            n = len(ys)
            cols = min(n, 2 if big else 3)
            nrows = (n + cols - 1) // cols
            fig.subplots_adjust(hspace=0.62, wspace=0.34, left=0.08, right=0.98, top=0.92, bottom=0.14)
            for idx, ych in enumerate(ys):
                ax = fig.add_subplot(nrows, cols, idx + 1)
                self._style(ax)
                uy = None
                k = 0
                for (c, name, x, y, _ux, uy_) in rows:
                    if c is not ych:
                        continue
                    uy = uy_
                    col, ls = self._style_for(name, k); k += 1
                    ax.plot(x, y, color=col, ls=ls, lw=1.5, label=name or ych.label)
                import textwrap
                ax.set_ylabel(textwrap.fill(ych.label, 34 if big else 26) + f'\n({uy})',
                              color='#888888', fontsize=9 if big else 7.5, labelpad=2)
                ax.set_xlabel(xlabel, color='#888888', fontsize=9 if big else 8, labelpad=2)
                self._zero_lines(ax, rows[0][2])
                if k > 1:
                    ax.legend(fontsize=8 if big else 6.5, facecolor='#101014', edgecolor='#3a3a40',
                              labelcolor='#e0e0e4', framealpha=0.85, ncol=2 if k > 2 else 1)
                axes.append(ax)
        return axes

    @staticmethod
    def _style(ax):
        ax.set_facecolor('#080808')
        ax.tick_params(colors='#777777', labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#222222')
        ax.grid(True, color='#1a1a1a', lw=0.5)

    @staticmethod
    def _zero_lines(ax, x):
        try:
            x = np.asarray(x, float)
            if np.nanmin(x) < 0.0 < np.nanmax(x):
                ax.axvline(0, color='#333333', lw=0.8, ls=':')
        except Exception:
            pass

    @staticmethod
    def _style_for(name, i):
        if name in _CC:
            return _CC[name], _LS[name]
        return _CYCLE[i % len(_CYCLE)]

    # ── extras ─────────────────────────────────────────────────────────────
    def _open_big(self):
        from gui.plot_dialog import PlotDialog
        xch = self.x_channel()
        title = f'Plot anything — {xch.label if xch else ""} vs ' + ', '.join(c.label for c in self.selected_y())
        dlg = PlotDialog(self, Figure(facecolor='#000000'), title=title[:120])
        self.draw_into(dlg.canvas.figure, big=True)
        dlg.canvas.draw()
        dlg.show()
        self._big = dlg

    def csv_rows(self) -> list:
        """Header + rows of the plotted curves, in the units shown (column pairs x, y per curve)."""
        rows = self._last_rows or self.rows()
        if not rows:
            return []
        xch = self.x_channel()
        header = []
        cols = []
        for ych, name, x, y, ux, uy in rows:
            tag = f' {name}' if name else ''
            header += [f'{xch.label} ({ux}){tag}', f'{ych.label} ({uy}){tag}']
            cols += [x, y]
        n = max(len(c) for c in cols)
        out = [header]
        for i in range(n):
            out.append([('' if i >= len(c) or not np.isfinite(c[i]) else f'{c[i]:.6g}') for c in cols])
        return out

    def _export_csv(self):
        rows = self.csv_rows()
        if not rows:
            self._hint.setText('nothing on the graph to export')
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save the plotted curves', 'plot_anything.csv', 'CSV (*.csv)')
        if not path:
            return
        with open(path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows(rows)
        self._hint.setText(f'saved {path}')


class FreePlotWindow(QDialog):
    def __init__(self, main, source='lap', parent=None):
        super().__init__(parent)
        self.setWindowTitle('Plot anything (X vs Y)')
        self.resize(1150, 680)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.panel = FreePlotPanel(main, sources=('kin', 'dyn', 'lap'))
        self.panel.set_source(source)
        lay.addWidget(self.panel)
        self._main = main

    def closeEvent(self, ev):
        try:
            self._main._free_plot_panels.remove(self.panel)
        except Exception:
            pass
        super().closeEvent(ev)


def open_free_plot_window(main, source='lap', parent=None):
    w = FreePlotWindow(main, source=source, parent=parent)
    main._free_plot_windows = getattr(main, '_free_plot_windows', [])
    main._free_plot_windows.append(w)
    w.show()
    return w
