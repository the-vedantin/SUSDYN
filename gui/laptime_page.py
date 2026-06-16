"""
gui/laptime_page.py — the LAPTIME page.

Layout contract (user-specified):
  * the TRACK MAP is always visible at the top (proportionate, never scrolled)
  * every other graph is a CHECKBOX — only checked graphs render, splitting
    the remaining canvas height between them (fewer checked = bigger graphs)
  * the user never scrolls to see a graph

Sim: vahan.laptime.LapSimulator on the LIVE MainWindow car (single-model),
with aero (Cl·A / Cd·A / ρ / CoP), grip derate, and a full 6-speed GEARBOX
(primary × gear × final, torque-plateau/constant-power engine, redline).
"""
from __future__ import annotations

import os

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QDoubleSpinBox, QSpinBox, QCheckBox, QScrollArea, QFrame, QComboBox,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Selectable tracks: (label, json filename).  The page lists every file that
# exists; missing ones are skipped.
_TRACKS = [
    ('2024', 'autocross26.json'),
    ('2025', 'autocross_2025.json'),
]

# The tire-dataset dropdown is built DYNAMICALLY from whatever .mat files the
# user has placed in tire_data/ (which is gitignored — TTC data is never in
# the repo).  Each entry's label is read from the file's own tire_id AT
# RUNTIME, so no TTC tire names / run numbers are hard-coded in this (public)
# source file.  See LaptimePage._discover_tires().

from vahan.laptime import Track, LapSimulator

# colour-blind-safe corner colours (matches the suspension page)
_CC = {'FL': '#FFD600', 'FR': '#E53935', 'RL': '#FFFFFF', 'RR': '#42A5F5'}
_LS = {'FL': '-', 'FR': '--', 'RL': '-.', 'RR': ':'}

# user-supplied 2026 gearbox (tooth counts verified: 33/12, 32/16, 30/18,
# 26/18, 30/23, 29/24)
_DEFAULT_GEARS = [2.750, 2.000, 1.667, 1.444, 1.304, 1.208]


class _SimWorker(QThread):
    progress = pyqtSignal(str, int)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, sim: LapSimulator, track: Track, n_detail: int):
        super().__init__()
        self._sim, self._track, self._nd = sim, track, n_detail

    def run(self):
        try:
            res = self._sim.simulate(
                self._track, n_detail=self._nd,
                progress_cb=lambda m, p: self.progress.emit(m, p))
            self.finished_ok.emit(res)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.failed.emit(str(e))


class LaptimePage(QWidget):
    """The Laptime page widget.  `main` is the MainWindow (for the live car)."""

    # (key, checkbox label, default-on) — each graph is single-purpose and
    # isolatable (no dual-axis mashups except rpm+gear, which reads as one).
    _GRAPHS = [
        ('speed',  'Speed',            True),
        ('gg',     'Lat / Lon g',      True),
        ('util',   'Tire utilization', True),
        ('lltd',   'LLTD',             False),
        ('roll',   'Roll',             False),
        ('susp',   'Travel + shock',   False),
        ('aero',   'Aero down/drag',   False),
        ('rpm',    'RPM + gear',       False),
        ('power',  'Power',            False),
    ]

    def __init__(self, main):
        super().__init__()
        self._main = main
        self._track: Track | None = None
        self._worker: _SimWorker | None = None
        self._result = None
        self._last_lap_s: float | None = None    # for the Δ-vs-last readout
        self._build()
        self._load_default_track()
        self._redraw()               # placeholder until first sim
        # Seed CdA / air density from the live car (single source: the
        # dynamics panel owns them; these are sim-local overrides).
        try:
            veh = main._build_dynamics_solver()._veh
            self._cda.setValue(float(veh.cda_m2))
            self._rho.setValue(float(veh.air_density_kg_m3))
        except Exception:
            pass
        self._refresh_tire_active()

    # ── UI ───────────────────────────────────────────────────────────────
    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        # ── left sidebar (scrollable — the GRAPHS never scroll, this may) ─
        side = QVBoxLayout()
        side.setSpacing(6)

        title = QLabel('LAP TIME SIM')
        title.setStyleSheet('color:#FFD600; font-size:15px; font-weight:bold;')
        side.addWidget(title)

        self._track_lbl = QLabel('— no track —')
        self._track_lbl.setStyleSheet('color:#aaa; font-size:11px;')
        self._track_lbl.setWordWrap(True)
        side.addWidget(self._track_lbl)

        def hdr(text):
            l = QLabel(text)
            l.setStyleSheet('color:#FFD600; font-size:11px; font-weight:bold;'
                            'padding-top:6px;')
            side.addWidget(l)

        def grid():
            g = QGridLayout(); g.setSpacing(4)
            side.addLayout(g)
            return g

        def spin(g, row, col, label, lo, hi, val, dec, step, suffix, tip=''):
            g.addWidget(QLabel(label), row, col * 2)
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi); sb.setDecimals(dec)
            sb.setSingleStep(step); sb.setValue(val); sb.setSuffix(suffix)
            if tip:
                sb.setToolTip(tip)
            g.addWidget(sb, row, col * 2 + 1)
            return sb

        # ── TIRE dataset selector — populated from local tire_data/ files ─
        self._tire_rounds = self._discover_tires()
        if self._tire_rounds:
            hdr('TIRE  (local dataset)')
            g = grid()
            g.addWidget(QLabel('Tire:'), 0, 0)
            self._tire_combo = QComboBox()
            self._tire_combo.setStyleSheet('QComboBox { font-size:11px; }')
            # index 0 = neutral placeholder (whatever's currently loaded), so
            # the dropdown never claims a tire it hasn't loaded.
            self._tire_combo.addItem('(loaded tire — pick to swap)')
            for lbl, _fn in self._tire_rounds:
                self._tire_combo.addItem(lbl)
            self._tire_combo.setToolTip(
                'Swap the loaded tire data.  The list is built from the .mat '
                'files in your local tire_data/ folder; labels come from each '
                "file's own tire id.  Re-Sim to compare lap times.")
            # 'activated' fires on every USER pick (even the same item) so a
            # tire always loads when chosen — currentIndexChanged would skip a
            # re-pick of the current index.
            self._tire_combo.activated.connect(self._on_tire_round)
            g.addWidget(self._tire_combo, 0, 1)
            self._tire_active = QLabel('')
            self._tire_active.setStyleSheet('color:#8a8a92; font-size:10px;')
            self._tire_active.setWordWrap(True)
            side.addWidget(self._tire_active)

        hdr('AERO + GRIP')
        g = grid()
        self._cla = spin(g, 0, 0, 'Cl·A (+down):', -5.0, 10.0, 0.0, 2, 0.1,
                         ' m²',
                         'SIGN CONVENTION: positive = DOWNFORCE, negative = '
                         'net LIFT (bad bodywork).  0 = no aero package.')
        self._cda = spin(g, 1, 0, 'Cd·A:', 0.0, 5.0, 1.0, 2, 0.05, ' m²')
        self._rho = spin(g, 2, 0, 'Air ρ:', 0.8, 1.5, 1.225, 3, 0.005, ' kg/m³')
        self._cop = spin(g, 3, 0, 'CoP rear:', 0.0, 100.0, 50.0, 0, 5.0, ' %',
                         'Share of total downforce on the REAR axle.')
        self._grip = spin(g, 4, 0, 'Grip scale:', 0.3, 1.2, 0.65, 2, 0.05, ' ×',
                          'Track-vs-TTC-belt derate (0.6–0.7 typical).')
        # Live reference-speed readout: ties the Cl·A input to the number
        # aero people actually quote ("N at 60 mph"), so a wrong entry is
        # obvious immediately.
        self._aero_ref = QLabel('')
        self._aero_ref.setStyleSheet('color:#8a8a92; font-size:10px;')
        side.addWidget(self._aero_ref)

        def _upd_ref(*_):
            v = 60 / 2.23694                      # 60 mph in m/s
            q = 0.5 * float(self._rho.value()) * v * v
            F = float(self._cla.value()) * q
            self._aero_ref.setText(
                f'≈ {F:+.0f} N ({F/9.81:+.0f} kgf) downforce @ 60 mph · '
                f'drag {float(self._cda.value())*q:.0f} N @ 60 mph')
        for sb in (self._cla, self._cda, self._rho):
            sb.valueChanged.connect(_upd_ref)
        _upd_ref()

        hdr('GEARBOX  (primary × gear × final)')
        g = grid()
        self._primary = spin(g, 0, 0, 'Primary:', 0.5, 10.0, 1.0, 3, 0.01, '',
                             '1.0 = no primary reduction (this car: total = '
                             'gear × chain only).')
        self._final = spin(g, 1, 0, 'Final (chain):', 0.5, 10.0, 3.545, 3,
                           0.01, '', 'Driven/drive sprocket, e.g. 39/11.')
        self._redline = spin(g, 2, 0, 'Redline:', 3000, 20000, 11000, 0, 250,
                             ' rpm')
        self._gear_spins = []
        for i, ratio in enumerate(_DEFAULT_GEARS):
            sb = spin(g, 3 + i // 2, i % 2, f'{i + 1}:', 0.0, 5.0,
                      ratio, 3, 0.01, '',
                      'Gearbox internal ratio.  0 = gear not fitted.')
            self._gear_spins.append(sb)

        hdr('SIM')
        g = grid()
        g.addWidget(QLabel('Detail solves:'), 0, 0)
        self._ndetail = QSpinBox()
        self._ndetail.setRange(8, 400); self._ndetail.setValue(80)
        self._ndetail.setToolTip(
            'Stations where the FULL steady-state suspension solve runs.')
        g.addWidget(self._ndetail, 0, 1)

        self._btn = QPushButton('►  Sim Current Car')
        self._btn.setStyleSheet(
            'QPushButton { background:#FFD600; color:#0a0a0a; padding:10px;'
            ' font-weight:bold; font-size:13px; border-radius:4px; }'
            'QPushButton:hover { background:#FFEB3B; }'
            'QPushButton:disabled { background:#3a3a3a; color:#777; }')
        self._btn.clicked.connect(self._on_sim)
        side.addWidget(self._btn)

        self._status = QLabel('')
        self._status.setStyleSheet('color:#8a8a92; font-size:11px;')
        self._status.setWordWrap(True)
        side.addWidget(self._status)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet('color:#2a2a2a;')
        side.addWidget(sep)

        self._readout = QLabel('—')
        self._readout.setStyleSheet(
            "color:#e8e8ec; font-family:'Consolas','SF Mono',monospace;"
            'font-size:12px; background:#0e0e10; padding:8px;'
            'border:1px solid #2a2a2a; border-radius:4px;')
        self._readout.setWordWrap(True)
        side.addWidget(self._readout)

        # Standing scope caption — the grip model is a fresh-tire, single
        # thermal-state snapshot from the TTC cornering data.  Valid for
        # autocross (one ~50 s lap); it does NOT model endurance heat fade,
        # so the compound ranking here is fresh pace, not stint durability.
        scope = QLabel('Scope: fresh-tire, single-thermal-state grip (TTC '
                       'test window 30–73 °C).  Autocross-valid; endurance '
                       'heat-fade / degradation NOT modelled — compound order '
                       'may invert over a stint.')
        scope.setStyleSheet('color:#8a7a3a; font-size:10px; '
                            'padding:4px 2px; font-style:italic;')
        scope.setWordWrap(True)
        side.addWidget(scope)
        side.addStretch(1)

        side_inner = QWidget(); side_inner.setLayout(side)
        side_scroll = QScrollArea()
        side_scroll.setWidget(side_inner)
        side_scroll.setWidgetResizable(True)
        side_scroll.setFixedWidth(300)
        side_scroll.setStyleSheet('QScrollArea { border:0; background:#000; }')
        root.addWidget(side_scroll)

        # ── right side: ONE BIG GRID — track on top, graphs stacked below,
        #    all in a single figure/canvas.  Scroll = zoom, drag = pan.
        right = QVBoxLayout(); right.setSpacing(4)

        # top bar: track selector + checkboxes + Solo (one compact control row)
        topbar = QHBoxLayout(); topbar.setSpacing(8)
        topbar.addWidget(QLabel('Track:'))
        self._track_combo = QComboBox()
        self._track_combo.setStyleSheet('QComboBox { font-size:12px; }')
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._track_files = []
        for label, fname in _TRACKS:
            if os.path.isfile(os.path.join(base, 'tracks', fname)):
                self._track_combo.addItem(label)
                self._track_files.append(fname)
        self._track_combo.currentIndexChanged.connect(self._on_track_changed)
        topbar.addWidget(self._track_combo)
        topbar.addSpacing(12)
        topbar.addWidget(QLabel('Solo:'))
        self._solo = QComboBox()
        self._solo.addItem('— off —', None)
        for key, label, _ in self._GRAPHS:
            self._solo.addItem(label, key)
        self._solo.setToolTip('Isolate a single graph full-height (overrides '
                              'the checkboxes).')
        self._solo.currentIndexChanged.connect(self._redraw)
        topbar.addWidget(self._solo)
        topbar.addStretch(1)
        nav_hint = QLabel('scroll = zoom · drag = pan · double-click = reset')
        nav_hint.setStyleSheet('color:#666; font-size:10px;')
        topbar.addWidget(nav_hint)
        right.addLayout(topbar)

        # graph checkboxes
        cb_grid = QGridLayout(); cb_grid.setSpacing(8)
        cb_grid.addWidget(QLabel('Graphs:'), 0, 0)
        self._graph_cbs = {}
        for i, (key, label, on) in enumerate(self._GRAPHS):
            cb = QCheckBox(label)
            cb.setChecked(on)
            cb.setStyleSheet('QCheckBox { font-size:11px; }')
            cb.toggled.connect(self._redraw)
            self._graph_cbs[key] = cb
            cb_grid.addWidget(cb, i // 5, 1 + i % 5)
        cb_grid.setColumnStretch(6, 1)
        right.addLayout(cb_grid)

        # the ONE canvas — no toolbar; navigation via the scroll/drag handler.
        # constrained_layout cohabits the equal-aspect track map, the colorbar
        # and the stacked graphs without the collapse tight_layout caused.
        self._fig = Figure(facecolor='#000000', layout='constrained')
        self._canvas = FigureCanvas(self._fig)
        right.addWidget(self._canvas, 1)
        root.addLayout(right, 1)

        # custom navigation: scroll-zoom + drag-pan + double-click reset,
        # per-subplot (acts on the axes under the cursor).
        self._home_lims = {}      # ax -> (xlim, ylim) for double-click reset
        self._pan = None
        self._canvas.mpl_connect('scroll_event', self._on_scroll)
        self._canvas.mpl_connect('button_press_event', self._on_press)
        self._canvas.mpl_connect('motion_notify_event', self._on_drag)
        self._canvas.mpl_connect('button_release_event', self._on_release)

        # Hover value-readout — same machinery as the kinematics canvas.
        try:
            from gui.main_window import HoverAnnotator
            self._hover = HoverAnnotator(self._canvas)
        except Exception:
            pass

    # ── canvas navigation: scroll-zoom, drag-pan, dbl-click reset ─────────
    def _on_scroll(self, event):
        ax = event.inaxes
        if ax is None or event.xdata is None:
            return
        base = 1.25
        scale = (1.0 / base) if event.button == 'up' else base   # up = zoom in
        xd, yd = event.xdata, event.ydata
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        ax.set_xlim(xd - (xd - x0) * scale, xd + (x1 - xd) * scale)
        ax.set_ylim(yd - (yd - y0) * scale, yd + (y1 - yd) * scale)
        self._canvas.draw_idle()

    def _on_press(self, event):
        if event.inaxes is None:
            return
        if event.dblclick:                       # double-click → reset view
            lims = self._home_lims.get(event.inaxes)
            if lims:
                event.inaxes.set_xlim(lims[0]); event.inaxes.set_ylim(lims[1])
                self._canvas.draw_idle()
            return
        if event.button == 1:                    # left-drag → pan
            ax = event.inaxes
            bbox = ax.get_window_extent()
            self._pan = dict(ax=ax, px=event.x, py=event.y,
                             xlim=ax.get_xlim(), ylim=ax.get_ylim(),
                             wpx=max(bbox.width, 1.0), hpx=max(bbox.height, 1.0))

    def _on_drag(self, event):
        p = self._pan
        if not p or event.x is None:
            return
        ax = p['ax']
        x0, x1 = p['xlim']; y0, y1 = p['ylim']
        dx = (event.x - p['px']) / p['wpx'] * (x1 - x0)
        dy = (event.y - p['py']) / p['hpx'] * (y1 - y0)
        ax.set_xlim(x0 - dx, x1 - dx)
        ax.set_ylim(y0 - dy, y1 - dy)
        self._canvas.draw_idle()

    def _on_release(self, event):
        self._pan = None

    # ── track ────────────────────────────────────────────────────────────
    def _load_default_track(self):
        if self._track_files:
            self._load_track_file(self._track_files[0])
        else:
            self._track_lbl.setText('no track JSON in tracks/ — run '
                                    'tools/trace_track.py')

    def _discover_tires(self):
        """List the tire .mat files in the local tire_data/ folder (gitignored).
        Each label is the file's own tire_id, read at RUNTIME — so the public
        source never names a TTC tire or run.  Returns [(label, filename), …]."""
        import glob
        base = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'tire_data')
        try:
            import scipy.io as _sio
            import numpy as _np
        except Exception:
            _sio = None
        out, seen = [], set()
        for path in sorted(glob.glob(os.path.join(base, '*.mat'))):
            label = os.path.splitext(os.path.basename(path))[0]
            if _sio is not None:
                try:
                    m = _sio.loadmat(path, variable_names=['tireid'])
                    if 'tireid' in m and _np.size(m['tireid']):
                        label = str(_np.ravel(m['tireid'])[0])
                except Exception:
                    pass
            # dedupe by tire id: many files are the same tire at different test
            # conditions — show one entry per distinct tire (first file wins).
            if label in seen:
                continue
            seen.add(label)
            out.append((label, os.path.basename(path)))
        return out

    def _refresh_tire_active(self):
        """Show the currently-loaded tire's compound under the dropdown."""
        if not hasattr(self, '_tire_active'):
            return
        tm = getattr(self._main, '_tire_model', None)
        tid = getattr(tm, 'tire_id', None)
        if tid:
            self._tire_active.setText(f'active: {tid}')
        else:
            self._tire_active.setText('active: (parametric fallback — no TTC '
                                      'data loaded)')

    def _on_tire_round(self, combo_idx: int):
        """Swap the global tire model to the selected TTC round, then (if a
        lap already exists) auto re-sim so the comparison is immediate.
        combo index 0 is the neutral '(loaded tire)' placeholder — no-op."""
        idx = combo_idx - 1                       # account for placeholder
        if not (0 <= idx < len(self._tire_rounds)):
            return
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, 'tire_data', self._tire_rounds[idx][1])
        try:
            self._main._on_tire_file(path)        # swaps _tire_model app-wide
        except Exception as e:
            self._status.setText(f'Tire load failed: {e}')
            return
        self._refresh_tire_active()
        had_result = self._result is not None
        self._status.setText(
            f'Tire → {self._tire_rounds[idx][0]}'
            + ('  · re-simulating…' if had_result else '  · hit Sim to compare'))
        if had_result:
            self._on_sim()                        # instant A/B

    def _on_track_changed(self, idx: int):
        if 0 <= idx < len(self._track_files):
            # a new track invalidates the old result FIRST so the redraw
            # doesn't colour the new map with the old run's speed array.
            self._result = None
            self._last_lap_s = None
            self._readout.setText('—')
            self._load_track_file(self._track_files[idx])

    def _load_track_file(self, fname: str):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, 'tracks', fname)
        self._track = Track.from_json(path)
        t = self._track
        self._track_lbl.setText(
            f'{t.name}\n{t.length_m:.0f} m · width {t.width_m:.1f} m '
            f'· {t.n} stations (1 m)')
        self._redraw()

    # ── sim ──────────────────────────────────────────────────────────────
    def _on_sim(self):
        if self._track is None:
            self._status.setText('No track loaded.')
            return
        try:
            ss = self._main._build_dynamics_solver()
        except Exception as e:
            self._status.setText(f'Could not build the car: {e}')
            return
        sim = LapSimulator(
            ss,
            cla_m2=float(self._cla.value()),
            cda_m2=float(self._cda.value()),
            air_density=float(self._rho.value()),
            aero_cop_rear_frac=float(self._cop.value()) / 100.0,
            grip_scale=float(self._grip.value()),
        )
        # remembered for the aero graph title + front/rear split
        self._last_aero = (float(self._cla.value()), float(self._cda.value()),
                           float(self._cop.value()))
        sim.set_gearbox(
            [sb.value() for sb in self._gear_spins],
            primary_ratio=float(self._primary.value()),
            final_drive=float(self._final.value()),
            redline_rpm=float(self._redline.value()),
        )
        self._btn.setEnabled(False)
        self._status.setText('Simulating…')
        self._worker = _SimWorker(sim, self._track,
                                  int(self._ndetail.value()))
        self._worker.progress.connect(
            lambda m, p: self._status.setText(f'{m}  ({p}%)'))
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_fail(self, msg: str):
        self._btn.setEnabled(True)
        self._status.setText(f'Sim failed: {msg}')

    def _on_done(self, res):
        self._btn.setEnabled(True)
        self._result = res
        if getattr(res, 'notes', None):
            self._status.setText('⚠ ' + '  |  '.join(res.notes))
            self._status.setToolTip('\n'.join(res.notes))
        else:
            self._status.setText('Done.')
            self._status.setToolTip('')
        m, s = divmod(res.lap_time_s, 60.0)
        sect = '  /  '.join(f'{x:.2f}' for x in res.sector_times_s)
        top_gear = int(np.nanmax(res.gear)) if res.gear is not None else 0
        # Geared top speed @ redline — instant sanity check on the ratio set.
        # (If this reads absurdly low/high, one of primary/final/redline is
        # inconsistent with the gear ratios.)
        try:
            r_t = self._main._build_dynamics_solver()._veh.tire_radius_m
        except Exception:
            r_t = 0.2032
        ratio_top = (float(self._primary.value())
                     * float(min(sb.value() for sb in self._gear_spins
                                 if sb.value() > 0))
                     * float(self._final.value()))
        v_top = (float(self._redline.value()) / ratio_top / 60.0
                 * 2 * np.pi * r_t * 3.6)
        # Δ vs the previous run — the design-iteration number.
        delta = ''
        if self._last_lap_s is not None:
            dv = res.lap_time_s - self._last_lap_s
            delta = f'   ({dv:+.2f} s vs last)'
        self._last_lap_s = res.lap_time_s
        self._readout.setText(
            f'LAP TIME   {int(m)}:{s:05.2f}{delta}\n'
            f'sectors    {sect}\n'
            f'avg speed  {res.avg_speed_kph:6.1f} kph\n'
            f'max speed  {res.max_speed_kph:6.1f} kph\n'
            f'peak lat   {np.nanmax(res.lat_g):6.2f} g\n'
            f'peak brake {np.nanmin(res.lon_g):6.2f} g\n'
            f'peak accel {np.nanmax(res.lon_g):6.2f} g\n'
            f'top gear   {top_gear}   (geared top {v_top:.0f} kph '
            f'@ {self._redline.value():.0f})')
        self._redraw()

    # ── drawing ──────────────────────────────────────────────────────────
    @staticmethod
    def _style(ax, xlabel, ylabel, title):
        ax.set_facecolor('#080808')
        ax.tick_params(colors='#777777', labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#222222')
        ax.set_xlabel(xlabel, color='#888888', fontsize=8)
        ax.set_ylabel(ylabel, color='#888888', fontsize=8)
        ax.set_title(title, color='#cccccc', fontsize=9)
        ax.grid(True, color='#1a1a1a', lw=0.5)

    @staticmethod
    def _legend(ax):
        ax.legend(fontsize=7, facecolor='#06060e', labelcolor='white',
                  framealpha=0.7, ncol=4, loc='best', handlelength=1.2)

    def _checked(self):
        return [k for k, _l, _d in self._GRAPHS
                if self._graph_cbs[k].isChecked()]

    def _redraw(self, *_):
        """ONE big grid in a single canvas: the TRACK map always on top
        (equal-aspect, speed-coloured), then the checked graphs (or the Solo
        graph) stacked below as proper rectangular plots.  Navigate per
        subplot with scroll (zoom), drag (pan), double-click (reset)."""
        if self._track is None:
            return
        self._fig.clf()
        self._home_lims = {}
        t = self._track
        res = self._result

        solo_key = self._solo.currentData()
        if res is None:
            checked = []
        elif solo_key is not None:
            checked = [solo_key]
        else:
            checked = self._checked()

        n_rows = 1 + len(checked)
        # Track row ratio scales with how many graphs share the canvas, so the
        # map stays a usable size without a giant equal-aspect gap.
        track_ratio = 1.8 if len(checked) <= 1 else 1.3
        ratios = [track_ratio] + [1.0] * len(checked)
        gs = self._fig.add_gridspec(n_rows, 1, height_ratios=ratios)

        # ── row 0: the track, always ──────────────────────────────────────
        tax = self._fig.add_subplot(gs[0])
        if res is None:
            self._style(tax, 'x (m)', 'y (m)',
                        f'{t.name} — {t.length_m:.0f} m  (Sim to colour by speed)')
            tax.plot(t.x_m, t.y_m, color='#666666', lw=2.5)
        else:
            self._style(tax, 'x (m)', 'y (m)',
                        f'{t.name} — speed (lap {res.lap_time_s:.2f} s)')
            sc = tax.scatter(t.x_m, t.y_m, c=res.v_ms * 3.6, s=9, cmap='plasma')
            cb = self._fig.colorbar(sc, ax=tax, fraction=0.025, pad=0.01)
            cb.set_label('kph', color='#888888', fontsize=8)
            cb.ax.tick_params(colors='#777777', labelsize=7)
        tax.plot(t.x_m[0], t.y_m[0], 'g^', ms=12)
        tax.set_aspect('equal')
        self._home_lims[tax] = (tax.get_xlim(), tax.get_ylim())

        if res is None:
            self._canvas.draw_idle()
            return

        # ── graph rows: proper rectangles, shared distance x-axis ─────────
        s = res.s_m
        d = res.det_channels
        ds_ = res.det_s_m
        ax0 = None
        for gi, key in enumerate(checked):
            ax = self._fig.add_subplot(gs[1 + gi], sharex=ax0)
            if ax0 is None:
                ax0 = ax
            if key == 'speed':
                self._style(ax, 's (m)', 'kph', 'Speed vs distance')
                vmax_plot = np.nanmax(res.v_ms * 3.6) * 1.15
                cap = res.v_corner_cap_ms * 3.6
                cap_masked = np.where(cap < vmax_plot * 1.05, cap, np.nan)
                ax.plot(s, cap_masked, color='#555555',
                        lw=1.0, ls='--', label='corner cap')
                ax.plot(s, res.v_ms * 3.6, color='#FFD600', lw=1.6,
                        label='speed')
                ax.set_ylim(0, np.nanmax(res.v_ms * 3.6) * 1.15)
                self._legend(ax)
            elif key == 'gg':
                self._style(ax, 's (m)', 'g',
                            'Lateral + longitudinal acceleration')
                ax.plot(s, res.lat_g, color='#42A5F5', lw=1.4, label='lat |g|')
                ax.plot(s, res.lon_g, color='#E53935', lw=1.2, label='lon g')
                ax.axhline(0, color='#333', lw=0.6)
                self._legend(ax)
            elif key == 'util':
                self._style(ax, 's (m)', 'util',
                            'Tire utilization (full susp. solve)')
                for c in ('FL', 'FR', 'RL', 'RR'):
                    ax.plot(ds_, d[f'util_{c}'], color=_CC[c], ls=_LS[c],
                            lw=1.3, label=c)
                ax.axhline(1.0, color='#B0BEC5', lw=0.9, ls='--', alpha=0.7)
                self._legend(ax)
            elif key == 'lltd':
                self._style(ax, 's (m)', '% front', 'LLTD (total LT front share)')
                ax.ticklabel_format(axis='y', useOffset=False)
                ax.plot(ds_, d['lltd_pct'], color='#FFD600', lw=1.5,
                        label='LLTD %F')
                self._legend(ax)
            elif key == 'roll':
                self._style(ax, 's (m)', 'deg', 'Body roll')
                ax.plot(ds_, d['roll_deg'], color='#42A5F5', lw=1.5,
                        label='roll')
                ax.axhline(0, color='#333', lw=0.6)
                self._legend(ax)
            elif key == 'susp':
                self._style(ax, 's (m)', 'mm',
                            'Suspension travel (corner) + shock = travel × MR '
                            '(corner-coil convention; decoupled cars: see '
                            'heave/roll springs on the Suspension page)')
                for c in ('FL', 'FR', 'RL', 'RR'):
                    ax.plot(ds_, d[f'travel_{c}'], color=_CC[c], ls=_LS[c],
                            lw=1.1, label=c)
                ax.plot(ds_, d['shock_F_mm'], color='#FF9100', lw=1.6,
                        label='shock F')
                ax.plot(ds_, d['shock_R_mm'], color='#B0BEC5', lw=1.6,
                        label='shock R')
                self._legend(ax)
            elif key == 'aero':
                cla, cda, cop = getattr(self, '_last_aero',
                                        (0.0, 0.0, 50.0))
                self._style(ax, 's (m)', 'N',
                            f'Aero — Cl·A {cla:+.2f} · Cd·A {cda:.2f} · '
                            f'CoP {cop:.0f}%R   (down and drag are both '
                            f'∝ v² — same shape, different scale)')
                # distinct LINESTYLES: with Cl·A == Cd·A the curves are
                # numerically identical and would hide one another.
                ax.plot(s, res.aero_down_N, color='#42A5F5', lw=2.2,
                        ls='-', label='downforce total')
                ax.plot(s, res.aero_down_N * (1.0 - cop / 100.0),
                        color='#FFFFFF', lw=1.0, ls='--', label='down F')
                ax.plot(s, res.aero_down_N * (cop / 100.0),
                        color='#FFD600', lw=1.0, ls='-.', label='down R')
                ax.plot(s, res.aero_drag_N, color='#E53935', lw=1.4,
                        ls=':', label='drag')
                ax.axhline(0, color='#333', lw=0.6)
                if abs(np.nanmax(np.abs(res.aero_down_N))) < 1.0:
                    ax.text(0.5, 0.7, 'Cl·A = 0  —  no downforce package',
                            transform=ax.transAxes, ha='center',
                            color='#777777', fontsize=9)
                self._legend(ax)
            elif key == 'rpm':
                self._style(ax, 's (m)', 'rpm', 'Engine RPM + gear')
                ax.plot(s, res.engine_rpm, color='#FFD600', lw=1.3,
                        label='rpm')
                ax.axhline(float(self._redline.value()), color='#E53935',
                           lw=0.8, ls='--', alpha=0.6, label='redline')
                if res.gear is not None and np.nanmax(res.gear) > 0:
                    ax2 = ax.twinx()
                    ax2.step(s, res.gear, color='#42A5F5', lw=1.3,
                             where='post')
                    ax2.set_ylabel('gear', color='#42A5F5', fontsize=8)
                    ax2.tick_params(colors='#42A5F5', labelsize=8)
                    ax2.set_yticks(range(0, int(np.nanmax(res.gear)) + 2))
                    ax2.set_ylim(0, np.nanmax(res.gear) + 1)
                self._legend(ax)
            elif key == 'power':
                self._style(ax, 's (m)', 'hp', 'Power used (drive)')
                ax.plot(s, res.power_used_W / 745.7, color='#FFD600',
                        lw=1.3, label='power')
                try:
                    pk = float(self._main._dynamics_panel._power_hp.value())
                    ax.axhline(pk, color='#E53935', lw=0.8, ls='--',
                               alpha=0.6, label='engine peak')
                except Exception:
                    pass
                self._legend(ax)
            # only the bottom graph keeps its x tick labels (shared axis)
            if gi < len(checked) - 1:
                ax.tick_params(labelbottom=False)
                ax.set_xlabel('')
            self._home_lims[ax] = (ax.get_xlim(), ax.get_ylim())

        self._canvas.draw_idle()
