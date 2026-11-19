"""Ride page (Ctrl+8) — road-model ride analysis and THE RIDE-RATE SOLVE.

ONE MODEL: the vehicle is the main window's solved VehicleParams
(`_build_dynamics_solver()._veh`: ride/wheel/spring rates, motion ratios from
the kinematic solver, ARB rates, tire rate, masses, wheelbase, tracks). This
page only reads inputs, calls vahan.ride_solve / vahan.ride, and plots. No
physics is computed here.

Inputs that are NOT in the model and are the user's to supply (labelled
ASSUMED until the user confirms them): pitch inertia, roll inertia, the four
wheel-frame damping coefficients. Road class, speed bracket, seeds, the
left/right coherence bracket, the bump-stop travel margin, the bump and the
sweep range are analysis choices. All of them persist in the project car
dict under `ride_*` keys so they round-trip through .vahan save/load.

DAMPERS box: model = assumed linear (the four coefficients) or a dyno file
(vahan.damper; car['damper'] = {model, dyno_file, valve, lsc, hsc, lsr, hsr},
read back through MainWindow._damper_model(), the one accessor every damper
consumer uses).  With a dyno model the ride study iterates the equivalent
linear coefficients per road case and runs the nonlinear time-domain path
(vahan.ride_solve); the four coefficients are then not used.

Colours: graph LINES use the in-app corner palette (yellow/red/white/blue);
text follows docs/DESIGN.md (no yellow text, accent red for emphasis).
"""
from __future__ import annotations

import numpy as np
from gui.plot_dialog import ReadableCanvas
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from dataclasses import replace as _dc_replace
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QComboBox,
    QDoubleSpinBox, QSpinBox, QPushButton, QGroupBox, QTabWidget, QScrollArea,
    QProgressBar, QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QSplitter, QApplication, QLineEdit, QFileDialog,
    QListWidget, QListWidgetItem, QListView,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from vahan import ride_solve as RS
from vahan.ride import CORNERS
from vahan.road import RoadSpectrum

# docs/DESIGN.md dark tokens (text) — yellow is banned from text.
_INK, _MUTED, _ACCENT, _PANEL, _LINE, _PAPER = ('#ECECEE', '#9A9AA2', '#E23B48',
                                               '#141416', '#2A2A2E', '#0B0B0D')
# Graph LINE colours only (in-app corner palette, same as the Laptime page).
_CC = {'FL': '#FFD600', 'FR': '#E53935', 'RL': '#FFFFFF', 'RR': '#42A5F5'}
_SERIES = ['#FFD600', '#E53935', '#FFFFFF', '#42A5F5']       # per-speed lines
_STYLES = ['-', '--', ':', '-.']                              # per-coherence / per-sweep-axis
_LBF = RS.LBF_PER_IN_TO_NPM
# x axis of every frequency plot, in words (user 2026-10-23: "the x axis is hertz not velocity, so its confusing")
_HZ_LABEL = 'bumps hitting the wheel per second (Hz)  —  at 72 km/h: 2 Hz = a bump every 10 m, 20 Hz = every 1 m'
SUMMARY_SPEEDS_KPH = (20., 40., 60., 80., 100., 120.)

# Car-dict keys + placeholder defaults. The three ASSUMED groups are round
# placeholders on purpose — the page shouts until the user confirms them.
DEFAULTS = {
    'ride_pitch_inertia_kgm2': 100.0,
    'ride_roll_inertia_kgm2': 40.0,
    'ride_damping_wheel_Nspm': [1000.0, 1000.0, 1000.0, 1000.0],
    'ride_inertia_entered': False,       # False = the estimate from the car
    'ride_road_class': 'A',
    'ride_speed_min_kph': 36.0,
    'ride_speed_max_kph': 108.0,
    'ride_speed_count': 3,
    'ride_seed_first': 1,
    'ride_seed_count': 1,
    'ride_coherence': 'both',
    'ride_stop_margin_mm': None,          # None -> derived from the model's travel
    'ride_bump_height_mm': 25.0,
    'ride_bump_length_m': 0.5,
    'ride_flat_ride_max_ratio': 1.0,
    'ride_freq_front_min_Hz': 1.5,
    'ride_freq_front_max_Hz': 3.5,
    'ride_freq_rear_min_Hz': 1.5,
    'ride_freq_rear_max_Hz': 3.5,
    'ride_sweep_points': 5,
    'ride_damper_vref_mmps': 50.0,        # display-only equivalent-c velocity (Dampers box)
}
_DAMPER_MODELS = ('NO dyno data: straight-line placeholder', 'dyno file (measured)')


class _SweepWorker(QThread):
    progress = pyqtSignal(int, int)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, veh, inputs, front_rates, rear_rates):
        super().__init__()
        self._args = (veh, inputs, front_rates, rear_rates)
        self.cancel = False

    def run(self):
        try:
            sweep = RS.sweep_ride_rates(*self._args,
                                        progress=lambda d, t: self.progress.emit(d, t),
                                        cancelled=lambda: self.cancel)
            self.done.emit(sweep)
        except Exception as e:                      # surfaced on the page
            self.failed.emit(f'{type(e).__name__}: {e}')


# ── Springs x dampers vs speed (user 2026-10-23) ─────────────────────────────
# Front ride frequencies tried (Hz); the car's own value is always added and
# the rear keeps the car's rear/front ratio (vahan.ride_solve.ride_rate_options).
GRID_FRONT_HZ = (2.0, 2.4, 3.2, 3.6)
GRID_SPEEDS_KPH = SUMMARY_SPEEDS_KPH
# Damper settings tried, as Ohlins clicks/turns (LS: clicks from closed,
# 0 = stiffest; HS: turns from open, 0 = softest); the project's own setting
# is 'current'.  Compression and rebound set the same way, as on the dyno sheet.
GRID_DAMPERS = (('firm', 2, 3.5), ('current', None, None), ('soft', 12, 1.0), ('softest', 20, 0.0))
# soft -> stiff spring = blue -> white -> yellow (current) -> red (colour-blind safe)
_GRID_STYLE = [('#42A5F5', '--'), ('#42A5F5', '-'), ('#FFD600', '-'), ('#E53935', '-'), ('#E53935', '--')]
GRID_SHOW = (
    ('road', 'The road itself (both roads)', None, None),
    ('load_rms_N', 'Tyre load swing (N, average)', 'tyre load swing, average (N)', 1.),
    ('load_pct', 'Tyre load swing (% of tyre load)', 'swing / tyre load (%)', 1.),
    ('lat_rms_mm', 'Contact patch side-to-side (mm, average)', 'side-to-side movement, average (mm)', 1.),
    ('lat_p99_mm', 'Contact patch side-to-side (mm, 1-in-100 moments)', 'side-to-side, exceeded 1 % of the time (mm)', 1.),
    ('long_rms_mm', 'Contact patch fore-aft (mm, average)', 'fore-aft movement, average (mm)', 1.),
    ('long_p99_mm', 'Contact patch fore-aft (mm, 1-in-100 moments)', 'fore-aft, exceeded 1 % of the time (mm)', 1.),
)


def grid_damper_settings(project_setting):
    """[(label, setting dict)] for GRID_DAMPERS around the project's setting."""
    out = []
    for name, ls, hs in GRID_DAMPERS:
        s = dict(project_setting)
        if ls is not None:
            s.update(lsc=ls, lsr=ls, hsc=hs, hsr=hs)
        out.append((f'{name}: LS {s["lsc"]:g} clicks / HS {s["hsc"]:g} turns', s))
    return out


class _GridWorker(QThread):
    progress = pyqtSignal(int, int)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self):
        try:
            self.done.emit(self._fn(lambda d, t: self.progress.emit(d, t)))
        except Exception as e:                      # surfaced on the page
            self.failed.emit(f'{type(e).__name__}: {e}')


def plot_grid_road(fig, roads):
    """The fixed reference road: 100 m of it, and how tall the bumps are for
    each bump length (A and B are the same bumps, B twice as tall)."""
    fig.clear()
    ax1, ax2 = fig.subplots(1, 2, gridspec_kw={'width_ratios': [1.5, 1]})
    for cls, col in (('B', '#E53935'), ('A', '#FFD600')):
        r = roads[cls]
        keep = r['distance_m'] <= 100.
        ax1.plot(r['distance_m'][keep], r['left_m'][keep] * 1000, color=col, lw=1.,
                 label=f'road {cls}: height varies ±{r["rms_m"]*1000:.1f} mm on average')
        lo, hi = r['band_edges_cpm'][:-1], r['band_edges_cpm'][1:]
        x = np.arange(len(lo)) + (0.2 if cls == 'B' else -0.2)
        ax2.bar(x, r['band_rms_m'] * 1000, 0.4, color=col, label=f'road {cls}')
    _style_axes(ax1, 'The SAME road for every spring, damper and speed (first 100 m of 1 km, left wheel track)',
                'distance along the road (m)', 'road height (mm)')
    _legend(ax1, loc='upper right')
    r = roads['A']
    lo, hi = r['band_edges_cpm'][:-1], r['band_edges_cpm'][1:]
    ax2.set_xticks(np.arange(len(lo)))
    ax2.set_xticklabels([f'{1/h:.2g}-{1/l:.2g}' for l, h in zip(lo, hi)], rotation=45, ha='right')
    _style_axes(ax2, 'How tall the bumps are, by bump length', 'distance between bumps (m)', 'bump height, average (mm)')
    _legend(ax2)
    fig.suptitle(RS.ROAD_A_NOTE + ' Road B = the same bumps, twice as tall (rough public road).\n'
                 'The bumps are fixed in METRES, so a faster car meets them more often: '
                 'at 72 km/h a 10 m bump hits 2 times a second, a 1 m bump 20 times.',
                 color=_INK, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))


def plot_grid_study(fig, grid, road_class, key):
    """Rows front / rear tyre, one column per damper setting, one line per
    spring choice, x = car speed."""
    fig.clear()
    meta = {k: (lab, ylab) for k, lab, ylab, _ in GRID_SHOW}
    lab, ylab = meta[key]
    d = grid['study'][road_class]
    rates, dampers = grid['rates'], grid['damper_labels']
    axes = fig.subplots(2, len(dampers), sharex=True, sharey='row', squeeze=False)
    n_off = 0
    for row, (corner, name) in enumerate(((0, 'FRONT tyre'), (2, 'REAR tyre'))):
        top = 0.
        for j, dl in enumerate(dampers):
            ax = axes[row, j]
            for i, ro in enumerate(rates):
                col, ls = _GRID_STYLE[i % len(_GRID_STYLE)]
                cur = ro['is_current']
                y = d[key][i, j, :, corner]
                top = max(top, float(np.max(y)))
                ax.plot(d['speed_kph'], y, ls, color=col, lw=2.6 if cur else 1.4, marker='o', ms=3,
                        label=(f'{ro["spring_front_lbfin"]:.0f}/{ro["spring_rear_lbfin"]:.0f} lbf/in '
                               f'({ro["front_Hz"]:.1f}/{ro["rear_Hz"]:.1f} Hz)' + ('  = NOW' if cur else '')))
                off = d['min_load_N'][i, j, :, corner] <= 0.
                n_off += int(off.sum())
                if off.any():
                    ax.plot(d['speed_kph'][off], y[off], 'x', color='#FFFFFF', ms=9, mew=2.2, zorder=5)
            _style_axes(ax, f'{name}, dampers {dl}', 'car speed (km/h)' if row == 1 else '', ylab if j == 0 else '')
        axes[row, 0].set_ylim(0., 1.05 * top if top > 0 else 1.)
    _legend(axes[0, 0], title='springs front/rear', title_fontsize=6.5)
    i0 = next(i for i, r in enumerate(rates) if r['is_current'])
    j0 = next(j for j, l in enumerate(dampers) if l.startswith('current'))
    v60 = int(np.argmin(np.abs(d['speed_kph'] - 60.)))
    lo, hi = float(np.min(d[key][:, :, v60, 0])), float(np.max(d[key][:, :, v60, 0]))
    fig.suptitle(f'{lab} - road {road_class} - front tyre at 60 km/h: {d[key][i0, j0, v60, 0]:.2f} now, '
                 f'{lo:.2f} to {hi:.2f} across every spring and damper tried.\n'
                 + (RS.ROAD_A_NOTE if road_class == 'A' else 'Road B = the same bumps as road A, twice as tall.')
                 + ' Downforce and the Ohlins dyno curves included.'
                 + (f'\nWhite X = the tyre LEAVES THE ROAD at some moment in that run ({n_off} points): '
                    'the numbers there are not trustworthy (the model assumes the tyre stays on the road).'
                    if n_off else '\nThe tyre stays on the road in every run shown.'),
                 color=_INK, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))


def _style_axes(ax, title, xlabel, ylabel):
    ax.set_facecolor(_PANEL)
    ax.set_title(title, color=_INK, fontsize=9, loc='left')
    ax.set_xlabel(xlabel, color=_MUTED, fontsize=8)
    ax.set_ylabel(ylabel, color=_MUTED, fontsize=8)
    ax.tick_params(colors=_MUTED, labelsize=7)
    ax.grid(color=_LINE, lw=0.6)
    for s in ax.spines.values():
        s.set_color('#33333a')


def _legend(ax, **kw):
    ax.legend(fontsize=6.5, facecolor='#1b1b1e', edgecolor='#33333a', labelcolor=_INK, **kw)


class RidePage(QWidget):
    def __init__(self, main):
        super().__init__()
        self._main = main
        self._worker = None
        self.last_study = None
        self.last_sweep = None
        self.last_selection = None
        self.setStyleSheet(
            f'QWidget{{background:{_PAPER};color:{_INK};font-size:13px;}}'
            f'QGroupBox{{border:1px solid {_LINE};border-radius:4px;margin-top:14px;padding-top:6px;}}'
            f'QGroupBox::title{{subcontrol-origin:margin;left:8px;color:{_ACCENT};'
            'font-weight:650;letter-spacing:1px;font-size:11px;}'
            'QDoubleSpinBox,QSpinBox,QComboBox,QPushButton{background:#1e1e24;border:1px solid #444;'
            'border-radius:3px;padding:2px 4px;}'
            'QPushButton:hover{background:#2a2a32;}'
            f'QTableWidget{{background:{_PANEL};gridline-color:{_LINE};}}'
            f'QHeaderView::section{{background:#1e1e24;color:{_MUTED};padding:4px;border:0;font-size:11px;}}')
        root = QHBoxLayout(self); root.setContentsMargins(10, 10, 10, 10); root.setSpacing(10)

        # ── left: inputs (scrollable) ──────────────────────────────────────
        left_w = QWidget(); left = QVBoxLayout(left_w); left.setSpacing(6)
        left.setContentsMargins(0, 0, 6, 0)
        self._build_inputs(left)
        from gui.side_panel import SideScroll, side_split
        scroll = SideScroll(left_w, 440)   # resizable, never clipped

        # ── right: warning banner + tabs ───────────────────────────────────
        right = QVBoxLayout(); right.setSpacing(6)
        self._banner = QLabel(''); self._banner.setWordWrap(True)
        right.addWidget(self._banner)
        self._tabs = QTabWidget()
        self._figs, self._canvases = {}, {}
        for key, title in (('summary', 'SUMMARY: tyre load vs speed'),
                           ('grid', 'Springs x dampers vs speed'),
                           ('metrics', 'Detail: every road case'),
                           ('bump', 'One big bump'),
                           ('contact', 'Tyre load vs spring choice'),
                           ('solve', 'Spring search'), ('result', 'Spring search result'),
                           ('launch', 'Launch'),
                           ('damper', 'Dampers'),
                           ('dynosheet', 'Dyno curves'),
                           ('transfer', 'Advanced 1'),
                           ('psd', 'Advanced 2')):
            if key == 'result':
                self._tabs.addTab(self._build_result_tab(), title)
                continue
            if key == 'grid':
                self._tabs.addTab(self._build_grid_tab(), title)
                continue
            if key == 'dynosheet':
                self._tabs.addTab(self._build_dynosheet_tab(), title)
                continue
            fig = Figure(figsize=(10, 6.5), facecolor='#1b1b1e')
            canvas = ReadableCanvas(fig)
            self._figs[key], self._canvases[key] = fig, canvas
            self._tabs.addTab(canvas, title)
        right.addWidget(self._tabs, 1)
        self._status = QLabel('Press "Analyse current car" for the road metrics of the springs '
                              'in the model, or "Solve ride rates" for the sweep.')
        self._status.setWordWrap(True); self._status.setStyleSheet(f'color:{_MUTED};font-size:12px')
        right.addWidget(self._status)
        self._side_split = side_split(scroll, right, 440, key='ride')
        root.addWidget(self._side_split, 1)

        self._load_saved()
        self._refresh_model_readout()
        self._refresh_banner()
        self._refresh_damper_info()

    # ── inputs ────────────────────────────────────────────────────────────
    def _spin(self, lo, hi, val, dec, step, suffix='', tip=''):
        s = QDoubleSpinBox(); s.setRange(lo, hi); s.setDecimals(dec); s.setSingleStep(step)
        s.setValue(float(val)); s.setSuffix(suffix); s.setToolTip(tip)
        s.valueChanged.connect(self._store)
        return s

    def _ispin(self, lo, hi, val, tip=''):
        s = QSpinBox(); s.setRange(lo, hi); s.setValue(int(val)); s.setToolTip(tip)
        s.valueChanged.connect(self._store)
        return s

    def _row(self, grid, r, text, widget):
        lab = QLabel(text); lab.setStyleSheet(f'color:{_INK}')
        grid.addWidget(lab, r, 0); grid.addWidget(widget, r, 1)
        return lab

    def _build_inputs(self, left):
        d = DEFAULTS
        # 1. what the page uses, every number with where it comes from (ONE MODEL)
        box = QGroupBox('WHAT THIS PAGE USES (read only)'); lay = QVBoxLayout(box)
        self._model_lbl = QLabel(''); self._model_lbl.setWordWrap(True)
        self._model_lbl.setStyleSheet(f"font-family:Consolas,'SF Mono',monospace;font-size:11px;color:{_INK}")
        lay.addWidget(self._model_lbl)
        self._used_lbl = QLabel(''); self._used_lbl.setWordWrap(True)
        self._used_lbl.setStyleSheet(f'color:{_INK};font-size:12px')
        lay.addWidget(self._used_lbl)
        rb = QPushButton('Re-read the car'); rb.clicked.connect(self._refresh_model_readout)
        lay.addWidget(rb)
        left.addWidget(box)

        # 2. body inertia: estimated from the car unless the user has measured values
        box = QGroupBox('BODY INERTIA (ESTIMATED, NOT MEASURED)'); g = QGridLayout(box); g.setSpacing(4)
        self._inertia_entered = QCheckBox('use my own measured values instead (tick to type them in)')
        self._inertia_entered.setStyleSheet(f'color:{_INK};font-size:12px')
        self._inertia_entered.toggled.connect(self._store)
        self._inertia_entered.toggled.connect(self._refresh_model_readout)
        self._pitch_I = self._spin(1, 2000, d['ride_pitch_inertia_kgm2'], 1, 5, ' kg·m²',
                                   'Measured body pitch inertia (only used when the box above is ticked).')
        self._roll_I = self._spin(1, 2000, d['ride_roll_inertia_kgm2'], 1, 5, ' kg·m²',
                                  'Measured body roll inertia (only used when the box above is ticked).')
        g.addWidget(self._inertia_entered, 0, 0, 1, 2)
        self._inertia_rows = (self._row(g, 1, 'Pitch inertia', self._pitch_I),
                              self._row(g, 2, 'Roll inertia', self._roll_I))
        self._inertia_lbl = QLabel(''); self._inertia_lbl.setWordWrap(True)
        self._inertia_lbl.setStyleSheet(f'color:{_MUTED};font-size:11px')
        g.addWidget(self._inertia_lbl, 3, 0, 1, 2)
        left.addWidget(box)
        self._damp = {}
        for i, c in enumerate(CORNERS):
            self._damp[c] = self._spin(0, 20000, d['ride_damping_wheel_Nspm'][i], 0, 50, ' N·s/m',
                                       f'{c} damping at the WHEEL, straight line (force / speed). Only used '
                                       'when there is no dyno file.')
        self._build_damper_box(left)

        # 3. the road: the one big assumption
        box = QGroupBox('ROAD (MADE UP, NOT MEASURED)'); g = QGridLayout(box); g.setSpacing(4)
        self._road = QComboBox(); self._road.addItems(['A', 'B'])
        self._road.setToolTip('ISO 8608 road roughness. A = very good public road. B = rough public road, '
                              'bumps twice as tall as A. Nobody has measured an FSAE track.')
        self._road.currentTextChanged.connect(self._store)
        self._coh = QComboBox(); self._coh.addItems(['both', 'coherent', 'independent'])
        self._coh.setToolTip('coherent = left and right wheels hit the same bumps (car only bounces and pitches). '
                             'independent = left and right bumps unrelated (car also rolls). both = run both, '
                             'keep the worse. Real roads are in between.')
        self._coh.currentTextChanged.connect(self._store)
        self._row(g, 0, 'Road for the detail tabs', self._road)
        self._row(g, 1, 'Left vs right bumps', self._coh)   # both = same bumps both sides AND different bumps, keep the worse
        note = QLabel('A = very good public road. B = rough public road (bumps 2x taller). '
                      'The Summary tab always shows both. Left vs right: "both" runs the same bumps on '
                      'both sides AND different bumps each side, and keeps the worse.')
        note.setWordWrap(True); note.setStyleSheet(f'color:{_MUTED};font-size:11px')
        g.addWidget(note, 2, 0, 1, 2)
        left.addWidget(box)

        # 4. what is left out
        box = QGroupBox('NOT CONSIDERED'); lay = QVBoxLayout(box)
        nc = QLabel('\n'.join('• ' + t for t in RS.NOT_CONSIDERED)); nc.setWordWrap(True)
        nc.setStyleSheet(f'color:{_INK};font-size:11px')
        lay.addWidget(nc)
        left.addWidget(box)

        # 5. calculation settings (not the car)
        box = QGroupBox('EXPERT SETTINGS (NOT THE CAR - LEAVE ALONE)'); g = QGridLayout(box); g.setSpacing(4)
        self._v_min = self._spin(3, 200, d['ride_speed_min_kph'], 0, 2, ' km/h', 'Slowest speed for the detail tabs.')
        self._v_max = self._spin(3, 200, d['ride_speed_max_kph'], 0, 2, ' km/h', 'Fastest speed for the detail tabs.')
        self._v_n = self._ispin(1, 8, d['ride_speed_count'], 'Number of speeds between slowest and fastest.')
        self._seed0 = self._ispin(0, 10 ** 6, d['ride_seed_first'], 'Which random road to draw (same roughness, '
                                                                      'different bump order).')
        self._seed_n = self._ispin(1, 8, d['ride_seed_count'],
                                   'How many random roads to run. Averages do not change with this; '
                                   'the single worst moments (peaks, lowest load) do.')
        r = 0
        for text, w in (('Detail tabs: slowest', self._v_min), ('Detail tabs: fastest', self._v_max),
                        ('Number of speeds', self._v_n), ('Random road number', self._seed0),
                        ('How many random roads', self._seed_n)):
            self._row(g, r, text, w); r += 1
        left.addWidget(box)

        # 6. limits: travel + one bump
        box = QGroupBox('TRAVEL + ONE BIG BUMP (YOUR CHOICE)'); g = QGridLayout(box); g.setSpacing(4)
        self._margin = self._spin(1, 200, 25., 1, 1, ' mm',
                                  'Wheel travel from ride height to the nearer stop. '
                                  'More travel than this on the road = hits the stop.')
        self._margin_btn = QPushButton('Take travel from the car')
        self._margin_btn.setToolTip('Fill from the model: (stroke - sag)/MR and sag/MR per axle, the smaller one, '
                                    'at the preload and stroke in the Motion panel.')
        self._margin_btn.clicked.connect(self._use_model_travel)
        self._margin_user = False
        self._margin.valueChanged.connect(self._on_margin_typed)
        self._bump_h = self._spin(1, 200, d['ride_bump_height_mm'], 1, 1, ' mm', 'Height of the single bump.')
        self._bump_l = self._spin(0.05, 5, d['ride_bump_length_m'], 2, 0.05, ' m', 'Length of the single bump along the road.')
        self._flat = self._spin(0.1, 10, d['ride_flat_ride_max_ratio'], 2, 0.05, '',
                                'After the bump: how much the car may rock nose-to-tail compared with how much it '
                                'bounces. 1.0 = no more rocking than bouncing (RCVD p.410 "lands flat").')
        r = 0
        self._row(g, r, 'Travel to the stop', self._margin); r += 1
        g.addWidget(self._margin_btn, r, 0, 1, 2); r += 1
        self._travel_lbl = QLabel(''); self._travel_lbl.setWordWrap(True)
        self._travel_lbl.setStyleSheet(f'color:{_MUTED};font-size:11px')
        g.addWidget(self._travel_lbl, r, 0, 1, 2); r += 1
        self._row(g, r, 'Bump height', self._bump_h); r += 1
        self._row(g, r, 'Bump length', self._bump_l); r += 1
        self._row(g, r, 'Max rocking / bouncing', self._flat); r += 1
        left.addWidget(box)

        # 7. spring search range
        box = QGroupBox('SPRING SEARCH RANGE (AS RIDE FREQUENCY)'); g = QGridLayout(box); g.setSpacing(4)
        tip = 'Ride frequency = how many times per second the body bounces on its springs. Stiffer spring = higher.'
        self._ff_min = self._spin(0.5, 10, d['ride_freq_front_min_Hz'], 2, 0.25, ' Hz', tip)
        self._ff_max = self._spin(0.5, 10, d['ride_freq_front_max_Hz'], 2, 0.25, ' Hz', tip)
        self._fr_min = self._spin(0.5, 10, d['ride_freq_rear_min_Hz'], 2, 0.25, ' Hz', tip)
        self._fr_max = self._spin(0.5, 10, d['ride_freq_rear_max_Hz'], 2, 0.25, ' Hz', tip)
        self._npts = self._ispin(2, 12, d['ride_sweep_points'], 'Steps per axle (front x rear pairs).')
        r = 0
        for text, w in (('Front softest', self._ff_min), ('Front stiffest', self._ff_max),
                        ('Rear softest', self._fr_min), ('Rear stiffest', self._fr_max), ('Steps per axle', self._npts)):
            self._row(g, r, text, w); r += 1
        self._range_lbl = QLabel(''); self._range_lbl.setWordWrap(True)
        self._range_lbl.setStyleSheet(f'color:{_MUTED};font-size:11px')
        g.addWidget(self._range_lbl, r, 0, 1, 2)
        for s in (self._ff_min, self._ff_max, self._fr_min, self._fr_max):
            s.valueChanged.connect(self._refresh_range_lbl)
        left.addWidget(box)

        # actions
        self._analyse_btn = QPushButton('Analyse current car')
        self._analyse_btn.setStyleSheet(f'background:{_ACCENT};color:#fff;font-weight:600;')
        self._analyse_btn.setToolTip('Tyre load vs speed on roads A and B, plus the detail tabs, for the '
                                     'springs and dampers in the car now.')
        self._analyse_btn.clicked.connect(self.analyse_current)
        self._solve_btn = QPushButton('Search springs')
        self._solve_btn.setToolTip('Try every front x rear spring pair in the search range; keep the one with the '
                                   'steadiest tyre load that stays off the stops and lands flat after the bump.')
        self._solve_btn.clicked.connect(lambda: self.run_solve(blocking=False))
        self._cancel_btn = QPushButton('Cancel'); self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel)
        left.addWidget(self._analyse_btn); left.addWidget(self._solve_btn); left.addWidget(self._cancel_btn)
        self._progress = QProgressBar(); self._progress.setRange(0, 1); self._progress.setValue(0)
        self._progress.setTextVisible(True)
        left.addWidget(self._progress)
        left.addStretch(1)

    def _build_damper_box(self, left):
        """DAMPERS: assumed linear (the four coefficients above) or a dyno
        file.  Persists in car['damper']; consumers read it back through
        MainWindow._damper_model()."""
        box = QGroupBox('DAMPERS (FROM THE DYNO FILE)'); g = QGridLayout(box); g.setSpacing(4)
        self._dmodel = QComboBox(); self._dmodel.addItems(list(_DAMPER_MODELS))
        self._dmodel.setToolTip('Placeholder = the four straight-line numbers below, NOT this car. Dyno file = '
                                'measured force-velocity curves (LS + HS adjusters combined as '
                                'parallel flow, see vahan/damper.py), at the solved motion ratio.')
        self._dfile = QLineEdit(); self._dfile.setPlaceholderText('dyno JSON (repo-relative path)')
        self._dfile.setToolTip('Path of the dyno file, relative to the repository root '
                               '(or absolute). Saved in the project as car["damper"]["dyno_file"].')
        self._dbrowse = QPushButton('Browse…'); self._dbrowse.clicked.connect(self._browse_dyno)
        self._dvalve = QComboBox(); self._dvalve.setToolTip('Valve specification (from the dyno file).')
        self._dlsc = QSpinBox(); self._dlsr = QSpinBox()
        for s in (self._dlsc, self._dlsr):
            s.setRange(0, 25); s.setValue(6)
            s.setToolTip('Low-speed adjuster, CLICKS FROM FULLY CLOSED (0 = stiffest).')
        self._dhsc = QDoubleSpinBox(); self._dhsr = QDoubleSpinBox()
        for s in (self._dhsc, self._dhsr):
            s.setRange(0., 4.3); s.setDecimals(2); s.setSingleStep(0.25); s.setValue(2.)
            s.setToolTip('High-speed adjuster, TURNS FROM FULLY OPEN (0 = softest, 4.3 = stiffest).')
        self._dvref = QDoubleSpinBox(); self._dvref.setRange(1., 1000.); self._dvref.setDecimals(0)
        self._dvref.setSingleStep(5.); self._dvref.setSuffix(' mm/s')
        self._dvref.setValue(DEFAULTS['ride_damper_vref_mmps'])
        self._dvref.setToolTip('Damper velocity at which the box quotes the equivalent linear '
                               'coefficient and damping ratios (display only). "Analyse current '
                               'car" iterates the real one per road case.')
        r = 0
        self._row(g, r, 'Damper data', self._dmodel); r += 1
        g.addWidget(self._dfile, r, 0); g.addWidget(self._dbrowse, r, 1); r += 1
        self._damp_labels = {}
        for c in CORNERS:
            self._damp_labels[c] = self._row(g, r, f'{c} placeholder at wheel', self._damp[c]); r += 1
        self._row(g, r, 'Valve spec', self._dvalve); r += 1
        self._row(g, r, 'LS comp (clicks)', self._dlsc); r += 1
        self._row(g, r, 'HS comp (turns)', self._dhsc); r += 1
        self._row(g, r, 'LS reb (clicks)', self._dlsr); r += 1
        self._row(g, r, 'HS reb (turns)', self._dhsr); r += 1
        self._row(g, r, 'Show numbers at', self._dvref); r += 1
        self._dinfo = QLabel(''); self._dinfo.setWordWrap(True)
        self._dinfo.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._dinfo.setStyleSheet(f"font-family:Consolas,'SF Mono',monospace;font-size:11px;color:{_INK}")
        g.addWidget(self._dinfo, r, 0, 1, 2)
        self._dmodel.currentIndexChanged.connect(self._on_damper_edit)
        self._dfile.editingFinished.connect(self._on_dyno_file)
        self._dvalve.currentTextChanged.connect(self._on_damper_edit)
        for s in (self._dlsc, self._dhsc, self._dlsr, self._dhsr, self._dvref):
            s.valueChanged.connect(self._on_damper_edit)
        left.addWidget(box)

    def _damper_widgets(self):
        return (self._dmodel, self._dfile, self._dvalve, self._dlsc, self._dhsc,
                self._dlsr, self._dhsr, self._dvref)

    def _browse_dyno(self):
        from vahan.damper import repo_root
        import os
        path, _ = QFileDialog.getOpenFileName(self, 'Damper dyno file', repo_root(), 'JSON (*.json)')
        if not path:
            return
        try:
            rel = os.path.relpath(path, repo_root())
            path = rel if not rel.startswith('..') else path
        except ValueError:                      # other drive
            pass
        self._dfile.setText(path.replace('\\', '/'))
        self._on_dyno_file()

    def _load_dyno_catalogue(self, select_valve=None):
        """Fill the valve list and the adjuster ranges from the file."""
        from vahan.damper import resolve_dyno_path, load_dyno
        path = self._dfile.text().strip()
        if not path:
            return None
        dyno = load_dyno(resolve_dyno_path(path, getattr(self._main, '_project_path', None)))
        self._dvalve.blockSignals(True)
        try:
            want = select_valve or self._dvalve.currentText() or dyno.delivered_valve_spec
            self._dvalve.clear(); self._dvalve.addItems(dyno.valve_specs)
            if want in dyno.valve_specs:
                self._dvalve.setCurrentText(want)
            elif dyno.delivered_valve_spec in dyno.valve_specs:
                self._dvalve.setCurrentText(dyno.delivered_valve_spec)
        finally:
            self._dvalve.blockSignals(False)
        valve = self._dvalve.currentText()
        for s, d in ((self._dlsc, 'comp'), (self._dlsr, 'reb')):
            lo, hi = dyno.ls_range(valve, d); s.setRange(int(np.floor(lo)), int(np.ceil(hi)))
        for s, d in ((self._dhsc, 'comp'), (self._dhsr, 'reb')):
            lo, hi = dyno.hs_range(valve, d); s.setRange(lo, hi)
        return dyno

    def _on_dyno_file(self):
        try:
            self._load_dyno_catalogue()
        except Exception as e:
            self._dinfo.setText(f'dyno file not usable: {type(e).__name__}: {e}')
        self._on_damper_edit()

    def _on_damper_edit(self, *_):
        self._store()
        self._refresh_damper_info()
        sync = getattr(self._main, '_sync_damper_consumers', None)
        if sync is not None:
            sync()

    def _store_damper(self, car):
        """car['damper'] from the box.  A project that never chose a dyno
        keeps NO block (behaviour unchanged); switching back to assumed
        linear keeps the block with model 'linear' so the adjusters survive."""
        dyno = self._dmodel.currentIndex() == 1
        block = car.get('damper') if isinstance(car.get('damper'), dict) else None
        car['ride_damper_vref_mmps'] = float(self._dvref.value())
        if not dyno and block is None:
            return
        block = dict(block or {})
        block.update(model='dyno' if dyno else 'linear', dyno_file=self._dfile.text().strip(),
                     valve=self._dvalve.currentText(), lsc=int(self._dlsc.value()),
                     hsc=float(self._dhsc.value()), lsr=int(self._dlsr.value()),
                     hsr=float(self._dhsr.value()))
        car['damper'] = block

    def _load_damper_saved(self, car):
        block = car.get('damper') if isinstance(car.get('damper'), dict) else {}
        self._dmodel.setCurrentIndex(1 if block.get('model') == 'dyno' else 0)
        self._dfile.setText(str(block.get('dyno_file', '') or ''))
        try:
            self._load_dyno_catalogue(select_valve=block.get('valve'))
        except Exception:
            pass                                    # surfaced by _refresh_damper_info
        for key, s in (('lsc', self._dlsc), ('lsr', self._dlsr)):
            if key in block:
                s.setValue(int(round(float(block[key]))))
        for key, s in (('hsc', self._dhsc), ('hsr', self._dhsr)):
            if key in block:
                s.setValue(float(block[key]))
        if 'ride_damper_vref_mmps' in car:
            self._dvref.setValue(float(car['ride_damper_vref_mmps']))

    def damper_model(self):
        """The project damper (MainWindow._damper_model) or None."""
        return self._main._damper_model()

    def _refresh_damper_info(self, *_):
        """Equivalent coefficients, damping ratios and the F-v plot of the
        current setting at the solved MR (or the linear placeholders)."""
        from vahan.damper import vehicle_damping_ratios
        try:
            dm = self.damper_model()
        except Exception as e:
            self._dinfo.setText(f'DAMPER MODEL ERROR: {type(e).__name__}: {e}')
            self._dinfo.setStyleSheet(f"font-family:Consolas,'SF Mono',monospace;font-size:11px;color:{_ACCENT}")
            return None
        self._dinfo.setStyleSheet(f"font-family:Consolas,'SF Mono',monospace;font-size:11px;color:{_INK}")
        for c in CORNERS:                       # placeholders only exist when there is no dyno file
            self._damp[c].setVisible(dm is None); self._damp_labels[c].setVisible(dm is None)
        try:
            veh = self.vehicle()
        except Exception as e:
            self._dinfo.setText(f'model not available: {type(e).__name__}: {e}')
            return None
        roll_I = self.inertias()['roll_kgm2']
        if dm is None:
            c = [float(self._damp[k].value()) for k in CORNERS]
            z = vehicle_damping_ratios(veh, 0.5 * (c[0] + c[1]), 0.5 * (c[2] + c[3]), roll_I)
            self._dinfo.setText(
                'NO DYNO DATA: straight-line placeholder numbers, not this car.\n'
                f'zeta ride F/R {z["ride_front"]:.2f}/{z["ride_rear"]:.2f}  '
                f'hop F/R {z["hop_front"]:.2f}/{z["hop_rear"]:.2f}  roll {z["roll"]:.2f}')
            self._plot_damper(None, veh)
            return {'model': 'linear', 'zeta': z}
        wd = dm.wheel_for_vehicle(veh)
        v = float(self._dvref.value()) / 1000.
        c_avg, c_comp, c_reb = wd.equivalent_wheel_c([v] * 4)
        z = vehicle_damping_ratios(veh, c_avg[0], c_avg[2], roll_I)
        s = dm.setting
        lines = [f'{dm.dyno.name}  {s["valve"]}',
                 f'LSC {s["lsc"]:g} cl / HSC {s["hsc"]:g} t   LSR {s["lsr"]:g} cl / HSR {s["hsr"]:g} t',
                 f'MR (solved) F/R {veh.motion_ratio_front:.3f}/{veh.motion_ratio_rear:.3f}',
                 f'at {v*1000:.0f} mm/s damper (secant):',
                 f' damper c comp/reb {c_comp[0]/wd.mr[0]**2:,.0f}/{c_reb[0]/wd.mr[0]**2:,.0f} N·s/m',
                 f' wheel c F {c_avg[0]:,.0f} (c {c_comp[0]:,.0f} / r {c_reb[0]:,.0f}) N·s/m',
                 f' wheel c R {c_avg[2]:,.0f} (c {c_comp[2]:,.0f} / r {c_reb[2]:,.0f}) N·s/m',
                 f' zeta ride F/R {z["ride_front"]:.2f}/{z["ride_rear"]:.2f}  hop F/R '
                 f'{z["hop_front"]:.2f}/{z["hop_rear"]:.2f}  roll {z["roll"]:.2f}']
        out = {'model': 'dyno', 'v_ref_mps': v, 'c_wheel_Nspm': c_avg, 'zeta': z}
        study = self.last_study
        if study is not None and study.get('damping_kind') == 'dyno':
            ref = study['reference_case_index']; e = study['damper_equivalent'][ref]
            cr = e['c_wheel_Nspm']
            zr = vehicle_damping_ratios(veh, cr[0], cr[2], roll_I)
            vr = e['damper_v_rms_mps']
            lines += [f'last analysis, {study["cases"][ref].label}:',
                      f' RMS damper v F/R {vr[0]*1000:.1f}/{vr[2]*1000:.1f} mm/s, '
                      f'{e["iterations"]} it, {"converged" if e["converged"] else "NOT converged"}',
                      f' wheel c F/R {cr[0]:,.0f}/{cr[2]:,.0f} N·s/m',
                      f' zeta ride F/R {zr["ride_front"]:.2f}/{zr["ride_rear"]:.2f}  hop F/R '
                      f'{zr["hop_front"]:.2f}/{zr["hop_rear"]:.2f}  roll {zr["roll"]:.2f}']
            out.update(reference=e, zeta_reference=zr)
        if s.get('clamped'):
            lines.append('CLAMPED to the measured range: ' + ', '.join(s['clamped']))
        if s.get('off_path'):
            lines.append('This sheet moves LS and HS together: HS set to the sheet path ('
                         + ', '.join(s['off_path']) + ')')
        lines.append(f'dyno range {wd.curve.vmax_c*1000:.0f} mm/s comp / {wd.curve.vmax_r*1000:.0f} reb; '
                     'faster than that: the curve is carried on in a straight line, and the page says so')
        self._dinfo.setText('\n'.join(lines))
        self._plot_damper(dm, veh)
        return out

    def _plot_damper(self, dm, veh):
        fig = self._figs['damper']; fig.clear()
        axes = fig.subplots(1, 2)
        lin = [float(self._damp[k].value()) for k in CORNERS]
        if dm is None:
            vw = np.linspace(-0.4, 0.4, 401)
            ax = axes[1]
            ax.plot(vw * 1000, lin[0] * vw, color=_CC['FL'], lw=1.6, label=f'front {lin[0]:.0f} N·s/m')
            ax.plot(vw * 1000, lin[2] * vw, color=_CC['FR'], lw=1.6, ls='--', label=f'rear {lin[2]:.0f} N·s/m')
            _style_axes(ax, 'Wheel frame: assumed linear coefficients', 'wheel velocity (mm/s, + bump)', 'force at wheel (N)')
            _legend(ax)
            axes[0].set_axis_off(); axes[0].set_facecolor(_PANEL)
            axes[0].text(0.02, 0.9, 'No dyno damper in the project.\nChoose "dyno file" in the DAMPERS box.',
                         transform=axes[0].transAxes, color=_MUTED, fontsize=9, va='top')
        else:
            wd = dm.wheel_for_vehicle(veh)
            cur = wd.curve
            vmax = max(cur.vmax_c, cur.vmax_r)
            v = np.linspace(-1.6 * vmax, 1.6 * vmax, 801)
            F = cur(v)
            inside = (v <= cur.vmax_c) & (v >= -cur.vmax_r)
            ax = axes[0]
            ax.plot(v[inside] * 1000, F[inside], color='#FFFFFF', lw=1.8, label='this setting (dyno range)')
            for part in (v > cur.vmax_c, v < -cur.vmax_r):
                ax.plot(v[part] * 1000, F[part], color='#FFFFFF', lw=1.2, ls='--')
            ax.plot([], [], color='#FFFFFF', lw=1.2, ls='--', label='beyond dyno: 50 mm/s secant')
            for x in (cur.vmax_c, -cur.vmax_r):
                ax.axvline(x * 1000, color=_ACCENT, lw=0.8, ls=':')
            ax.axhline(0, color=_LINE, lw=0.8); ax.axvline(0, color=_LINE, lw=0.8)
            s = dm.setting
            _style_axes(ax, f'Damper frame: {s["valve"]} LSC {s["lsc"]:g}/HSC {s["hsc"]:g}  '
                            f'LSR {s["lsr"]:g}/HSR {s["hsr"]:g}',
                        'damper velocity (mm/s, + compression)', 'damper force (N)')
            _legend(ax, loc='upper left')
            ax = axes[1]
            vw = np.linspace(-1.6 * vmax / min(wd.mr), 1.6 * vmax / min(wd.mr), 801)
            fw = wd(vw[:, None] * np.ones(4))
            ax.plot(vw * 1000, fw[:, 0], color=_CC['FL'], lw=1.7, label=f'front (MR {wd.mr[0]:.3f})')
            ax.plot(vw * 1000, fw[:, 2], color=_CC['FR'], lw=1.7, ls='--', label=f'rear (MR {wd.mr[2]:.3f})')
            ax.axhline(0, color=_LINE, lw=0.8); ax.axvline(0, color=_LINE, lw=0.8)
            _style_axes(ax, 'Wheel frame: F_damper(MR·v_wheel)·MR at the solved MR',
                        'wheel velocity (mm/s, + bump)', 'force at wheel (N)')
            _legend(ax, loc='upper left')
        fig.tight_layout()
        self._canvases['damper'].draw_idle()
        self._plot_dyno_sheet(dm)

    def _sheet_records(self, dyno, valve):
        """vahan.damper.compare_curve for every printed curve of one valve
        spec (+ the sheet's comparison page), cached per (file, valve)."""
        from vahan.damper import sheet_curves, compare_curve
        cache = self.__dict__.setdefault('_sheet_cache', {})
        key = (dyno.source, id(dyno), valve)
        if key not in cache:
            curves = [c for c in sheet_curves(dyno.raw)
                      if c['valve'] == valve or c['group'] == 'comparison']
            cache.clear()
            cache[key] = [compare_curve(dyno, c) for c in curves if c['valve'] in dyno.valve_specs]
        return cache[key]

    def _build_dynosheet_tab(self):
        """Dyno curves tab: the plot plus controls to pick WHICH curves show
        (any valve on the sheet, any printed curve, measured dots on/off, the
        project's own setting on/off) and to add any LS/HS setting the sheet
        does not print (model, interpolated between printed curves)."""
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(4, 4, 4, 4)
        row = QHBoxLayout()
        row.addWidget(QLabel('Valve'))
        self._ds_valve = QComboBox(); self._ds_valve.setToolTip('Any valve code on the dyno sheet, not only the project one')
        row.addWidget(self._ds_valve)
        self._ds_dots = QCheckBox('measured dots'); self._ds_dots.setChecked(True)
        self._ds_proj = QCheckBox("project's setting (DAMPERS box)"); self._ds_proj.setChecked(True)
        self._ds_reb = QCheckBox('rebound'); self._ds_reb.setChecked(True)
        for cb in (self._ds_dots, self._ds_proj, self._ds_reb):
            row.addWidget(cb)
        b_all = QPushButton('All curves'); b_none = QPushButton('No curves')
        row.addWidget(b_all); row.addWidget(b_none)
        row.addStretch(1)
        lay.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel('Add any setting (model, between printed curves):  LS clicks'))
        self._ds_ls = QDoubleSpinBox(); self._ds_ls.setRange(0, 25); self._ds_ls.setDecimals(1); self._ds_ls.setValue(3)
        row2.addWidget(self._ds_ls)
        row2.addWidget(QLabel('HS turns'))
        self._ds_hs = QDoubleSpinBox(); self._ds_hs.setRange(0, 4.3); self._ds_hs.setDecimals(1)
        self._ds_hs.setSingleStep(0.5); self._ds_hs.setValue(2)
        row2.addWidget(self._ds_hs)
        b_add = QPushButton('Add'); b_clr = QPushButton('Clear added')
        row2.addWidget(b_add); row2.addWidget(b_clr); row2.addStretch(1)
        lay.addLayout(row2)
        self._ds_list = QListWidget(); self._ds_list.setFlow(QListView.Flow.LeftToRight)
        self._ds_list.setWrapping(True); self._ds_list.setMaximumHeight(64)
        self._ds_list.setToolTip('Tick the printed curves to show')
        lay.addWidget(self._ds_list)
        fig = Figure(figsize=(10, 6.5), facecolor='#1b1b1e')
        canvas = ReadableCanvas(fig)
        self._figs['dynosheet'], self._canvases['dynosheet'] = fig, canvas
        lay.addWidget(canvas, 1)
        self._ds_hidden = set()          # (valve, key) the user unticked
        self._ds_added = []              # [(valve, ls, hs)]
        self._ds_filling = False

        def redraw(*_):
            self._plot_dyno_sheet(getattr(self, '_ds_dm', None))

        def ticked(item):
            if self._ds_filling:
                return
            k = (self._ds_valve.currentText(), item.text())
            (self._ds_hidden.discard if item.checkState() == Qt.CheckState.Checked else self._ds_hidden.add)(k)
            redraw()

        def set_all(on):
            v = self._ds_valve.currentText()
            for i in range(self._ds_list.count()):
                k = (v, self._ds_list.item(i).text())
                (self._ds_hidden.discard if on else self._ds_hidden.add)(k)
            self._ds_list_valve = None   # force re-fill with the new ticks
            redraw()

        def add():
            self._ds_added.append((self._ds_valve.currentText(), float(self._ds_ls.value()), float(self._ds_hs.value())))
            redraw()

        def clear():
            self._ds_added.clear(); redraw()

        self._ds_list.itemChanged.connect(ticked)
        self._ds_valve.currentTextChanged.connect(redraw)
        for cb in (self._ds_dots, self._ds_proj, self._ds_reb):
            cb.toggled.connect(redraw)
        b_all.clicked.connect(lambda: set_all(True)); b_none.clicked.connect(lambda: set_all(False))
        b_add.clicked.connect(add); b_clr.clicked.connect(clear)
        self._ds_list_valve = None
        return w

    def _plot_dyno_sheet(self, dm):
        """The curves the dyno sheet prints for the chosen valve spec,
        recreated by the ONE damper model (lines) over the digitised dyno
        points (dots); compression up, rebound down.  What shows is picked on
        the tab (valve, curves, dots, the project's setting, added settings);
        nothing is fixed.  Sweeps layout: LS sweep | HS sweep.  Path layout:
        path | comparison page."""
        from matplotlib.colors import LinearSegmentedColormap
        from vahan.damper import check_summary
        self._ds_dm = dm
        fig = self._figs['dynosheet']; fig.clear()
        if dm is None:
            ax = fig.add_subplot(111); ax.set_axis_off(); ax.set_facecolor(_PANEL)
            ax.text(0.02, 0.9, 'No dyno damper in the project.\nChoose "dyno file" in the DAMPERS box.',
                    transform=ax.transAxes, color=_MUTED, fontsize=9, va='top')
            self._canvases['dynosheet'].draw_idle()
            return
        dyno, s = dm.dyno, dm.setting
        # valve picker: every valve on the sheet, default = the project's
        have = [self._ds_valve.itemText(i) for i in range(self._ds_valve.count())]
        if have != list(dyno.valve_specs):
            self._ds_valve.blockSignals(True); self._ds_valve.clear()
            self._ds_valve.addItems(list(dyno.valve_specs)); self._ds_valve.setCurrentText(s['valve'])
            self._ds_valve.blockSignals(False)
        valve = self._ds_valve.currentText() or s['valve']
        recs = self._sheet_records(dyno, valve)
        groups = (('LS sweep', 'HS sweep') if dyno.layout(valve) == 'sweeps'
                  else ('path', 'comparison'))
        # curve list (ticks) for this valve
        pre = {'LS sweep': 'LS ', 'HS sweep': 'HS ', 'path': '', 'comparison': 'compare '}
        lab = lambda r: pre.get(r['group'], '') + str(r['key'])
        keys = [lab(r) for r in recs if r['group'] in groups]
        if self._ds_list_valve != (valve, tuple(keys)) or self._ds_list.count() != len(keys):
            self._ds_filling = True; self._ds_list.clear()
            for k in keys:
                it = QListWidgetItem(k)
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(Qt.CheckState.Unchecked if (valve, k) in self._ds_hidden else Qt.CheckState.Checked)
                self._ds_list.addItem(it)
            self._ds_filling = False; self._ds_list_valve = (valve, tuple(keys))
        show_dots, show_reb = self._ds_dots.isChecked(), self._ds_reb.isChecked()
        dirs = (('comp', 1.), ('reb', -1.)) if show_reb else (('comp', 1.),)
        cmap = LinearSegmentedColormap.from_list('byw', ['#42A5F5', '#FFFFFF', '#FFD600'])
        axes = fig.subplots(1, 2)
        for ax, grp in zip(axes, groups):
            rows_all = [r for r in recs if r['group'] == grp and (grp == 'comparison' or r['valve'] == valve)]
            rows = [r for r in rows_all if (valve, lab(r)) not in self._ds_hidden]
            if not rows_all:
                ax.set_axis_off(); ax.set_facecolor(_PANEL)
                ax.text(0.02, 0.9, f'no {grp} curves on this sheet', transform=ax.transAxes,
                        color=_MUTED, fontsize=9, va='top')
                continue
            for r in rows:
                i = rows_all.index(r)
                col = cmap(i / max(len(rows_all) - 1, 1))
                like = not (r['clamped'] or r['off_path'])
                for d, sign in dirs:
                    if d not in r:
                        continue
                    c = r[d]
                    v = np.linspace(0., float(c['v_mps'].max()), 120)
                    F = dyno.force(sign * v, r['setting'])
                    ax.plot(sign * v * 1000, F, color=col, lw=1.3, ls='-' if like else '--',
                            label=(lab(r) + ('' if like else ' (model off the printed setting)'))
                            if d == 'comp' else None)
                    if show_dots:
                        ax.plot(sign * c['v_mps'] * 1000, sign * c['F_meas_N'], 'o', ms=2.6, color=col,
                                mec='none', alpha=0.85)
            if grp != 'comparison':
                lo = -dyno.dyno_max_velocity(s, 'reb') if show_reb else 0.
                if self._ds_proj.isChecked() and s['valve'] == valve:
                    v = np.linspace(lo, dyno.dyno_max_velocity(s, 'comp'), 241)
                    ax.plot(v * 1000, dyno.force(v, s), color='#E53935', lw=2.2,
                            label=f"project: LSC {s['lsc']:g}/HSC {s['hsc']:g} LSR {s['lsr']:g}/HSR {s['hsr']:g}")
                for j, (av, ls_, hs_) in enumerate(self._ds_added):
                    if av != valve:
                        continue
                    st = {'valve': av, 'lsc': ls_, 'hsc': hs_, 'lsr': ls_, 'hsr': hs_}
                    try:
                        v = np.linspace(-dyno.dyno_max_velocity(st, 'reb') if show_reb else 0.,
                                        dyno.dyno_max_velocity(st, 'comp'), 241)
                        ax.plot(v * 1000, dyno.force(v, st), color='#F2C14E', lw=2.0,
                                ls=('--', '-.', ':')[j % 3], label=f'added LS {ls_:g} / HS {hs_:g} (model)')
                    except ValueError:
                        pass
            ax.axhline(0, color=_LINE, lw=0.8); ax.axvline(0, color=_LINE, lw=0.8)
            sm = check_summary(rows_all)
            st0 = rows_all[0]['setting']
            what = {'LS sweep': f"LS clicks (Ohlins measured these at HS {st0['hsc']:g} turns)",
                    'HS sweep': f"HS turns (Ohlins measured these at LS {st0['lsc']:g} clicks)",
                    'path': 'LS-HS settings (combined path)', 'comparison': 'comparison page'}[grp]
            _style_axes(ax, f'{valve} {what}\nmodel vs dyno from 25 mm/s: worst {sm["max_abs_N"]:.1f} N, '
                            f'rms {sm["rms_N"]:.2f} N' if sm['n'] else f'{valve} {what}',
                        'damper velocity (mm/s, + compression)', 'damper force (N)')
            _legend(ax, loc='upper left')
        fig.suptitle(f'{dyno.name}: lines = Vahan damper model, dots = digitised dyno', color=_INK, fontsize=9)
        fig.tight_layout()
        self._canvases['dynosheet'].draw_idle()

    def _build_grid_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        top = QHBoxLayout()
        self._grid_run = QPushButton('Run springs x dampers (about 5 min)')
        self._grid_run.setStyleSheet(f'background:{_ACCENT};color:white;font-weight:bold;padding:4px 10px')
        self._grid_run.clicked.connect(self._on_grid_run)
        top.addWidget(self._grid_run)
        top.addWidget(QLabel('Show'))
        self._grid_show = QComboBox()
        for k, lab, _, _ in GRID_SHOW:
            self._grid_show.addItem(lab, k)
        self._grid_show.setCurrentIndex(1)
        top.addWidget(self._grid_show, 1)
        top.addWidget(QLabel('Road'))
        self._grid_road = QComboBox(); self._grid_road.addItems(['A', 'B'])
        top.addWidget(self._grid_road)
        self._grid_bar = QProgressBar(); self._grid_bar.setVisible(False)
        top.addWidget(self._grid_bar)
        lay.addLayout(top)
        self._grid_note = QLabel(
            f'Every spring choice (front {", ".join(f"{h:g}" for h in GRID_FRONT_HZ)} Hz and the car\'s own, rear at '
            "the car's rear/front ratio) x every damper setting (firm / current / soft / softest clicks on the "
            'Ohlins dyno curves) x speeds ' + ', '.join(f'{v:.0f}' for v in GRID_SPEEDS_KPH) + ' km/h, on the SAME '
            'fixed road. Contact patch movement = where the tyre touches the road, side-to-side and fore-aft, '
            'relative to the frame, from the solved suspension (same as the 3-D view) at each wheel position.')
        self._grid_note.setWordWrap(True); self._grid_note.setStyleSheet(f'color:{_MUTED};font-size:12px')
        lay.addWidget(self._grid_note)
        self._figs['grid'] = Figure(figsize=(10, 6.5), facecolor='#1b1b1e')
        self._canvases['grid'] = ReadableCanvas(self._figs['grid'])
        lay.addWidget(self._canvases['grid'], 1)
        self._grid = None
        self._grid_worker = None
        self._grid_show.currentIndexChanged.connect(self._grid_replot)
        self._grid_road.currentIndexChanged.connect(self._grid_replot)
        return w

    def grid_study(self, veh=None, progress=None):
        """Springs x dampers x speed on the fixed reference road (vahan.ride_solve.
        spring_damper_speed_study) with this page's inputs, the project's dyno
        dampers and the solved corners' contact-patch tables."""
        from vahan.damper import WheelDampers
        veh = veh or self.vehicle()
        dm = self.damper_model()
        if dm is None:
            raise ValueError('needs the dyno dampers: choose "dyno file (measured)" in the Dampers box')
        inputs = self.summary_inputs(veh)
        mr = RS.corner_motion_ratios(veh)
        sets = grid_damper_settings(dm.setting)
        dampers = [(lab, WheelDampers(dm.dyno, st, mr)) for lab, st in sets]
        mf, _ = RS.sprung_corner_masses_kg(veh)
        ff0 = RS.ride_frequency_Hz(veh.ride_rate_front_Npm, mf)
        freqs = sorted({float(h) for h in GRID_FRONT_HZ if abs(h - ff0) > 0.1} | {ff0})
        rates = RS.ride_rate_options(veh, freqs)
        solvers = self._main._solvers
        tables = [RS.contact_patch_table(solvers[c], veh.tire_radius_m) for c in CORNERS]
        speeds = np.asarray(GRID_SPEEDS_KPH) / 3.6
        study = RS.spring_damper_speed_study(veh, inputs, rates, dampers, speeds, tables, progress=progress)
        return {'study': study, 'rates': rates, 'damper_labels': [lab for lab, _ in sets],
                'damper_settings': [st for _, st in sets], 'cp_tables': tables,
                'roads': {c: RS.reference_road(c, inputs.record_length_m, inputs.spacing_m) for c in 'AB'}}

    def _on_grid_run(self):
        if self._grid_worker is not None and self._grid_worker.isRunning():
            return
        try:
            veh = self.vehicle()
        except Exception as e:
            self._grid_note.setText(f'Cannot build the car: {type(e).__name__}: {e}')
            return
        self._grid_run.setEnabled(False); self._grid_bar.setVisible(True); self._grid_bar.setValue(0)
        self._grid_worker = _GridWorker(lambda prog: self.grid_study(veh, prog))
        self._grid_worker.progress.connect(lambda d, t: (self._grid_bar.setMaximum(t), self._grid_bar.setValue(d)))
        self._grid_worker.done.connect(self._on_grid_done)
        self._grid_worker.failed.connect(self._on_grid_failed)
        self._grid_worker.start()

    def _on_grid_done(self, grid):
        self._grid = grid
        self._grid_run.setEnabled(True); self._grid_bar.setVisible(False)
        self._grid_replot()

    def _on_grid_failed(self, msg):
        self._grid_run.setEnabled(True); self._grid_bar.setVisible(False)
        self._grid_note.setText(f'Springs x dampers run failed: {msg}')

    def _grid_replot(self, *_):
        if self._grid is None:
            return
        key = self._grid_show.currentData()
        if key == 'road':
            plot_grid_road(self._figs['grid'], self._grid['roads'])
        else:
            plot_grid_study(self._figs['grid'], self._grid, self._grid_road.currentText(), key)
        self._canvases['grid'].draw_idle()

    def _build_result_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self._result_lbl = QLabel('No solve yet.'); self._result_lbl.setWordWrap(True)
        self._result_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._result_lbl.setStyleSheet(f"font-family:Consolas,'SF Mono',monospace;font-size:12px;color:{_INK}")
        lay.addWidget(self._result_lbl)
        split = QSplitter(Qt.Orientation.Vertical)
        self._spring_tables = {}
        cols = ['Catalogue spring\n(lbf/in)', 'Spring\n(N/m)', 'Wheel rate\nit gives (N/m)',
                'Ride rate\nit gives (N/m)', 'Ride freq\nit gives (Hz)', 'vs solved\nride rate (%)',
                'Preload to hold\nride height (mm)', 'Shock sag at\nthat preload (mm)',
                'Ride-height change\nif preload < 0 (mm)']
        for axle in RS.AXLES:
            box = QGroupBox(f'{axle.upper()} — nearest catalogue springs (25 lbf/in steps)')
            bl = QVBoxLayout(box)
            t = QTableWidget(0, len(cols)); t.setHorizontalHeaderLabels(cols)
            t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            t.horizontalHeader().setFixedHeight(48)
            t.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
            t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            t.verticalHeader().setVisible(False)
            bl.addWidget(t); split.addWidget(box)
            self._spring_tables[axle] = t
        lay.addWidget(split, 1)
        return w

    # ── car dict persistence ──────────────────────────────────────────────
    def _car(self):
        return getattr(self._main, '_car', {})

    def _load_saved(self):
        car = self._car()
        widgets = {'ride_pitch_inertia_kgm2': self._pitch_I, 'ride_roll_inertia_kgm2': self._roll_I,
                   'ride_speed_min_kph': self._v_min, 'ride_speed_max_kph': self._v_max,
                   'ride_speed_count': self._v_n, 'ride_seed_first': self._seed0,
                   'ride_seed_count': self._seed_n, 'ride_bump_height_mm': self._bump_h,
                   'ride_bump_length_m': self._bump_l, 'ride_flat_ride_max_ratio': self._flat,
                   'ride_freq_front_min_Hz': self._ff_min, 'ride_freq_front_max_Hz': self._ff_max,
                   'ride_freq_rear_min_Hz': self._fr_min, 'ride_freq_rear_max_Hz': self._fr_max,
                   'ride_sweep_points': self._npts}
        for w in list(widgets.values()) + list(self._damp.values()) + [self._margin]:
            w.blockSignals(True)
        for cb in (self._road, self._coh, self._inertia_entered):
            cb.blockSignals(True)
        try:
            for key, w in widgets.items():
                if key in car:
                    w.setValue(type(w.value())(car[key]))
            damp = car.get('ride_damping_wheel_Nspm')
            if isinstance(damp, (list, tuple)) and len(damp) == 4:
                for c, v in zip(CORNERS, damp):
                    self._damp[c].setValue(float(v))
            if car.get('ride_road_class') in ('A', 'B'):
                self._road.setCurrentText(car['ride_road_class'])
            if car.get('ride_coherence') in RS.COHERENCE_CHOICES:
                self._coh.setCurrentText(car['ride_coherence'])
            self._inertia_entered.setChecked(bool(car.get('ride_inertia_entered',
                                                          car.get('ride_inputs_confirmed', False))))
            self._margin_user = bool(car.get('ride_stop_margin_user', False))
            margin = car.get('ride_stop_margin_mm') if self._margin_user else None
            if margin is None:
                m = self._model_travel_mm()
                margin = m['margin_mm'] if m else 25.
            self._margin.setValue(float(margin))
            for w in self._damper_widgets():
                w.blockSignals(True)
            self._load_damper_saved(car)
        finally:
            for w in list(widgets.values()) + list(self._damp.values()) + [self._margin]:
                w.blockSignals(False)
            for cb in (self._road, self._coh, self._inertia_entered):
                cb.blockSignals(False)
            for w in self._damper_widgets():
                w.blockSignals(False)
        self._store()

    def _store(self, *_):
        car = self._car()
        car['ride_pitch_inertia_kgm2'] = float(self._pitch_I.value())
        car['ride_roll_inertia_kgm2'] = float(self._roll_I.value())
        car['ride_damping_wheel_Nspm'] = [float(self._damp[c].value()) for c in CORNERS]
        car['ride_inertia_entered'] = bool(self._inertia_entered.isChecked())
        car.pop('ride_inputs_confirmed', None)
        car['ride_road_class'] = self._road.currentText()
        car['ride_speed_min_kph'] = float(self._v_min.value())
        car['ride_speed_max_kph'] = float(self._v_max.value())
        car['ride_speed_count'] = int(self._v_n.value())
        car['ride_seed_first'] = int(self._seed0.value())
        car['ride_seed_count'] = int(self._seed_n.value())
        car['ride_coherence'] = self._coh.currentText()
        car['ride_stop_margin_mm'] = float(self._margin.value())
        car['ride_stop_margin_user'] = bool(getattr(self, '_margin_user', False))
        car['ride_bump_height_mm'] = float(self._bump_h.value())
        car['ride_bump_length_m'] = float(self._bump_l.value())
        car['ride_flat_ride_max_ratio'] = float(self._flat.value())
        car['ride_freq_front_min_Hz'] = float(self._ff_min.value())
        car['ride_freq_front_max_Hz'] = float(self._ff_max.value())
        car['ride_freq_rear_min_Hz'] = float(self._fr_min.value())
        car['ride_freq_rear_max_Hz'] = float(self._fr_max.value())
        car['ride_sweep_points'] = int(self._npts.value())
        self._store_damper(car)
        self._refresh_banner()

    def _refresh_banner(self, *_):
        dyno = self._dmodel.currentIndex() == 1 if hasattr(self, '_dmodel') else False
        bad = []
        if not dyno:
            bad.append('dampers are PLACEHOLDER straight lines (no dyno file)')
        if not self._aero_label().startswith('ON'):
            bad.append('NO downforce (aero is off in the Dynamics panel)')
        text = ('From the car: masses, springs, motion ratios, bars, tyre rate, travel'
                + (', dampers (dyno file)' if dyno else '') + '. '
                + 'Estimated: body inertia' + (' (you entered measured values)' if self._inertia_entered.isChecked()
                                               else ' (from mass and size, not measured)') + '. '
                + 'Made up: the road (ISO 8608, nobody measured a track). '
                + 'Left out: see NOT CONSIDERED.')
        if bad:
            self._banner.setText('CHECK: ' + '; '.join(bad) + '.  ' + text)
            self._banner.setStyleSheet(f'color:{_ACCENT};font-size:12px;font-weight:600;padding:4px;'
                                       f'border:1px solid {_ACCENT};border-radius:4px;')
        else:
            self._banner.setText(text)
            self._banner.setStyleSheet(f'color:{_INK};font-size:12px;padding:4px;'
                                       f'border:1px solid {_LINE};border-radius:4px;')

    def inertias(self):
        """Body pitch / roll inertia the page uses: the ONE estimate
        (vahan.ride_solve.estimated_inertias, same rules as the transient
        model) unless the user ticked 'use my own measured values'."""
        if self._inertia_entered.isChecked():
            return {'pitch_kgm2': float(self._pitch_I.value()), 'roll_kgm2': float(self._roll_I.value()),
                    'source': 'entered'}
        e = RS.estimated_inertias(self.vehicle())
        e['source'] = 'estimated'
        return e

    def _aero_label(self):
        p = self._aero()
        if not p:
            return 'OFF: no downforce in this analysis (aero is off in the Dynamics panel)'
        try:
            from gui.build_tolerance_page import aero_package
            return f'ON: {aero_package(self._main)["label"]} (Dynamics panel); loads at each speed include it'
        except Exception:
            return f'ON: ClA {p["aero_cla_m2"]:.2f} m², {100 * p["aero_cop_rear"]:.0f} % rear'

    def _refresh_range_lbl(self, *_):
        """Ride frequency -> the spring it means on THIS car (lbf/in)."""
        try:
            veh = self.vehicle()
            mf, mr = RS.sprung_corner_masses_kg(veh)
            parts = []
            for axle, m, lo, hi in (('front', mf, self._ff_min.value(), self._ff_max.value()),
                                    ('rear', mr, self._fr_min.value(), self._fr_max.value())):
                k = [RS.spring_rate_for_ride_rate_Npm(veh, axle, RS.ride_rate_from_frequency_Npm(f, m)) / _LBF
                     for f in (lo, hi)]
                parts.append(f'{axle} {lo:.2f}-{hi:.2f} Hz = springs {k[0]:.0f}-{k[1]:.0f} lbf/in')
            self._range_lbl.setText('On this car: ' + '; '.join(parts) + '.')
        except Exception as e:
            self._range_lbl.setText(f'spring range not available: {type(e).__name__}: {e}')

    # ── model access (ONE MODEL) ──────────────────────────────────────────
    def vehicle(self):
        return self._main._build_dynamics_solver()._veh

    def _motion(self):
        mp = getattr(self._main, '_motion_panel', None)
        return dict(preload_front_mm=float(getattr(mp, 'preload_front_mm', 0.) or 0.),
                    preload_rear_mm=float(getattr(mp, 'preload_rear_mm', 0.) or 0.),
                    stroke_mm=float(getattr(mp, 'stroke_mm', 55.) or 55.))

    def _model_travel_mm(self, veh=None):
        """Wheel travel to the bump and droop stops per axle from the model's
        static sag (stroke - sag)/MR and sag/MR; margin = the smallest."""
        try:
            veh = veh or self.vehicle()
            mo = self._motion()
            sag = veh.static_sag(**mo)
            out = {}
            for axle in RS.AXLES:
                mr = float(getattr(veh, f'motion_ratio_{axle}'))
                s = float(sag[f'sag_shock_{axle}_mm'])
                out[f'{axle}_bump_mm'] = (mo['stroke_mm'] - s) / mr
                out[f'{axle}_droop_mm'] = s / mr
            out['margin_mm'] = max(1., min(out[k] for k in out))
            return out
        except Exception:
            return None

    def _on_margin_typed(self, *_):
        self._margin_user = True
        self._store()
        self._refresh_model_readout()

    def _use_model_travel(self):
        m = self._model_travel_mm()
        if m:
            self._margin.blockSignals(True)
            self._margin.setValue(m['margin_mm'])
            self._margin.blockSignals(False)
            self._margin_user = False
            self._store()
            self._refresh_model_readout()

    def _margin_words(self, inputs):
        return (f'travel to the stop {inputs.stop_margin_m*1000:.1f} mm '
                + ('(YOUR value)' if self._margin_user else '(from the car)'))

    def _refresh_model_readout(self):
        try:
            veh = self.vehicle()
        except Exception as e:
            self._model_lbl.setText(f'model not available: {type(e).__name__}: {e}')
            return
        mf, mr = RS.sprung_corner_masses_kg(veh)
        ff, fr = (RS.ride_frequency_Hz(veh.ride_rate_front_Npm, mf),
                  RS.ride_frequency_Hz(veh.ride_rate_rear_Npm, mr))
        lines = [
            f'sprung mass          {veh.sprung_mass_kg:8.1f} kg',
            f'unsprung F / R       {veh.unsprung_mass_front_kg:5.1f} / {veh.unsprung_mass_rear_kg:5.1f} kg per axle',
            f'sprung corner F / R  {mf:5.1f} / {mr:5.1f} kg',
            f'wheelbase            {veh.wheelbase_m*1000:8.0f} mm   CG to front axle {veh.cg_to_front_axle_m*1000:.0f} mm',
            f'track F / R          {veh.front_track_m*1000:5.0f} / {veh.rear_track_m*1000:5.0f} mm',
            f'spring F / R         {veh.spring_rate_front_Npm:7.0f} / {veh.spring_rate_rear_Npm:7.0f} N/m '
            f'({veh.spring_rate_front_Npm/_LBF:.0f} / {veh.spring_rate_rear_Npm/_LBF:.0f} lbf/in)',
            f'motion ratio F / R   {veh.motion_ratio_front:5.3f} / {veh.motion_ratio_rear:5.3f} (kinematic solve)',
            f'wheel rate F / R     {veh.wheel_rate_front_Npm:7.0f} / {veh.wheel_rate_rear_Npm:7.0f} N/m',
            f'tire rate            {veh.tire_rate_Npm:8.0f} N/m',
            f'RIDE rate F / R      {veh.ride_rate_front_Npm:7.0f} / {veh.ride_rate_rear_Npm:7.0f} N/m',
            f'ride frequency F / R {ff:5.2f} / {fr:5.2f} Hz   rear/front {fr/ff:.3f} (RCVD flat ride ~1.10)',
            f'ARB wheel rate F / R {veh.arb_rate_front_Npm:7.0f} / {veh.arb_rate_rear_Npm:7.0f} N/m',
        ]
        self._model_lbl.setText('\n'.join(lines))
        I = self.inertias()
        if I['source'] == 'estimated':
            self._inertia_lbl.setText(f"Using ESTIMATE: pitch {I['pitch_kgm2']:.0f} kg·m² = {I['pitch_method']}; "
                                      f"roll {I['roll_kgm2']:.0f} kg·m² = {I['roll_method']}. Same estimate the "
                                      'corner-exit model uses.')
        else:
            self._inertia_lbl.setText(f"Using YOUR values: pitch {I['pitch_kgm2']:.0f}, roll {I['roll_kgm2']:.0f} kg·m².")
        for w in (self._pitch_I, self._roll_I) + self._inertia_rows:   # boxes only when measured values are used
            w.setVisible(I['source'] == 'entered')
        try:
            dm = self.damper_model()
            damp = (f'ON: {dm.label}, measured curves at the motion ratio above' if dm is not None
                    else 'PLACEHOLDER straight lines, not this car (no dyno file)')
        except Exception as e:
            damp = f'ERROR: {type(e).__name__}: {e}'
        self._used_lbl.setText(f'Aero: {self._aero_label()}\nDampers: {damp}\n'
                               f"Body inertia: {'measured (you)' if I['source'] == 'entered' else 'estimated'}"
                               ' (box below)\nRoad: made up, ISO 8608 A and B')
        self._refresh_range_lbl()
        self._refresh_banner()
        m = self._model_travel_mm(veh)
        if m:
            warn = ''
            if abs(float(self._margin.value()) - m['margin_mm']) > 0.05:
                warn = (f' THE BOX ({self._margin.value():.1f} mm) IS NOT THE CAR VALUE ({m["margin_mm"]:.1f} mm): '
                        'press "Take travel from the car".')
            self._travel_lbl.setText(warn + 
                f"Model travel at the wheel: front bump {m['front_bump_mm']:.1f} / droop {m['front_droop_mm']:.1f} mm, "
                f"rear bump {m['rear_bump_mm']:.1f} / droop {m['rear_droop_mm']:.1f} mm "
                f"(stroke and preload from the Motion panel). Smallest = {m['margin_mm']:.1f} mm.")

    # ── inputs -> RideInputs ──────────────────────────────────────────────
    def inputs(self, veh=None):
        """RideInputs from the page.  With a dyno damper in the project the
        inputs carry it (WheelDampers at the solved MR) and the four linear
        coefficients become the EQUIVALENT ones of the reference case (middle
        speed, coherent, first seed) — used by the outputs that take a
        single linear set (contact-vs-frequency, launch lag, start of each
        per-case iteration)."""
        base = self._inputs_linear()
        dm = self.damper_model()
        if dm is None:
            return base
        veh = veh or self.vehicle()
        withd = _dc_replace(base, damper=dm.wheel_for_vehicle(veh))
        cases = RS.build_road_cases(withd, veh.wheelbase_m)
        ref = RS.equivalent_damping_for_case(veh, withd, cases[RS.reference_case_index(cases, withd.speeds_mps)])
        return _dc_replace(withd, corner_damping_Nspm=tuple(float(c) for c in ref['c_wheel_Nspm']))

    def _inputs_linear(self):
        n = int(self._v_n.value())
        v0, v1 = float(self._v_min.value()) / 3.6, float(self._v_max.value()) / 3.6
        if v1 < v0:
            raise ValueError('speed max must be at least speed min')
        speeds = tuple(float(v) for v in (np.linspace(v0, v1, n) if n > 1 else [v0]))
        seeds = tuple(range(int(self._seed0.value()), int(self._seed0.value()) + int(self._seed_n.value())))
        inertia = self.inertias()
        return RS.RideInputs(
            pitch_inertia_kgm2=inertia['pitch_kgm2'],
            roll_inertia_kgm2=inertia['roll_kgm2'],
            corner_damping_Nspm=tuple(float(self._damp[c].value()) for c in CORNERS),
            road_class=self._road.currentText(), speeds_mps=speeds, seeds=seeds,
            coherence=self._coh.currentText(),
            stop_margin_m=float(self._margin.value()) / 1000.,
            bump_height_m=float(self._bump_h.value()) / 1000.,
            bump_length_m=float(self._bump_l.value()),
            flat_ride_max_ratio=float(self._flat.value()),
            bump_stop_margin_m=self._bump_margin_m(),
            **self._aero())

    def _bump_margin_m(self):
        """Wheel travel to the bump stop from the model (smaller axle), when
        it exceeds the margin box; else None (= the box, symmetric)."""
        m = self._model_travel_mm()
        if not m:
            return None
        bump = min(m['front_bump_mm'], m['rear_bump_mm'])
        return bump / 1000. if bump > float(self._margin.value()) else None

    def _aero(self):
        """The car's aero package (Dynamics panel F_ref / V_ref / CoP, the
        same one the Build tolerance aero ride-height page uses) when aero is
        applied; otherwise none.  Each road case then runs at its own speed's
        downforce (loads + sink)."""
        main = self._main
        if not getattr(main, '_aero_active', False):
            return {}
        try:
            from gui.build_tolerance_page import aero_package
            pk = aero_package(main)
        except Exception:
            return {}
        return dict(aero_cla_m2=float(pk['cla_m2']), aero_cop_rear=float(pk['cop_rear']),
                    air_density=float(pk['rho']))

    def sweep_grids(self, veh):
        n = int(self._npts.value())
        fr, ff = RS.ride_rate_grid_Npm(veh, 'front', self._ff_min.value(), self._ff_max.value(), n)
        rr, fR = RS.ride_rate_grid_Npm(veh, 'rear', self._fr_min.value(), self._fr_max.value(), n)
        return fr, rr, ff, fR

    # ── actions ───────────────────────────────────────────────────────────
    def analyse_current(self):
        """Road metrics / transfer / PSD / bump for the springs in the model."""
        try:
            veh = self.vehicle()
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                inputs = self.inputs(veh)
                summ = self.summary_study(veh)
                study = RS.current_car_study(veh, inputs)
            finally:
                QApplication.restoreOverrideCursor()
        except Exception as e:
            self._status.setText(f'Analysis failed: {type(e).__name__}: {e}')
            return None
        self.last_study, self.last_summary = study, summ
        self._refresh_damper_info()
        self._plot_summary(summ, veh, summ['inputs'])
        self._plot_metrics(study, inputs)
        self._plot_transfer(study, veh)
        self._plot_psd(study, inputs)
        self._plot_bump(study, inputs)
        for extra in (self._plot_contact, self._plot_launch):
            try:
                extra(veh, inputs)
            except Exception as e:
                self._status.setText(f'{extra.__name__} failed: {type(e).__name__}: {e}')
        a = summ['swing']['A']; i60 = int(np.argmin(np.abs(a['speed_kph'] - 60.)))
        lowest = min(float(np.min(m.min_load_N)) for m in study['metrics'])
        self._status.setText(
            f'Front tyre at {a["speed_kph"][i60]:.0f} km/h on road A: normal load {a["mean_N"][i60, 0]:.0f} N, '
            f'moves ±{a["rms_N"][i60, 0]:.0f} N on average ({a["pct"][i60, 0]:.0f} %).\n'
            f'Detail tabs (road {inputs.road_class}, {", ".join(f"{v*3.6:.0f}" for v in inputs.speeds_mps)} km/h, '
            f'full time run): lowest tyre load {lowest:.0f} N'
            f'{" - the tyre never leaves the road" if lowest > 0 else " - A TYRE LEAVES THE ROAD"}.')
        self._tabs.setCurrentWidget(self._canvases['summary'])
        return study

    def run_solve(self, blocking=False):
        """The ride-rate solve. blocking=True runs inline (tests/headless) and
        returns the selection dict; otherwise a worker thread with progress."""
        try:
            veh = self.vehicle(); inputs = self.inputs(veh)
            fr, rr, ff, fR = self.sweep_grids(veh)
        except Exception as e:
            self._status.setText(f'Solve not started: {type(e).__name__}: {e}')
            return None
        self._solve_veh, self._solve_inputs, self._solve_freqs = veh, inputs, (ff, fR)
        total = len(fr) * len(rr)
        self._progress.setRange(0, total); self._progress.setValue(0)
        if blocking:
            sweep = RS.sweep_ride_rates(veh, inputs, fr, rr,
                                        progress=lambda d, t: self._progress.setValue(d))
            return self._finish(sweep)
        if self._worker is not None and self._worker.isRunning():
            self._status.setText('A solve is already running.')
            return None
        self._worker = _SweepWorker(veh, inputs, fr, rr)
        self._worker.progress.connect(lambda d, t: self._progress.setValue(d))
        self._worker.done.connect(self._finish)
        self._worker.failed.connect(self._failed)
        self._solve_btn.setEnabled(False); self._cancel_btn.setEnabled(True)
        n_cases = len(inputs.seeds) * len(inputs.coherence_values) * len(inputs.speeds_mps)
        self._status.setText(f'Sweeping {total} front x rear pairs over {n_cases} road cases '
                             f'and {len(inputs.speeds_mps)} bump speeds…')
        self._worker.start()
        return None

    def _cancel(self):
        if self._worker is not None:
            self._worker.cancel = True

    def _failed(self, msg):
        self._solve_btn.setEnabled(True); self._cancel_btn.setEnabled(False)
        self._status.setText(f'Solve failed: {msg}')

    def _finish(self, sweep):
        self._solve_btn.setEnabled(True); self._cancel_btn.setEnabled(False)
        veh, inputs = self._solve_veh, self._solve_inputs
        sel = RS.describe_selection(sweep, veh, inputs, **self._motion())
        sel['front_freqs_Hz'], sel['rear_freqs_Hz'] = self._solve_freqs
        self.last_sweep, self.last_selection = sweep, sel
        self._plot_sweep(sweep, sel, inputs)
        self._show_result(sel, inputs, veh)
        self._status.setText(
            f'Solve: {sel["status"]}. Front {sel["front_ride_frequency_Hz"]:.2f} Hz / rear '
            f'{sel["rear_ride_frequency_Hz"]:.2f} Hz -> springs {sel["front_spring_rate_lbf_in"]:.0f} / '
            f'{sel["rear_spring_rate_lbf_in"]:.0f} lbf/in; worst-corner DLC {sel["worst_dlc"]:.3f} '
            f'(grid spans {sel["dlc_grid_min"]:.3f}-{sel["dlc_grid_max"]:.3f}); '
            f'{sel["n_feasible"]}/{sel["n_grid"]} pairs meet travel + flat ride + tyre contact.')
        self._tabs.setCurrentWidget(self._canvases['solve'])
        return sel

    # ── plots ─────────────────────────────────────────────────────────────
    def _plot_sweep(self, sw, sel, inputs):
        fig = self._figs['solve']; fig.clear()
        ff, fR = sel['front_freqs_Hz'], sel['rear_freqs_Hz']
        i, j = sel['i_front'], sel['j_rear']
        speeds = sw.speeds_mps
        axes = fig.subplots(2, 3)
        # (a) worst-corner DLC map
        ax = axes[0, 0]
        im = ax.pcolormesh(ff, fR, sw.worst_dlc.T, cmap='cividis', shading='nearest')
        fig.colorbar(im, ax=ax).ax.tick_params(colors=_MUTED, labelsize=7)
        bad = ~sw.feasible
        if bad.any():
            ii, jj = np.nonzero(bad)
            ax.plot(ff[ii], fR[jj], 'x', color=_ACCENT, ms=5, mew=1.2, label='fails travel, flat ride or tyre contact')
        ax.plot([ff[i]], [fR[j]], 'o', color='#fff', mec=_ACCENT, mew=2, ms=9, label='chosen')
        _style_axes(ax, 'Worst-corner DLC map (lower = steadier tire load)',
                    'front ride frequency (Hz)', 'rear ride frequency (Hz)')
        _legend(ax, loc='lower right', framealpha=0.85)
        # (b)/(c) DLC per corner along each axis at the chosen other axis
        for ax, axis_name in ((axes[0, 1], 'front'), (axes[0, 2], 'rear')):
            for s, v in enumerate(speeds):
                cases = sw.cases_at_speed(v)
                for c, corner in enumerate(CORNERS):
                    # index in two steps: a slice between advanced indices would
                    # move the case axis to the front
                    if axis_name == 'front':
                        y = sw.dlc[:, j][:, cases, c].max(axis=1); x = ff
                    else:
                        y = sw.dlc[i][:, cases, c].max(axis=1); x = fR
                    label = None
                    if s == 0:
                        label = corner
                    if c == 0 and s > 0:
                        label = f'{v*3.6:.0f} km/h ({_STYLES[s % 4]})'
                    ax.plot(x, y, _STYLES[s % 4], color=_CC[corner], lw=1.3, label=label)
            other = f'rear {fR[j]:.2f}' if axis_name == 'front' else f'front {ff[i]:.2f}'
            _style_axes(ax, f'DLC per corner vs {axis_name} ({other} Hz held)',
                        f'{axis_name} ride frequency (Hz)', 'DLC = RMS(Fz - mean) / mean')
            _legend(ax, ncol=2)
        # (d) body heave acceleration, (e) travel usage, (f) flat ride
        for ax, key, title, ylab, limit in (
                (axes[1, 0], 'acc', 'Body heave acceleration RMS (comfort)', 'm/s² RMS', None),
                (axes[1, 1], 'travel', 'Worst-corner travel usage (peak / margin)', 'fraction of travel to the stop', 1.0),
                (axes[1, 2], 'flat', 'Flat ride: pitch/bounce after the bump', 'RMS pitch at axle / RMS bounce (mean over speeds gated)', sw.flat_ride_max_ratio)):
            for s, v in enumerate(speeds):
                cases = sw.cases_at_speed(v)
                if key == 'acc':
                    yf = sw.body_heave_acc_rms_mps2[:, j][:, cases].max(axis=1)
                    yr = sw.body_heave_acc_rms_mps2[i][:, cases].max(axis=1)
                elif key == 'travel':
                    yf = sw.travel_usage[:, j][:, cases].max(axis=(1, 2))
                    yr = sw.travel_usage[i, :][:, cases].max(axis=(1, 2))
                else:
                    yf = sw.pitch_to_bounce[:, j, s]; yr = sw.pitch_to_bounce[i, :, s]
                col = _SERIES[s % 4]
                ax.plot(ff, yf, '-', color=col, lw=1.3, label=f'vs front @ {v*3.6:.0f} km/h')
                ax.plot(fR, yr, '--', color=col, lw=1.3, label=f'vs rear @ {v*3.6:.0f} km/h')
            if limit is not None:
                ax.axhline(limit, color=_ACCENT, lw=1, ls=':', label='limit')
            _style_axes(ax, title, 'ride frequency of the swept axle (Hz)', ylab)
            _legend(ax, ncol=2)
        tail = '' if sel['status'] == 'feasible' else f' — {sel["status"]}'
        fig.suptitle(f'Ride-rate sweep — class {inputs.road_class}, {len(sw.case_labels)} road cases, '
                     f'{sel["n_feasible"]}/{sel["n_grid"]} pairs meet travel + flat ride + tyre contact{tail}',
                     color=_INK, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        self._canvases['solve'].draw_idle()

    def _show_result(self, sel, inputs, veh):
        mo = self._motion()
        text = [
            f'STATUS: {sel["status"]}',
            f'Objective: worst-corner DLC {sel["worst_dlc"]:.4f} at the chosen pair; the whole grid spans '
            f'{sel["dlc_grid_min"]:.4f} to {sel["dlc_grid_max"]:.4f} (class {inputs.road_class}, '
            f'{len(inputs.speeds_mps)} speeds, {len(inputs.seeds)} seed(s), coherence "{inputs.coherence}").',
            f'Constraints at the chosen pair: travel usage {sel["worst_travel_usage"]:.2f} of the '
            f'{inputs.stop_margin_m*1000:.0f} mm margin ({"ok" if sel["travel_ok"] else "HITS THE STOP"}); '
            f'flat ride pitch/bounce mean {sel["mean_pitch_to_bounce"]:.2f} / worst speed '
            f'{sel["worst_pitch_to_bounce"]:.2f} vs limit {inputs.flat_ride_max_ratio:.2f} '
            f'({"ok" if sel["flat_ride_ok"] else "NOT MET"}); '
            f'lowest tyre load {sel["worst_min_load_N"]:.0f} N, samples off the road '
            f'{100 * sel["worst_contact_loss_fraction"]:.1f} % '
            f'({"ok" if sel["contact_ok"] else "TYRE LEAVES THE ROAD"}).',
            '',
            f'CHOSEN RIDE RATE   front {sel["front_ride_rate_Npm"]:8.0f} N/m = {sel["front_ride_frequency_Hz"]:.2f} Hz    '
            f'rear {sel["rear_ride_rate_Npm"]:8.0f} N/m = {sel["rear_ride_frequency_Hz"]:.2f} Hz    '
            f'rear/front {sel["flat_ride_ratio"]:.3f} (RCVD ~1.10)',
            f'-> WHEEL RATE      front {sel["front_wheel_rate_Npm"]:8.0f} N/m    rear {sel["rear_wheel_rate_Npm"]:8.0f} N/m'
            f'    (tire {veh.tire_rate_Npm:.0f} N/m in series)',
            f'-> SPRING RATE     front {sel["front_spring_rate_Npm"]:8.0f} N/m = {sel["front_spring_rate_lbf_in"]:6.1f} lbf/in'
            f'    rear {sel["rear_spring_rate_Npm"]:8.0f} N/m = {sel["rear_spring_rate_lbf_in"]:6.1f} lbf/in'
            f'    through motion ratio {sel["motion_ratio_front"]:.3f} / {sel["motion_ratio_rear"]:.3f} '
            f'(VehicleParams wheel-rate relation incl. its geometric term)',
            f'CURRENTLY IN MODEL ride {sel["current_front_ride_rate_Npm"]:.0f} / {sel["current_rear_ride_rate_Npm"]:.0f} N/m, '
            f'springs {sel["current_front_spring_rate_Npm"]/_LBF:.0f} / {sel["current_rear_spring_rate_Npm"]/_LBF:.0f} lbf/in',
            f'Sprung corner masses front {sel["sprung_corner_mass_front_kg"]:.1f} / rear {sel["sprung_corner_mass_rear_kg"]:.1f} kg; '
            f'preload front {mo["preload_front_mm"]:.1f} / rear {mo["preload_rear_mm"]:.1f} mm, stroke {mo["stroke_mm"]:.0f} mm (Motion panel).',
            'Catalogue rows: preload = collar compression (mm of spring) that puts the car at the SAME ride height '
            'the exact spring would; negative = that spring already sits higher at zero preload.',
        ]
        self._result_lbl.setText('\n'.join(text))
        for axle in RS.AXLES:
            t = self._spring_tables[axle]; rows = sel[f'standard_springs_{axle}']
            t.setRowCount(len(rows))
            for r, row in enumerate(rows):
                vals = (f'{row["spring_lbf_in"]:.0f}', f'{row["spring_Npm"]:.0f}', f'{row["wheel_rate_Npm"]:.0f}',
                        f'{row["ride_rate_Npm"]:.0f}', f'{row["ride_frequency_Hz"]:.2f}',
                        f'{row["ride_rate_deviation_pct"]:+.1f}', f'{row["preload_to_hold_ride_height_mm"]:+.1f}',
                        f'{row["sag_shock_mm_at_that_preload"]:.1f}', f'{row["ride_height_change_at_wheel_mm"]:+.1f}')
                best = abs(row['ride_rate_deviation_pct']) == min(abs(x['ride_rate_deviation_pct']) for x in rows)
                for c, v in enumerate(vals):
                    it = QTableWidgetItem(v); it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    if best:
                        it.setForeground(Qt.GlobalColor.white)
                        f = it.font(); f.setBold(True); it.setFont(f)
                    t.setItem(r, c, it)

    def _spring_top_axis(self, ax, veh, axle, freqs):
        """Second x axis on top: the spring (lbf/in) each ride frequency
        needs on THIS car (VehicleParams' own ride -> spring relation)."""
        mass = RS.sprung_corner_masses_kg(veh)[0 if axle == 'front' else 1]
        ticks = [float(f) for f in freqs[::2]]
        labels = []
        for f in ticks:
            try:
                labels.append(f'{RS.spring_rate_for_ride_rate_Npm(veh, axle, RS.ride_rate_from_frequency_Npm(f, mass)) / _LBF:.0f}')
            except ValueError:
                labels.append('-')
        top = ax.twiny(); top.set_xlim(ax.get_xlim()); top.set_xticks(ticks); top.set_xticklabels(labels)
        top.tick_params(colors=_MUTED, labelsize=7)
        top.set_xlabel(f'{axle} spring on this car (lbf/in)', color=_MUTED, fontsize=8)

    def summary_inputs(self, veh=None):
        """The inputs the SUMMARY uses: the page inputs with both tracks
        (left/right same and unrelated, worse kept)."""
        return _dc_replace(self.inputs(veh), coherence='both')

    def _plot_summary(self, summ, veh, inputs):
        """Headline: tyre-load swing against SPEED on road A and B, front and
        rear, with the independent one-corner check."""
        fig = self._figs['summary']; fig.clear()
        gs = fig.add_gridspec(1, 3, width_ratios=[1., 1., 0.9])
        ax_n, ax_p, ax_t = (fig.add_subplot(gs[0, i]) for i in range(3))
        for cls, ls in (('A', '-'), ('B', '--')):
            d = summ['swing'][cls]; v = d['speed_kph']
            for c, name, col in ((0, 'front', _CC['FL']), (2, 'rear', _CC['RL'])):
                ax_n.plot(v, d['rms_N'][:, c], ls, color=col, lw=1.8, marker='o', ms=3, label=f'{name}, road {cls}')
                ax_p.plot(v, d['pct'][:, c], ls, color=col, lw=1.8, marker='o', ms=3, label=f'{name}, road {cls}')
            ax_n.plot(v, summ['check'][cls], 'x', color=_CC['RR'], ms=7, mew=1.6,
                      label='hand check, front' if cls == 'A' else None)
        _style_axes(ax_n, 'Average tyre-load swing, one tyre', 'car speed (km/h)', 'N  (up and down from normal)')
        _style_axes(ax_p, 'Same, as % of the normal tyre load (downforce included)', 'car speed (km/h)', '% of normal load')
        ax_n.set_ylim(bottom=0); ax_p.set_ylim(bottom=0)
        _legend(ax_n, loc='upper left'); _legend(ax_p, loc='upper left')
        ax_t.set_axis_off(); ax_t.set_facecolor(_PANEL)
        a = summ['swing']['A']; b = summ['swing']['B']
        i60 = int(np.argmin(np.abs(a['speed_kph'] - 60.)))
        txt = ('HOW TO READ THIS\n\n'
               'On a bumpy road the tyre load never sits still.\n'
               'The lines are the AVERAGE amount it moves\n'
               'away from normal. For a moment now and then it\n'
               'goes 2-3x further.\n'
               'Lower = steadier tyre = more grip on average.\n'
               '(The old "160-200 N" was road B at\n'
               '72-108 km/h - the dashed lines.)\n\n'
               f'At {a["speed_kph"][i60]:.0f} km/h, front tyre:\n'
               f'  normal load {a["mean_N"][i60, 0]:.0f} N\n'
               f'  road A  ±{a["rms_N"][i60, 0]:.0f} N  ({a["pct"][i60, 0]:.0f} %)\n'
               f'  road B  ±{b["rms_N"][i60, 0]:.0f} N  ({b["pct"][i60, 0]:.0f} %)\n\n'
               'Road A = very good public road, B = rough\n'
               'public road. Nobody has measured an FSAE track,\n'
               'so trust the CHANGE between two setups more\n'
               'than the exact newtons.\n\n'
               'Blue x = hand check: one corner on its own,\n'
               'written separately from the full car model.\n'
               'Right plot drops after 80 km/h: downforce\n'
               'grows faster than the swing.\n\n'
               f'Aero: {"ON" if inputs.aero_cla_m2 > 0 else "OFF - no downforce"}\n'
               f'Dampers: {"dyno file" if inputs.damper is not None else "PLACEHOLDER"}\n'
               f'Body inertia: {self.inertias()["source"]}')
        ax_t.text(0.0, 1.0, txt, transform=ax_t.transAxes, va='top', ha='left', color=_INK, fontsize=8,
                  family='monospace')
        iw = int(np.argmax(b['rms_N'][:, 2]))
        lowA, lowB = float(np.min(a['min_N'])), float(np.min(b['min_N']))
        fig.suptitle(f'WORST CASE: rear tyre, rough road B, {b["speed_kph"][iw]:.0f} km/h - its load swings ±{b["rms_N"][iw, 2]:.0f} N '
                     f'on average ({b["pct"][iw, 2]:.0f} % of {b["mean_N"][iw, 2]:.0f} N). Same tyre on road A: '
                     f'±{a["rms_N"][iw, 2]:.0f} N ({a["pct"][iw, 2]:.0f} %).\n'
                     f'Lowest tyre load in these runs (any corner, 20-120 km/h): {lowA:.0f} N on road A, {lowB:.0f} N on road B'
                     f'{" - the tyre never leaves the road" if min(lowA, lowB) > 0 else " - A TYRE LEAVES THE ROAD"}. '
                     'No pass/fail limit is set for the swing: use it to COMPARE setups.',
                     color=_INK, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        self._canvases['summary'].draw_idle()

    def summary_study(self, veh=None):
        """Tyre-load swing vs speed (vahan.ride_solve.load_swing_vs_speed)
        plus the one-corner hand check per speed (front corner, same damping)."""
        veh = veh or self.vehicle()
        inputs = self.summary_inputs(veh)
        speeds = np.asarray(SUMMARY_SPEEDS_KPH) / 3.6
        swing = RS.load_swing_vs_speed(veh, inputs, speeds)
        mf, _ = RS.sprung_corner_masses_kg(veh)
        check = {cls: np.array([RS.quarter_car_load_rms_N(mf, veh.unsprung_mass_front_kg / 2.,
                                                          veh.wheel_rate_front_Npm, d['c_wheel_Nspm'][i, 0],
                                                          veh.tire_rate_Npm, cls, v)
                                for i, v in enumerate(speeds)])
                 for cls, d in swing.items()}
        return {'swing': swing, 'check': check, 'inputs': inputs}

    def _plot_metrics(self, study, inputs):
        fig = self._figs['metrics']; fig.clear()
        metrics, cases = study['metrics'], study['cases']
        n = len(cases); x = np.arange(n); wbar = 0.2
        axes = fig.subplots(2, 3)
        labels = [f'{c.speed_mps*3.6:.0f} km/h\n{"same L+R" if c.coherence == "coherent" else "diff L+R"}'
                  + (f' #{c.seed}' if len({k.seed for k in cases}) > 1 else '')
                  for c in cases]
        panels = (
            (axes[0, 0], 'dlc', 'Tyre-load swing as a share of normal load', 'share (0.10 = 10 %)', None),
            (axes[0, 1], 'min_load_N', 'Lowest tyre load in the run', 'N (0 = tyre leaves the road)', 0.),
            (axes[0, 2], 'contact_loss_fraction_linear', 'Share of time the tyre is off the road', 'fraction', None),
            (axes[1, 0], 'travel_usage', 'Wheel travel used (biggest / travel to the stop)', 'fraction (1 = hits the stop)', 1.),
            (axes[1, 1], 'damper_velocity_rms_mps', 'Damper speed: average (bars), biggest (marks)', 'm/s at the damper', None),
        )
        for ax, key, title, ylab, limit in panels:
            top = 0.
            for c, corner in enumerate(CORNERS):
                vals = [getattr(m, key)[c] for m in metrics]
                top = max(top, max(vals))
                ax.bar(x + (c - 1.5) * wbar, vals, wbar, color=_CC[corner], label=corner)
                if key == 'damper_velocity_rms_mps':
                    peaks = [m.damper_velocity_peak_mps[c] for m in metrics]
                    top = max(top, max(peaks))
                    ax.plot(x + (c - 1.5) * wbar, peaks, 'v', color=_CC[corner], mec='#000', ms=5)
            if limit is not None:
                ax.axhline(limit, color=_ACCENT, lw=1, ls=':')
            if key == 'contact_loss_fraction_linear':
                ax.set_ylim(0, max(0.01, top * 1.3))
                if top == 0.:
                    ax.text(0.5, 0.5, 'the tyre never leaves the road\n(load stays above 0 in every case)',
                            transform=ax.transAxes, ha='center', va='center', color=_MUTED, fontsize=8)
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6)
            _style_axes(ax, title, '', ylab); _legend(ax, ncol=4)
        # pitch/roll accelerations expressed as vertical acceleration at the
        # axle / at the wheel track (model geometry), so one m/s² axis.
        bcm = study['model'].body_corner_map
        half_wb = 0.5 * float(bcm[0, 1] - bcm[2, 1]); half_track = float(abs(bcm[0, 2]))
        ax = axes[1, 2]
        ax.bar(x - wbar, [m.body_heave_acc_rms_mps2 for m in metrics], wbar, color=_SERIES[2], label='heave at CG')
        ax.bar(x, [m.body_pitch_acc_rms_radps2 * half_wb for m in metrics], wbar, color=_SERIES[0],
               label=f'pitch, at the axle (x {half_wb:.2f} m)')
        ax.bar(x + wbar, [m.body_roll_acc_rms_radps2 * half_track for m in metrics], wbar, color=_SERIES[1],
               label=f'roll, at the wheel (x {half_track:.2f} m)')
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6)
        _style_axes(ax, 'Body shaking: bounce, pitch, roll (average)', '', 'm/s²'); _legend(ax)
        fig.suptitle(f'Every road case, current springs — made-up road {inputs.road_class} (ISO 8608), '
                     f'{inputs.record_length_m:.0f} m of road per run, {self._margin_words(inputs)}',
                     color=_INK, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        self._canvases['metrics'].draw_idle()

    def _plot_transfer(self, study, veh):
        fig = self._figs['transfer']; fig.clear()
        tr = study['transfer']; f = tr['frequency_Hz']
        axes = fig.subplots(2, 3)
        for ax, key, title, ylab in (
                (axes[0, 0], 'heave_m_per_m', 'Body heave / road (4 wheels in phase)', 'm per m'),
                (axes[0, 1], 'pitch_rad_per_m', 'Body pitch / road (front +, rear -)', 'rad per m'),
                (axes[0, 2], 'roll_rad_per_m', 'Body roll / road (left +, right -)', 'rad per m'),
                (axes[1, 0], 'wheel_m_per_m', 'FL wheel motion per FL road', 'm per m'),
                (axes[1, 1], 'load_N_per_m', 'FL dynamic tire load per FL road', 'N per m')):
            ax.semilogx(f, tr[key], color=_SERIES[2], lw=1.4)
            _style_axes(ax, title, 'bumps hitting the wheel per second (Hz)', ylab)
        ax = axes[1, 2]; ax.set_axis_off(); ax.set_facecolor(_PANEL)
        mf, mr = RS.sprung_corner_masses_kg(veh)
        hop_f = np.sqrt((veh.wheel_rate_front_Npm + veh.tire_rate_Npm) / (veh.unsprung_mass_front_kg / 2)) / (2 * np.pi)
        hop_r = np.sqrt((veh.wheel_rate_rear_Npm + veh.tire_rate_Npm) / (veh.unsprung_mass_rear_kg / 2)) / (2 * np.pi)
        ax.text(0.02, 0.95,
                'Undamped frequencies from the model\n\n'
                f'front ride   {study["front_ride_frequency_Hz"]:.2f} Hz  (ride rate {veh.ride_rate_front_Npm:.0f} N/m on {mf:.1f} kg)\n'
                f'rear ride    {study["rear_ride_frequency_Hz"]:.2f} Hz  (ride rate {veh.ride_rate_rear_Npm:.0f} N/m on {mr:.1f} kg)\n'
                f'rear/front   {study["flat_ride_ratio"]:.3f}  (RCVD flat ride ~1.10)\n'
                f'front wheel hop {hop_f:.1f} Hz   rear wheel hop {hop_r:.1f} Hz\n\n'
                'Dampers, inertia and road: see WHAT THIS PAGE USES (left).',
                transform=ax.transAxes, va='top', ha='left', color=_INK, fontsize=8, family='monospace')
        fig.suptitle('Transfer functions of the 7-DOF model (heave, pitch, roll + 4 wheels)', color=_INK, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        self._canvases['transfer'].draw_idle()

    def _plot_contact(self, veh, inputs):
        """Contact-patch load deviation as each axle's ride frequency is varied (other axle as modelled)."""
        fig = self._figs['contact']; fig.clear()
        freqs = np.linspace(1.5, 3.5, 9)
        fixed = RS.contact_patch_vs_ride_frequency(veh, inputs, freqs)
        ratio = RS.contact_patch_vs_ride_frequency(veh, inputs, freqs, hold_damping_ratio=True)
        self.last_contact = {'fixed_damper': fixed, 'same_damping_ratio': ratio}
        axes = fig.subplots(2, 2)
        for r, axle in enumerate(RS.AXLES):
            d = fixed[axle]; ax = axes[r, 0]
            for (f_hz, curve), col, ls in zip(sorted(d['load_per_mm_road_N'].items()),
                                              ('#9A9AA2', '#FFFFFF', '#FFD600', '#E53935', '#42A5F5'), (':', '-', '--', '-.', ':')):
                now = abs(f_hz - d['current_Hz']) < 2e-3
                ax.plot(fixed['transfer_frequency_Hz'], curve, color='#FFFFFF' if now else col, ls='-' if now else ls,
                        lw=1.8 if now else 1.3, label=f'{f_hz:.2f} Hz' + (' (in the model)' if now else ''))
            _style_axes(ax, f'{axle.upper()}: tyre load change per mm of bump (lower = steadier)', _HZ_LABEL, 'N per mm')
            _legend(ax)
            ax = axes[r, 1]
            ax.plot(freqs, d['rms_pct_of_static'], color='#FFFFFF', lw=1.8, marker='o', ms=3, label='dampers as now')
            ax.plot(freqs, ratio[axle]['rms_pct_of_static'], color='#FFD600', lw=1.6, ls='--', marker='o', ms=3,
                    label='dampers re-adjusted to match the new spring')
            ax.axvline(d['current_Hz'], color='#E53935', lw=1.1, ls=':', label=f'in the model {d["current_Hz"]:.2f} Hz')
            _style_axes(ax, f'{axle.upper()}: average tyre-load swing, road {inputs.road_class}, mean of '
                            f'{", ".join(f"{v*3.6:.0f}" for v in inputs.speeds_mps)} km/h\n'
                            f'(% of the tyre load with downforce at that speed; at rest {d["static_load_N"]:.0f} N)', f'{axle} ride frequency (Hz) - stiffer spring to the right',
                        '% of static load')
            _legend(ax)
            self._spring_top_axis(ax, veh, axle, freqs)
        span = {ax_: float(np.ptp(fixed[ax_]['rms_pct_of_static'])) for ax_ in RS.AXLES}
        fig.suptitle(f'Tyre load vs spring choice - made-up road {inputs.road_class}. TAKEAWAY: over the whole spring range the average '
                     f'swing changes by only {span["front"]:.1f} (front) / {span["rear"]:.1f} (rear) points of %.\n'
                     'Top axis = the spring that ride frequency needs on this car. Left: the hump near 20 Hz is the wheel '
                     'bouncing on its own tyre (wheel hop); springs barely move it.',
                     color=_INK, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        self._canvases['contact'].draw_idle()

    def _plot_launch(self, veh, inputs):
        """How fast the rear tyres receive the launch load transfer, vs rear anti-squat (chassis-mounted diff)."""
        from vahan import metrics_catalog as MC
        fig = self._figs['launch']; fig.clear()
        rear = self._main._solvers['RL'].solve(0.); front = self._main._solvers['FL'].solve(0.)
        mass = float(veh.sprung_mass_kg + veh.unsprung_mass_front_kg + veh.unsprung_mass_rear_kg)
        brk = MC.anti_squat_breakdown(rear, cg_height_m=veh.cg_height_m, wheelbase_m=veh.wheelbase_m,
                                      total_mass_kg=mass, rear_ride_rate_Npm=veh.ride_rate_rear_Npm)
        if brk is None:
            raise ValueError('no rear side-view instant centre')
        study = RS.launch_lag_study(veh, inputs, rear_slope_from_wheel_centre=brk['tan_from_wheel_centre'],
                                    front_slope_from_wheel_centre=MC._sv_ic_coeff(front, 'wheel_centre'),
                                    rear_wheel_centre_height_m=float(rear.wheel_center[2]),
                                    front_wheel_centre_height_m=float(front.wheel_center[2]))
        self.last_launch = {'breakdown': brk, 'study': study}
        rises = sorted({c['thrust_rise_time_s'] for c in study['cases']})
        gs = fig.add_gridspec(1, len(rises) + 1, width_ratios=[1.0] * len(rises) + [0.95])
        axes = [fig.add_subplot(gs[0, i]) for i in range(len(rises) + 1)]
        look = {'as built': ('#FFFFFF', '-'), '0 %': ('#9A9AA2', ':'), '30 %': ('#FFD600', '--'),
                '60 %': ('#E53935', '-.'), '100 %': ('#42A5F5', ':')}
        for ax, rise in zip(axes, rises):
            for c in (c for c in study['cases'] if c['thrust_rise_time_s'] == rise):
                col, ls = look.get(c['label'], ('#FFFFFF', '-'))
                lab = (f'as built ({c["anti_squat_pct"]:+.0f} %)' if c['label'] == 'as built' else c['label'])
                ax.plot(c['time_s'] * 1000, c['rear_wheel_load_gain_N'], color=col, ls=ls, lw=1.6,
                        label=f'{lab}: half the load at {c["t50_s"]*1000:.0f} ms')
            ax.plot(c['time_s'] * 1000, c['thrust_fraction'] * c['final_load_gain_N'], color='#9A9AA2', lw=0.8,
                    label='load the thrust asks for')
            ax.set_xlim(0, 500)
            _style_axes(ax, 'thrust applied instantly' if rise == 0 else f'thrust builds over {rise*1000:.0f} ms',
                        'time after thrust starts (ms)', 'extra load on ONE rear tyre (N)' if rise == rises[0] else '')
            _legend(ax, loc='lower right')
        ax = axes[-1]; ax.set_axis_off(); ax.set_facecolor(_PANEL)
        ax.text(0.0, 0.98,
                'Rear anti-squat, differential on the chassis\n(RCVD p.617-619, fig 17.15b)\n\n'
                f'load moved to the rear at {brk["accel_g"]:.1f} g\n'
                f' = mass x accel x CG height / wheelbase\n'
                f' = {brk["total_mass_kg"]:.1f} kg x {brk["cg_height_m"]*1000:.0f} / {brk["wheelbase_m"]*1000:.0f} mm\n'
                f' = {brk["load_transfer_axle_N"]:.0f} N ({brk["load_transfer_per_wheel_N"]:.0f} N per wheel)\n'
                ' anti-squat does NOT change this number\n\n'
                f'side-view instant centre\n {brk["ic_ahead_of_axle_m"]*1000:.0f} mm ahead of the rear axle\n'
                f' {brk["ic_height_m"]*1000:.0f} mm high (wheel centre {brk["wheel_centre_height_m"]*1000:.0f})\n'
                f'slope from the WHEEL CENTRE {brk["tan_from_wheel_centre"]:+.4f}\n'
                f' -> anti-squat {brk["anti_squat_pct_wheel_centre"]:+.1f} %\n'
                f'(contact-patch line {brk["anti_squat_pct_contact_patch"]:+.1f} %:\n live axle only)\n\n'
                f'body squat at the rear axle {brk["squat_m_as_designed"]*1000:.1f} mm\n'
                f' 30 % -> {brk["squat_m_by_anti_pct"]["30"]*1000:.1f} mm, 60 % -> {brk["squat_m_by_anti_pct"]["60"]*1000:.1f} mm\n\n'
                'Not modelled: wheel spin-up inertia,\ntyre relaxation. Damping and pitch\ninertia are user inputs.',
                transform=ax.transAxes, va='top', ha='left', color=_INK, fontsize=7.5, family='monospace', clip_on=False)
        fig.suptitle('Launch: how fast the rear tyre receives its load, by rear anti-squat', color=_INK, fontsize=10)
        fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.1, wspace=0.22)
        self._canvases['launch'].draw_idle()

    def _plot_psd(self, study, inputs):
        fig = self._figs['psd']; fig.clear()
        axes = fig.subplots(2, 2)
        spectrum = RoadSpectrum(inputs.road_class)
        speeds = sorted({k[0] for k in study['psd']})
        for s, v in enumerate(speeds):
            col = _SERIES[s % 4]
            for t, coh in enumerate(inputs.coherence_values):
                f, out = study['psd'][(v, coh)]
                ls = _STYLES[t % 4]
                lab = f'{v*3.6:.0f} km/h {coh}'
                if t == 0:
                    good = f > 0
                    axes[0, 0].loglog(f[good], spectrum.temporal_psd(f[good], v), '-', color=col, lw=1.2, label=f'{v*3.6:.0f} km/h')
                axes[0, 1].loglog(f, out['body_acceleration']['psd'][:, 0], ls, color=col, lw=1.2, label=lab)
                axes[1, 0].loglog(f, out['suspension_displacement_m']['psd'][:, 0], ls, color=col, lw=1.2, label=lab)
                axes[1, 1].loglog(f, out['dynamic_tire_load_N']['psd'][:, 0], ls, color=col, lw=1.2, label=lab)
        _style_axes(axes[0, 0], f'Road input PSD at each speed (class {inputs.road_class})', 'bumps hitting the wheel per second (Hz)', 'm² per Hz')
        _style_axes(axes[0, 1], 'Body heave acceleration PSD', 'bumps hitting the wheel per second (Hz)', '(m/s²)² per Hz')
        _style_axes(axes[1, 0], 'FL wheel-travel PSD', 'bumps hitting the wheel per second (Hz)', 'm² per Hz')
        _style_axes(axes[1, 1], 'FL dynamic tire-load PSD', 'bumps hitting the wheel per second (Hz)', 'N² per Hz')
        for ax in axes.ravel():
            _legend(ax)
        fig.suptitle('Spectral road response (solid = coherent tracks, dashed = independent)', color=_INK, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        self._canvases['psd'].draw_idle()

    def _plot_bump(self, study, inputs):
        fig = self._figs['bump']; fig.clear()
        bumps = study['bumps']; n = len(bumps)
        axes = fig.subplots(n, 1, squeeze=False)[:, 0]
        for ax, (v, b) in zip(axes, sorted(bumps.items())):
            t = b['time_s']; tmax = min(t[-1], b['window_start_s'] + 2.5)
            m = t <= tmax
            ax.plot(t[m], b['road_front_m'][m] * 1000, color=_MUTED, lw=0.8, label='road under front wheels')
            ax.plot(t[m], b['body_at_front_axle_m'][m] * 1000, color=_CC['FL'], lw=1.3, label='body over front axle')
            ax.plot(t[m], b['body_at_rear_axle_m'][m] * 1000, color=_CC['FR'], lw=1.3, label='body over rear axle')
            ax.plot(t[m], (0.5 * (b['body_at_front_axle_m'] - b['body_at_rear_axle_m']))[m] * 1000,
                    color=_CC['RR'], lw=1.0, ls='--', label='pitch as motion at the axle')
            ax.axvline(b['window_start_s'], color=_ACCENT, lw=0.8, ls=':', label='rear wheels leave bump')
            _style_axes(ax, f'{v*3.6:.0f} km/h: after the bump the body rocks nose-to-tail {b["pitch_to_bounce_ratio"]:.1f}x as much as it '
                            f'bounces (your limit {inputs.flat_ride_max_ratio:.1f}x: {"ok, lands flat" if b["flat_ride_ok"] else "rocks more than the limit"}); '
                            f'front settle {b["front_settle_s"]:.2f} s, rear {b["rear_settle_s"]:.2f} s from own entry',
                        'time from front wheels entering the bump (s)', 'mm')
            _legend(ax, ncol=5)
        fig.suptitle(f'Occasional bump {inputs.bump_height_m*1000:.0f} mm x {inputs.bump_length_m:.2f} m, '
                     'current springs, both wheels of an axle, rear delayed by wheelbase/speed', color=_INK, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        self._canvases['bump'].draw_idle()
