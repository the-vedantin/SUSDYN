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

Colours: graph LINES use the in-app corner palette (yellow/red/white/blue);
text follows docs/DESIGN.md (no yellow text, accent red for emphasis).
"""
from __future__ import annotations

import numpy as np
from gui.plot_dialog import ReadableCanvas
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QComboBox,
    QDoubleSpinBox, QSpinBox, QPushButton, QGroupBox, QTabWidget, QScrollArea,
    QProgressBar, QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QSplitter, QApplication,
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

# Car-dict keys + placeholder defaults. The three ASSUMED groups are round
# placeholders on purpose — the page shouts until the user confirms them.
DEFAULTS = {
    'ride_pitch_inertia_kgm2': 100.0,
    'ride_roll_inertia_kgm2': 40.0,
    'ride_damping_wheel_Nspm': [1000.0, 1000.0, 1000.0, 1000.0],
    'ride_inputs_confirmed': False,
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
}
_ASSUMED_KEYS = ('ride_pitch_inertia_kgm2', 'ride_roll_inertia_kgm2', 'ride_damping_wheel_Nspm')


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
        self._assumed_labels = []
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
        scroll = QScrollArea(); scroll.setWidget(left_w); scroll.setWidgetResizable(True)
        scroll.setFixedWidth(360); scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        root.addWidget(scroll)

        # ── right: warning banner + tabs ───────────────────────────────────
        right = QVBoxLayout(); right.setSpacing(6)
        self._banner = QLabel(''); self._banner.setWordWrap(True)
        right.addWidget(self._banner)
        self._tabs = QTabWidget()
        self._figs, self._canvases = {}, {}
        for key, title in (('solve', 'Ride-rate solve'), ('result', 'Solve result'),
                           ('metrics', 'Current car: road metrics'),
                           ('transfer', 'Transfer functions'), ('psd', 'Road-response PSDs'),
                           ('bump', 'Occasional bump'),
                           ('contact', 'Contact patch vs ride Hz'),
                           ('launch', 'Launch load lag (anti-squat)')):
            if key == 'result':
                self._tabs.addTab(self._build_result_tab(), title)
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
        root.addLayout(right, 1)

        self._load_saved()
        self._refresh_model_readout()
        self._refresh_banner()

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

    def _row(self, grid, r, text, widget, assumed=False):
        lab = QLabel(text); lab.setStyleSheet(f'color:{_INK}')
        grid.addWidget(lab, r, 0); grid.addWidget(widget, r, 1)
        if assumed:
            self._assumed_labels.append((lab, text))
        return lab

    def _build_inputs(self, left):
        d = DEFAULTS
        # model readout (ONE MODEL — every number here comes from the solved car)
        box = QGroupBox('FROM THE SOLVED MODEL'); lay = QVBoxLayout(box)
        self._model_lbl = QLabel(''); self._model_lbl.setWordWrap(True)
        self._model_lbl.setStyleSheet(f"font-family:Consolas,'SF Mono',monospace;font-size:11px;color:{_INK}")
        lay.addWidget(self._model_lbl)
        rb = QPushButton('Re-read model'); rb.clicked.connect(self._refresh_model_readout)
        lay.addWidget(rb)
        left.addWidget(box)

        # user-owned physics (ASSUMED until confirmed)
        box = QGroupBox('CAR INPUTS NOT IN THE MODEL'); g = QGridLayout(box); g.setSpacing(4)
        self._pitch_I = self._spin(1, 2000, d['ride_pitch_inertia_kgm2'], 1, 5, ' kg·m²',
                                   'Sprung-mass pitch inertia about the sprung CG. Sets how much the '
                                   'body pitches per unit front/rear load difference.')
        self._roll_I = self._spin(1, 2000, d['ride_roll_inertia_kgm2'], 1, 5, ' kg·m²',
                                  'Sprung-mass roll inertia about the sprung CG (roll mode frequency).')
        self._damp = {}
        r = 0
        self._row(g, r, 'Pitch inertia (ASSUMED)', self._pitch_I, assumed=True); r += 1
        self._row(g, r, 'Roll inertia (ASSUMED)', self._roll_I, assumed=True); r += 1
        for i, c in enumerate(CORNERS):
            self._damp[c] = self._spin(0, 20000, d['ride_damping_wheel_Nspm'][i], 0, 50, ' N·s/m',
                                       f'{c} equivalent LINEAR damping at the WHEEL (shock coefficient x MR²). '
                                       'Bump/rebound asymmetry must be pre-averaged by you.')
            self._row(g, r, f'{c} damping at wheel (ASSUMED)', self._damp[c], assumed=True); r += 1
        self._confirm = QCheckBox('these are this car\'s values (clears ASSUMED)')
        self._confirm.setStyleSheet(f'color:{_INK};font-size:12px')
        self._confirm.toggled.connect(self._store); self._confirm.toggled.connect(self._refresh_banner)
        g.addWidget(self._confirm, r, 0, 1, 2)
        left.addWidget(box)

        # road + speed bracket
        box = QGroupBox('ROAD CASES'); g = QGridLayout(box); g.setSpacing(4)
        self._road = QComboBox(); self._road.addItems(['A', 'B'])
        self._road.setToolTip('ISO 8608 class-centre roughness: A = 16e-6 m³ at 0.1 cycles/m, '
                              'B = 64e-6 (4x the PSD, 2x the heights). Scenarios, not a measured site.')
        self._road.currentTextChanged.connect(self._store)
        self._v_min = self._spin(3, 200, d['ride_speed_min_kph'], 0, 2, ' km/h', 'Slowest speed of the bracket.')
        self._v_max = self._spin(3, 200, d['ride_speed_max_kph'], 0, 2, ' km/h', 'Fastest speed of the bracket.')
        self._v_n = self._ispin(1, 8, d['ride_speed_count'], 'Speeds evenly spaced across the bracket.')
        self._seed0 = self._ispin(0, 10 ** 6, d['ride_seed_first'], 'First road realisation seed.')
        self._seed_n = self._ispin(1, 8, d['ride_seed_count'],
                                   'Number of independent road realisations. RMS numbers (DLC, '
                                   'acceleration RMS) do not depend on the seed for coherent tracks '
                                   '(fixed amplitude spectrum); peaks (travel, min load, damper '
                                   'peak) and independent-track cases do — use several seeds for those.')
        self._coh = QComboBox(); self._coh.addItems(['both', 'coherent', 'independent'])
        self._coh.setToolTip('Left/right track bracket: coherent = identical tracks (pure heave/pitch), '
                             'independent = unrelated tracks (roll excited). Real roads sit between.')
        self._coh.currentTextChanged.connect(self._store)
        r = 0
        for text, w in (('Road class', self._road), ('Speed min', self._v_min), ('Speed max', self._v_max),
                        ('Number of speeds', self._v_n), ('First seed', self._seed0),
                        ('Number of seeds', self._seed_n), ('Left/right coherence', self._coh)):
            self._row(g, r, text, w); r += 1
        left.addWidget(box)

        # limits: travel margin + bump
        box = QGroupBox('LIMITS: BUMP STOP + OCCASIONAL BUMP'); g = QGridLayout(box); g.setSpacing(4)
        self._margin = self._spin(1, 200, 25., 1, 1, ' mm',
                                  'Wheel travel from static to the nearer bump/droop stop. '
                                  'Peak wheel travel / this = travel usage; > 1 = hits the stop.')
        self._margin_btn = QPushButton('Use model travel')
        self._margin_btn.setToolTip('Fill from the model: min over axles of (stroke - sag)/MR and sag/MR '
                                    'at the current preload and stroke (Motion panel).')
        self._margin_btn.clicked.connect(self._use_model_travel)
        self._bump_h = self._spin(1, 200, d['ride_bump_height_mm'], 1, 1, ' mm', 'Occasional single bump height (half-sine).')
        self._bump_l = self._spin(0.05, 5, d['ride_bump_length_m'], 2, 0.05, ' m', 'Occasional single bump length along the road.')
        self._flat = self._spin(0.1, 10, d['ride_flat_ride_max_ratio'], 2, 0.05, '',
                                'Flat-ride limit: after the rear wheels leave the bump, RMS pitch motion at the '
                                'axles / RMS bounce motion, averaged over the speed bracket. 1.0 = pitch no larger '
                                'than bounce (RCVD p.410 "lands flat").')
        r = 0
        self._row(g, r, 'Travel to the stop', self._margin); r += 1
        g.addWidget(self._margin_btn, r, 0, 1, 2); r += 1
        self._travel_lbl = QLabel(''); self._travel_lbl.setWordWrap(True)
        self._travel_lbl.setStyleSheet(f'color:{_MUTED};font-size:11px')
        g.addWidget(self._travel_lbl, r, 0, 1, 2); r += 1
        self._row(g, r, 'Bump height', self._bump_h); r += 1
        self._row(g, r, 'Bump length', self._bump_l); r += 1
        self._row(g, r, 'Max pitch/bounce after bump', self._flat); r += 1
        left.addWidget(box)

        # sweep range
        box = QGroupBox('RIDE-RATE SWEEP (AS RIDE FREQUENCY)'); g = QGridLayout(box); g.setSpacing(4)
        self._ff_min = self._spin(0.5, 10, d['ride_freq_front_min_Hz'], 2, 0.25, ' Hz', 'Front ride frequency, low end.')
        self._ff_max = self._spin(0.5, 10, d['ride_freq_front_max_Hz'], 2, 0.25, ' Hz', 'Front ride frequency, high end.')
        self._fr_min = self._spin(0.5, 10, d['ride_freq_rear_min_Hz'], 2, 0.25, ' Hz', 'Rear ride frequency, low end.')
        self._fr_max = self._spin(0.5, 10, d['ride_freq_rear_max_Hz'], 2, 0.25, ' Hz', 'Rear ride frequency, high end.')
        self._npts = self._ispin(2, 12, d['ride_sweep_points'], 'Grid points per axis (front x rear pairs).')
        r = 0
        for text, w in (('Front min', self._ff_min), ('Front max', self._ff_max),
                        ('Rear min', self._fr_min), ('Rear max', self._fr_max), ('Points per axis', self._npts)):
            self._row(g, r, text, w); r += 1
        left.addWidget(box)

        # actions
        self._analyse_btn = QPushButton('Analyse current car')
        self._analyse_btn.setToolTip('Road metrics, transfer functions, PSDs and the bump check for the '
                                     'springs currently in the model.')
        self._analyse_btn.clicked.connect(self.analyse_current)
        self._solve_btn = QPushButton('Solve ride rates')
        self._solve_btn.setStyleSheet(f'background:{_ACCENT};color:#fff;font-weight:600;')
        self._solve_btn.setToolTip('Sweep front x rear ride rate over the range at every road case and '
                                   'speed; pick the pair with the lowest worst-corner DLC that fits the '
                                   'travel margin and the flat-ride limit; convert to spring rates.')
        self._solve_btn.clicked.connect(lambda: self.run_solve(blocking=False))
        self._cancel_btn = QPushButton('Cancel'); self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel)
        left.addWidget(self._analyse_btn); left.addWidget(self._solve_btn); left.addWidget(self._cancel_btn)
        self._progress = QProgressBar(); self._progress.setRange(0, 1); self._progress.setValue(0)
        self._progress.setTextVisible(True)
        left.addWidget(self._progress)
        left.addStretch(1)

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
        for cb in (self._road, self._coh, self._confirm):
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
            self._confirm.setChecked(bool(car.get('ride_inputs_confirmed', False)))
            margin = car.get('ride_stop_margin_mm')
            if margin is None:
                m = self._model_travel_mm()
                margin = m['margin_mm'] if m else 25.
            self._margin.setValue(float(margin))
        finally:
            for w in list(widgets.values()) + list(self._damp.values()) + [self._margin]:
                w.blockSignals(False)
            for cb in (self._road, self._coh, self._confirm):
                cb.blockSignals(False)
        self._store()

    def _store(self, *_):
        car = self._car()
        car['ride_pitch_inertia_kgm2'] = float(self._pitch_I.value())
        car['ride_roll_inertia_kgm2'] = float(self._roll_I.value())
        car['ride_damping_wheel_Nspm'] = [float(self._damp[c].value()) for c in CORNERS]
        car['ride_inputs_confirmed'] = bool(self._confirm.isChecked())
        car['ride_road_class'] = self._road.currentText()
        car['ride_speed_min_kph'] = float(self._v_min.value())
        car['ride_speed_max_kph'] = float(self._v_max.value())
        car['ride_speed_count'] = int(self._v_n.value())
        car['ride_seed_first'] = int(self._seed0.value())
        car['ride_seed_count'] = int(self._seed_n.value())
        car['ride_coherence'] = self._coh.currentText()
        car['ride_stop_margin_mm'] = float(self._margin.value())
        car['ride_bump_height_mm'] = float(self._bump_h.value())
        car['ride_bump_length_m'] = float(self._bump_l.value())
        car['ride_flat_ride_max_ratio'] = float(self._flat.value())
        car['ride_freq_front_min_Hz'] = float(self._ff_min.value())
        car['ride_freq_front_max_Hz'] = float(self._ff_max.value())
        car['ride_freq_rear_min_Hz'] = float(self._fr_min.value())
        car['ride_freq_rear_max_Hz'] = float(self._fr_max.value())
        car['ride_sweep_points'] = int(self._npts.value())
        self._refresh_banner()

    def _refresh_banner(self, *_):
        confirmed = self._confirm.isChecked()
        for lab, text in self._assumed_labels:
            lab.setText(text.replace(' (ASSUMED)', '') if confirmed else text)
            lab.setStyleSheet(f'color:{_INK}' if confirmed else f'color:{_ACCENT}')
        if confirmed:
            self._banner.setText('Inertias and damping: supplied by you for this car (confirmed). '
                                 'Roads are ISO 8608 class-centre scenarios, not a measured course.')
            self._banner.setStyleSheet(f'color:{_MUTED};font-size:12px;padding:4px;')
        else:
            self._banner.setText(
                'ASSUMED INPUTS — pitch inertia, roll inertia and the four damping coefficients are '
                'round placeholders, not this car\'s values. Every DLC, travel, acceleration and '
                'settle number on this page depends on them. Set them and tick "these are this '
                'car\'s values". Roads are ISO 8608 class-centre scenarios, not a measured course.')
            self._banner.setStyleSheet(f'color:{_ACCENT};font-size:12px;font-weight:600;padding:4px;'
                                       f'border:1px solid {_ACCENT};border-radius:4px;')

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

    def _use_model_travel(self):
        m = self._model_travel_mm()
        if m:
            self._margin.setValue(m['margin_mm'])

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
        m = self._model_travel_mm(veh)
        if m:
            self._travel_lbl.setText(
                f"Model travel at the wheel: front bump {m['front_bump_mm']:.1f} / droop {m['front_droop_mm']:.1f} mm, "
                f"rear bump {m['rear_bump_mm']:.1f} / droop {m['rear_droop_mm']:.1f} mm "
                f"(stroke and preload from the Motion panel). Smallest = {m['margin_mm']:.1f} mm.")

    # ── inputs -> RideInputs ──────────────────────────────────────────────
    def inputs(self):
        n = int(self._v_n.value())
        v0, v1 = float(self._v_min.value()) / 3.6, float(self._v_max.value()) / 3.6
        if v1 < v0:
            raise ValueError('speed max must be at least speed min')
        speeds = tuple(float(v) for v in (np.linspace(v0, v1, n) if n > 1 else [v0]))
        seeds = tuple(range(int(self._seed0.value()), int(self._seed0.value()) + int(self._seed_n.value())))
        return RS.RideInputs(
            pitch_inertia_kgm2=float(self._pitch_I.value()),
            roll_inertia_kgm2=float(self._roll_I.value()),
            corner_damping_Nspm=tuple(float(self._damp[c].value()) for c in CORNERS),
            road_class=self._road.currentText(), speeds_mps=speeds, seeds=seeds,
            coherence=self._coh.currentText(),
            stop_margin_m=float(self._margin.value()) / 1000.,
            bump_height_m=float(self._bump_h.value()) / 1000.,
            bump_length_m=float(self._bump_l.value()),
            flat_ride_max_ratio=float(self._flat.value()))

    def sweep_grids(self, veh):
        n = int(self._npts.value())
        fr, ff = RS.ride_rate_grid_Npm(veh, 'front', self._ff_min.value(), self._ff_max.value(), n)
        rr, fR = RS.ride_rate_grid_Npm(veh, 'rear', self._fr_min.value(), self._fr_max.value(), n)
        return fr, rr, ff, fR

    # ── actions ───────────────────────────────────────────────────────────
    def analyse_current(self):
        """Road metrics / transfer / PSD / bump for the springs in the model."""
        try:
            veh = self.vehicle(); inputs = self.inputs()
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                study = RS.current_car_study(veh, inputs)
            finally:
                QApplication.restoreOverrideCursor()
        except Exception as e:
            self._status.setText(f'Analysis failed: {type(e).__name__}: {e}')
            return None
        self.last_study = study
        self._plot_metrics(study, inputs)
        self._plot_transfer(study, veh)
        self._plot_psd(study, inputs)
        self._plot_bump(study, inputs)
        for extra in (self._plot_contact, self._plot_launch):
            try:
                extra(veh, inputs)
            except Exception as e:
                self._status.setText(f'{extra.__name__} failed: {type(e).__name__}: {e}')
        worst = max(m.worst_dlc for m in study['metrics'])
        self._status.setText(
            f'Current springs on class {inputs.road_class}: worst-corner DLC {worst:.3f} over '
            f'{len(study["cases"])} road cases; ride frequency front {study["front_ride_frequency_Hz"]:.2f} / '
            f'rear {study["rear_ride_frequency_Hz"]:.2f} Hz (rear/front {study["flat_ride_ratio"]:.3f}); '
            f'flat-ride pitch/bounce per speed: '
            + ', '.join(f'{v*3.6:.0f} km/h {b["pitch_to_bounce_ratio"]:.2f}' for v, b in study['bumps'].items()))
        for k in ('metrics',):
            self._tabs.setCurrentWidget(self._canvases[k])
        return study

    def run_solve(self, blocking=False):
        """The ride-rate solve. blocking=True runs inline (tests/headless) and
        returns the selection dict; otherwise a worker thread with progress."""
        try:
            veh = self.vehicle(); inputs = self.inputs()
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

    def _plot_metrics(self, study, inputs):
        fig = self._figs['metrics']; fig.clear()
        metrics, cases = study['metrics'], study['cases']
        n = len(cases); x = np.arange(n); wbar = 0.2
        axes = fig.subplots(2, 3)
        labels = [f'{c.speed_mps*3.6:.0f} km/h\n{"coh" if c.coherence == "coherent" else "indep"} s{c.seed}'
                  for c in cases]
        panels = (
            (axes[0, 0], 'dlc', 'Dynamic load coefficient per corner', 'RMS(Fz - mean) / mean', None),
            (axes[0, 1], 'min_load_N', 'Minimum tire load (linear model)', 'N', 0.),
            (axes[0, 2], 'contact_loss_fraction_linear', 'Contact-loss indicator (samples at Fz <= 0)', 'fraction', None),
            (axes[1, 0], 'travel_usage', 'Wheel-travel usage (peak / travel to the stop)', 'fraction', 1.),
            (axes[1, 1], 'damper_velocity_rms_mps', 'Damper velocity RMS (bars), peak (marks)', 'm/s at the damper', None),
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
                    ax.text(0.5, 0.5, 'no contact loss predicted\n(linear load stays > 0 in every case)',
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
        _style_axes(ax, 'Body acceleration RMS (heave, pitch, roll)', '', 'm/s² RMS'); _legend(ax)
        fig.suptitle(f'Current springs — ISO 8608 class {inputs.road_class}, '
                     f'{inputs.record_length_m:.0f} m records, travel margin {inputs.stop_margin_m*1000:.0f} mm',
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
            _style_axes(ax, title, 'frequency (Hz)', ylab)
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
                'Damping and inertias: user inputs (left column).',
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
            _style_axes(ax, f'{axle.upper()}: contact-patch load change per mm of road', 'road input frequency at the wheel (Hz)', 'N per mm')
            _legend(ax)
            ax = axes[r, 1]
            ax.plot(freqs, d['rms_pct_of_static'], color='#FFFFFF', lw=1.8, marker='o', ms=3, label='damper as entered')
            ax.plot(freqs, ratio[axle]['rms_pct_of_static'], color='#FFD600', lw=1.6, ls='--', marker='o', ms=3,
                    label='same damping RATIO as the model')
            ax.axvline(d['current_Hz'], color='#E53935', lw=1.1, ls=':', label=f'in the model {d["current_Hz"]:.2f} Hz')
            _style_axes(ax, f'{axle.upper()}: RMS contact-patch load deviation, class {inputs.road_class} '
                            f'(static {d["static_load_N"]:.0f} N)', f'{axle} ride frequency (Hz)', '% of static load')
            _legend(ax)
        fig.suptitle('Contact-patch load deviation vs ride frequency - synthetic ISO 8608 road, damping is a user input',
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
        _style_axes(axes[0, 0], f'Road input PSD at each speed (class {inputs.road_class})', 'frequency (Hz)', 'm² per Hz')
        _style_axes(axes[0, 1], 'Body heave acceleration PSD', 'frequency (Hz)', '(m/s²)² per Hz')
        _style_axes(axes[1, 0], 'FL wheel-travel PSD', 'frequency (Hz)', 'm² per Hz')
        _style_axes(axes[1, 1], 'FL dynamic tire-load PSD', 'frequency (Hz)', 'N² per Hz')
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
            _style_axes(ax, f'{v*3.6:.0f} km/h: pitch/bounce after bump {b["pitch_to_bounce_ratio"]:.2f} '
                            f'(limit {inputs.flat_ride_max_ratio:.2f}, {"ok" if b["flat_ride_ok"] else "NOT MET"}); '
                            f'front settle {b["front_settle_s"]:.2f} s, rear {b["rear_settle_s"]:.2f} s from own entry',
                        'time from front wheels entering the bump (s)', 'mm')
            _legend(ax, ncol=5)
        fig.suptitle(f'Occasional bump {inputs.bump_height_m*1000:.0f} mm x {inputs.bump_length_m:.2f} m, '
                     'current springs, both wheels of an axle, rear delayed by wheelbase/speed', color=_INK, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        self._canvases['bump'].draw_idle()
