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
from gui.plot_dialog import ReadableCanvas
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QDoubleSpinBox, QSpinBox, QCheckBox, QScrollArea, QFrame, QComboBox,
    QLineEdit, QDialog,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

# Selectable tracks: (label, json filename).  The page lists every file that
# exists; missing ones are skipped.
_TRACKS = [
    ('2024', 'autocross26.json'),
    ('2025', 'autocross_2025.json'),
    ('MI 2018 Endurance', 'mi2018_endurance.json'),
]

# The tire-dataset dropdown is built DYNAMICALLY from whatever .mat files the
# user has placed in tire_data/ (which is gitignored — TTC data is never in
# the repo).  Each entry's label is read from the file's own tire_id AT
# RUNTIME, so no TTC tire names / run numbers are hard-coded in this (public)
# source file.  See LaptimePage._discover_tires().

from vahan.laptime import Track, LapSimulator, assumed_torque_curve
# the ONE primary ratio (vahan/acceleration.py, from the powertrain sheet:
# primary 2.111) - this page defaulted to 1.0, which halved the wheel torque
# and held the lap's forward g near 0.5-0.7 g (user 2026-11-14 "long g is
# capped at 0.5g again": every restart reset a hand-typed fix to 1.0)
from vahan.acceleration import DEFAULT_PRIMARY

# What the torque-curve box says when the user has no dyno sheet.  The word
# ASSUMED is the switch: leave it and the generic restrictor-limited 600 SHAPE
# (scaled to this car's own peak power) is used and LABELLED as assumed.
_ASSUMED_CURVE_HINT = 'ASSUMED  (generic restricted-600 shape)'

# colour-blind-safe corner colours (matches the suspension page)
_CC = {'FL': '#FFD600', 'FR': '#E53935', 'RL': '#FFFFFF', 'RR': '#42A5F5'}
_LS = {'FL': '-', 'FR': '--', 'RL': '-.', 'RR': ':'}
_CORNER_WORDS = {'FL': 'front left', 'FR': 'front right',
                 'RL': 'rear left', 'RR': 'rear right'}

# plot theme (dark, matches the app).  Colour-blind user: yellow / red /
# white / blue only — never red-vs-green or purple-vs-blue.
_BG_FIG, _BG_AX = '#000000', '#0a0a0c'
_GRID, _SPINE = '#26262a', '#3a3a40'
_TXT, _TXT_DIM = '#e0e0e4', '#a0a0a8'
_XLABEL = 'distance along the lap (m)'
# speed map: slow = blue -> white -> yellow -> fast = red
_SPEED_CMAP = LinearSegmentedColormap.from_list(
    'lap_speed', ['#1E6FD9', '#FFFFFF', '#FFD600', '#E53935'])

# user-supplied 2026 gearbox (tooth counts verified: 33/12, 32/16, 30/18,
# 26/18, 30/23, 29/24)
_DEFAULT_GEARS = [2.750, 2.000, 1.667, 1.444, 1.304, 1.208]

# Detail pass: stop adding full-solve points once straight lines between them
# reproduce the solve inputs (sideways / forward g) to within this.
_DETAIL_TOL_G = 0.03


class _SimWorker(QThread):
    progress = pyqtSignal(str, int)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, sim: LapSimulator, track: Track, n_detail: int,
                 method: str = 'ggv'):
        super().__init__()
        self._sim, self._track, self._nd = sim, track, n_detail
        self._method = method

    def run(self):
        try:
            # detail solves placed where the inputs change (smooth, faithful
            # detail graphs) — see LapSimulator._detail_stations
            res = self._sim.simulate(
                self._track, n_detail=self._nd,
                progress_cb=lambda m, p: self.progress.emit(m, p),
                detail_tol_g=_DETAIL_TOL_G, method=self._method)
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
        ('diff',   'Diff yaw moment',  False),
        ('rideh',  'Ride height',      False),
        ('segs',   'Straights vs corners', False),
        ('ttemp',  'Tyre temperature', False),
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
            # rotating inertia + shift model live on the CAR (VehicleParams);
            # these boxes are sim-local overrides seeded from it.
            self._I_wf.setValue(float(veh.wheel_inertia_front_kgm2))
            self._I_wr.setValue(float(veh.wheel_inertia_rear_kgm2))
            self._I_eng.setValue(float(veh.engine_inertia_kgm2))
            self._shift_time.setValue(float(veh.shift_time_s))
            self._shift_int.setValue(float(veh.min_shift_interval_s))
            self._shift_hyst.setValue(float(veh.shift_hysteresis_frac) * 100.0)
        except Exception:
            pass
        self._update_inertia_note()
        self._update_tq_note()
        self._refresh_tire_active()
        self._on_diff_changed()

    # ── UI ───────────────────────────────────────────────────────────────
    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        # ── left sidebar (scrollable — the GRAPHS never scroll, this may) ─
        side = QVBoxLayout()
        side.setSpacing(6)

        title = QLabel('LAP TIME SIM')
        title.setStyleSheet('color:#E23B48; font-size:15px; font-weight:bold;')
        side.addWidget(title)

        self._track_lbl = QLabel('— no track —')
        self._track_lbl.setStyleSheet('color:#aaa; font-size:11px;')
        self._track_lbl.setWordWrap(True)
        side.addWidget(self._track_lbl)

        def hdr(text):
            l = QLabel(text)
            l.setStyleSheet('color:#E23B48; font-size:11px; font-weight:bold;'
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

        # ── TIRE selector — separate FRONT and REAR axle dropdowns, both
        #    populated from local tire_data/ files.  Pick a different tire for
        #    each axle to run a SPLIT compound setup (e.g. one in front, one in
        #    back).  Labels are read from each file's tire id at runtime, so no
        #    tire names are hard-coded here (TTC-compliant).
        self._tire_rounds = self._discover_tires()
        if self._tire_rounds:
            hdr('TIRE  (local dataset)')
            g = grid()
            g.addWidget(QLabel('Front:'), 0, 0)
            self._tire_combo_f = QComboBox()
            self._tire_combo_f.setStyleSheet('QComboBox { font-size:11px; }')
            self._tire_combo_f.addItem('(loaded tire — pick to swap)')
            for lbl, _fn in self._tire_rounds:
                self._tire_combo_f.addItem(lbl)
            self._tire_combo_f.setToolTip(
                'Front-axle tire.  Built from your local tire_data/ files; '
                'labels are each file’s own tire id.  Re-Sim to compare.')
            self._tire_combo_f.activated.connect(self._on_tire_front)
            g.addWidget(self._tire_combo_f, 0, 1)

            g.addWidget(QLabel('Rear:'), 1, 0)
            self._tire_combo_r = QComboBox()
            self._tire_combo_r.setStyleSheet('QComboBox { font-size:11px; }')
            self._tire_combo_r.addItem('(same as front)')
            for lbl, _fn in self._tire_rounds:
                self._tire_combo_r.addItem(lbl)
            self._tire_combo_r.setToolTip(
                'Rear-axle tire.  Leave on "(same as front)" for one compound '
                'all round, or pick a different tire to run a SPLIT setup '
                '(e.g. a grippier compound on the rear).')
            self._tire_combo_r.activated.connect(self._on_tire_rear)
            g.addWidget(self._tire_combo_r, 1, 1)

            self._tire_active = QLabel('')
            self._tire_active.setStyleSheet('color:#8a8a92; font-size:10px;')
            self._tire_active.setWordWrap(True)
            side.addWidget(self._tire_active)

        hdr('AERO + GRIP')
        # WHICH aero this lap uses (2026-10-21): the car's own package from
        # the Dynamics panel (default — the same downforce every other page
        # uses), or numbers typed into the boxes below.
        _arow = QHBoxLayout()
        _arow.addWidget(QLabel('Aero from:'))
        self._aero_src = QComboBox()
        self._aero_src.addItem("the car (Dynamics panel)", 'car')
        self._aero_src.addItem('the boxes below', 'page')
        self._aero_src.setToolTip(
            "the car: Cl·A and rear share come from the car's aero package "
            '(Dynamics panel reference downforce / speed / rear share, Apply '
            'Aero).  Apply Aero off = no downforce.  The boxes then only SHOW '
            'it.\nthe boxes below: type your own Cl·A / rear share for a '
            'what-if lap.')
        _arow.addWidget(self._aero_src, 1)
        side.addLayout(_arow)
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
        self._rh_f = spin(g, 5, 0, 'Static RH F:', 0.0, 300.0, 50.0, 0, 2.0,
                          ' mm',
                          'Front ride height at STATIC SAG (car on its springs '
                          'with driver).  Aero downforce drops it from here as '
                          'speed builds — see the Ride height graph.')
        self._rh_r = spin(g, 6, 0, 'Static RH R:', 0.0, 300.0, 50.0, 0, 2.0,
                          ' mm', 'Rear ride height at static sag.')
        # Ride-height CAP: hold ride height >= cap at a reference speed.
        self._rh_refspeed = spin(g, 7, 0, 'Cap @ speed:', 5.0, 300.0, 100.0, 0,
                                 5.0, ' kph',
                                 'Reference speed to size/check the ride-height '
                                 'cap at (typ. your fastest sustained point).')
        self._rh_cap_f = spin(g, 8, 0, 'RH cap F:', 0.0, 300.0, 25.0, 0, 1.0,
                              ' mm', 'Minimum front ride height you will allow '
                              '(splitter/floor clearance, bump-stop gap).')
        self._rh_cap_r = spin(g, 9, 0, 'RH cap R:', 0.0, 300.0, 25.0, 0, 1.0,
                              ' mm', 'Minimum rear ride height you will allow.')
        # Live reference-speed readout: ties the Cl·A input to the number
        # aero people actually quote ("N at 60 mph"), so a wrong entry is
        # obvious immediately.
        self._aero_ref = QLabel('')
        self._aero_ref.setStyleSheet('color:#8a8a92; font-size:10px;')
        side.addWidget(self._aero_ref)
        self._rh_cap_readout = QLabel('')
        self._rh_cap_readout.setStyleSheet('font-size:10px;')
        self._rh_cap_readout.setWordWrap(True)
        self._rh_cap_readout.setToolTip(
            'First-order sizing at the operating-point rate.  The Ride height '
            'GRAPH is the accurate result — it integrates the real (possibly '
            'nonlinear) MR(travel) curve, so a progressive MR shows less heave '
            'there.')
        side.addWidget(self._rh_cap_readout)

        # The project's own aero package (Dynamics panel, Apply Aero) is NOT
        # copied into these boxes automatically — this page's boxes are
        # sim-local overrides.  Say so when they differ, and offer one click.
        self._aero_car_btn = QPushButton("Use the car's aero package")
        self._aero_car_btn.setToolTip(
            'Copy Cl·A and the rear downforce share from the Dynamics panel '
            '(downforce at reference speed, Apply Aero).')
        self._aero_car_btn.clicked.connect(self._use_car_aero)
        side.addWidget(self._aero_car_btn)

        def _upd_ref(*_):
            v = 60 / 2.23694                      # 60 mph in m/s
            q = 0.5 * float(self._rho.value()) * v * v
            F = float(self._cla.value()) * q
            txt = (f'≈ {F:+.0f} N ({F/9.81:+.0f} kgf) downforce @ 60 mph · '
                   f'drag {float(self._cda.value())*q:.0f} N @ 60 mph')
            car = self._project_aero()
            if self.aero_source() == 'car':
                txt = ('AERO IN USE: the car — ' + (
                    f'Cl·A {car[0]:.2f} m², {car[1]:.0f} % rear' if car is not None
                    else 'Apply Aero is OFF on the Dynamics panel, so NO downforce')
                    + '\n' + txt)
            else:
                txt = 'AERO IN USE: the boxes above (typed)\n' + txt
                if car is not None and (
                        abs(car[0] - float(self._cla.value())) > 0.02
                        or abs(car[1] - float(self._cop.value())) > 0.5):
                    txt += (f'\n⚠ the car has aero ON: Cl·A {car[0]:.2f} m², '
                            f'{car[1]:.0f} % rear — this run uses the boxes')
            self._aero_ref.setText(txt)
            self._aero_ref.setWordWrap(True)
            self._update_rh_cap()
        for sb in (self._cla, self._cda, self._rho, self._cop, self._rh_f,
                   self._rh_r, self._rh_refspeed, self._rh_cap_f, self._rh_cap_r):
            sb.valueChanged.connect(_upd_ref)
        self._upd_aero_ref = _upd_ref
        self._aero_src.currentIndexChanged.connect(
            lambda *_: (self._apply_aero_source(), _upd_ref()))
        self._apply_aero_source()
        _upd_ref()

        hdr('GEARBOX  (primary × gear × final)')
        g = grid()
        self._primary = spin(g, 0, 0, 'Primary:', 0.5, 10.0, DEFAULT_PRIMARY, 3, 0.01, '',
                             'Crank-to-gearbox reduction (powertrain sheet: '
                             '2.111).  Total = primary × gear × chain.  1.0 '
                             'halves the wheel torque.')
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

        # ── ROTATING INERTIA ────────────────────────────────────────────
        # The single biggest known bias in this lap number, and it always
        # flattered us: MEASURED +2.00 s on a 41.90 s lap of autocross26.
        hdr('ROTATING INERTIA  (ASSUMED — measure it)')
        g = grid()
        self._I_wf = spin(g, 0, 0, 'Wheel F (each):', 0.0, 2.0, 0.19, 3, 0.01,
                          ' kg·m²',
                          'ONE front wheel: tyre + rim + hub + rotor, about '
                          'the axle.  ASSUMED 0.19 for a 10-inch FSAE wheel — '
                          'measure it with a bifilar (two-string) swing.')
        self._I_wr = spin(g, 1, 0, 'Wheel R (each):', 0.0, 2.0, 0.19, 3, 0.01,
                          ' kg·m²', 'ONE rear wheel, same basis as the front.')
        self._I_eng = spin(g, 2, 0, 'Engine+clutch:', 0.0, 1.0, 0.05, 3, 0.005,
                           ' kg·m²',
                           'Crank + primary gear + clutch basket.  This one '
                           'DOMINATES: it is referred through the total gear '
                           'ratio SQUARED (2.111 × 2.75 × 3.545 = 20.6, '
                           '20.6² = 424 in 1st).  ASSUMED 0.05 '
                           'for a 600 cc four — get it from the engine '
                           'manufacturer or a run-down test.')
        self._inertia_readout = QLabel('')
        self._inertia_readout.setStyleSheet('color:#8a8a92; font-size:10px;')
        self._inertia_readout.setWordWrap(True)
        side.addWidget(self._inertia_readout)

        # ── SHIFT MODEL ─────────────────────────────────────────────────
        hdr('GEARSHIFT  (driver, ASSUMED)')
        g = grid()
        self._shift_time = spin(g, 0, 0, 'Dead time:', 0.0, 0.60, 0.10, 3,
                                0.01, ' s',
                                'Torque is CUT for this long, end to end: drop '
                                'drive, select, pick drive back up.  0 = the '
                                'old free instantaneous shift.')
        self._shift_int = spin(g, 1, 0, 'Min interval:', 0.0, 3.0, 0.60, 2,
                               0.05, ' s',
                               'Shortest allowed gap between two shifts.  This '
                               'is what bounds CHATTER: without it the picker '
                               'used 2nd gear for 0.08 s (one station) at the '
                               'top of a straight, five times a lap.')
        self._shift_hyst = spin(g, 2, 0, 'Hysteresis:', 0.0, 50.0, 4.0, 0, 1.0,
                                ' %',
                                'How much more wheel force the next gear must '
                                'make before the driver takes it.  On this '
                                'gearbox the 1-2 force step is about 37%, so '
                                'any sane hysteresis is inactive — the minimum '
                                'interval is the knob that does the work.')

        # ── ENGINE TORQUE MODEL ─────────────────────────────────────────
        hdr('ENGINE TORQUE')
        self._tq_use = QCheckBox('Use a torque curve instead of the plateau')
        # ON BY DEFAULT: the engine is a Honda CBR600RR behind the 20 mm
        # restrictor (user, 2026-07-30), and the plateau is measured 48% high
        # at 2300 rpm.  The active curve is still the ASSUMED restricted-600
        # SHAPE scaled to this car's own peak power — the engine's IDENTITY is
        # known, its dyno sheet is not.  Two columns from a dyno replace it.
        self._tq_use.setChecked(True)
        self._tq_use.setToolTip(
            'OFF = the two-segment model: flat torque below peak-power rpm, '
            'constant power above.  That plateau reads about 48% high on wheel '
            'force at 2300 rpm — the whole 75 m acceleration event lives '
            'there.\nON = the table below, interpolated (held flat outside its '
            'range, never extrapolated).')
        self._tq_use.stateChanged.connect(lambda *_: self._update_tq_note())
        side.addWidget(self._tq_use)
        self._tq_text = QLineEdit(_ASSUMED_CURVE_HINT)
        self._tq_text.setToolTip(
            'CRANK torque against engine speed: "rpm:Nm, rpm:Nm, …".  Two '
            'columns off a dyno sheet.  Leave it on the word ASSUMED to use '
            'the generic restrictor-limited 600 SHAPE scaled to this car’s own '
            'peak power — a placeholder, not a measurement.')
        self._tq_text.editingFinished.connect(self._update_tq_note)
        side.addWidget(self._tq_text)
        self._tq_note = QLabel('')
        self._tq_note.setStyleSheet('color:#8a8a92; font-size:10px;')
        self._tq_note.setWordWrap(True)
        side.addWidget(self._tq_note)
        for sb in (self._I_wf, self._I_wr, self._I_eng):
            sb.valueChanged.connect(lambda *_: self._update_inertia_note())
        for sb in (self._primary, self._final, *self._gear_spins):
            sb.valueChanged.connect(lambda *_: self._update_inertia_note())

        # ── DIFFERENTIAL (corner entry/exit balance via locking %) ────────
        from vahan.differential import DREXLER_OPTIONS, Differential
        # header with an info (ⓘ) button — same style as the suspension panels
        drow = QHBoxLayout(); drow.setSpacing(6)
        dlbl = QLabel('DIFFERENTIAL  (Drexler LSD)')
        dlbl.setStyleSheet('color:#E23B48; font-size:11px; font-weight:bold;'
                           'padding-top:6px;')
        drow.addWidget(dlbl)
        dinfo = QPushButton('ⓘ')
        dinfo.setFixedSize(20, 20)
        dinfo.setCursor(Qt.CursorShape.PointingHandCursor)
        dinfo.setStyleSheet(
            'QPushButton { background:#111111; color:#E23B48; '
            'border:1px solid #2a2a2a; border-radius:10px; font-weight:bold; '
            'font-size:12px; }'
            'QPushButton:hover { background:#2a1013; border-color:#E23B48; }'
            'QPushButton:pressed { background:#171208; }')
        dinfo.setToolTip('What does the differential do? Click for detail.')
        dinfo.clicked.connect(self._show_diff_info)
        drow.addWidget(dinfo)
        drow.addStretch(1)
        side.addLayout(drow)
        g = grid()
        g.addWidget(QLabel('Type:'), 0, 0)
        self._diff_type = QComboBox()
        self._diff_type.addItems(['Salisbury LSD', 'Open', 'Spool (locked)'])
        self._diff_type.setToolTip(
            'Open = no locking (free diff).  Spool = fully locked.  '
            'Salisbury = Drexler ramp/clutch LSD (tune via ramp + preload).')
        self._diff_type.currentIndexChanged.connect(self._on_diff_changed)
        g.addWidget(self._diff_type, 0, 1)
        g.addWidget(QLabel('Ramp (pwr/coast):'), 1, 0)
        self._diff_ramp = QComboBox()
        self._diff_opts = list(DREXLER_OPTIONS.items())
        for opt, (p, c) in self._diff_opts:
            d = Differential('salisbury', p, c, 30)
            self._diff_ramp.addItem(
                f'{p:.0f}°/{c:.0f}°  ({d.power_locking_pct:.0f}/'
                f'{d.coast_locking_pct:.0f}%)')
        self._diff_ramp.setToolTip(
            'Drexler ramp options (power°/coast°).  Smaller angle = more '
            'locking.  More power-lock = more exit understeer + traction; '
            'more coast-lock = more corner-entry stability (less rotation).')
        self._diff_ramp.currentIndexChanged.connect(self._on_diff_changed)
        g.addWidget(self._diff_ramp, 1, 1)
        g.addWidget(QLabel('Preload:'), 2, 0)
        self._diff_preload = QDoubleSpinBox()
        self._diff_preload.setRange(0.0, 100.0); self._diff_preload.setDecimals(0)
        self._diff_preload.setSingleStep(5.0); self._diff_preload.setValue(30.0)
        self._diff_preload.setSuffix(' Nm')
        self._diff_preload.setToolTip('Clutch pre-clamp (Drexler: 25–35 Nm). '
                                      'Acts at all times — baseline entry '
                                      'stability even at neutral throttle.')
        self._diff_preload.valueChanged.connect(self._on_diff_changed)
        g.addWidget(self._diff_preload, 2, 1)
        g.addWidget(QLabel('Engine braking:'), 3, 0)
        self._diff_engbrake = QDoubleSpinBox()
        self._diff_engbrake.setRange(0.0, 60.0); self._diff_engbrake.setDecimals(0)
        self._diff_engbrake.setSingleStep(2.0); self._diff_engbrake.setValue(12.0)
        self._diff_engbrake.setSuffix(' Nm')
        self._diff_engbrake.setToolTip(
            'Closed-throttle engine-braking torque at the CRANK.  This is the '
            'overrun torque that flows through the diff and works the COAST '
            'ramp (set it from your dyno/feel; without it the coast ramp does '
            'nothing on corner entry).')
        self._diff_engbrake.valueChanged.connect(self._on_diff_changed)
        g.addWidget(self._diff_engbrake, 3, 1)
        self._diff_readout = QLabel('')
        self._diff_readout.setStyleSheet('color:#8a8a92; font-size:10px;')
        self._diff_readout.setWordWrap(True)
        side.addWidget(self._diff_readout)

        hdr('SIM')
        g = grid()
        g.addWidget(QLabel('Detail solves:'), 0, 0)
        self._ndetail = QSpinBox()
        self._ndetail.setRange(8, 1000); self._ndetail.setValue(240)
        self._ndetail.setToolTip(
            'Most points where the FULL suspension solve runs (tyre grip '
            'used, roll, LLTD, travel graphs).  They are placed where the '
            'car changes what it is doing (turn-in, apex, braking point), '
            'not evenly, and it stops early once the graphs are faithful.  '
            'More = smoother detail graphs, ~0.05 s each.  The lap time '
            'does not depend on this.')
        g.addWidget(self._ndetail, 0, 1)
        g.addWidget(QLabel('Method:'), 1, 0)
        self._method = QComboBox()
        self._method.addItem("the car's G-G limits (full solve)", 'ggv')
        self._method.addItem('old quick method', 'pointmass')
        self._method.setToolTip(
            "the car's G-G limits: before the lap, the full suspension solve "
            'finds how much the car can corner, brake and accelerate (and '
            'every mix of them) at several speeds, with the downforce at each '
            'speed, brake balance and the driven rear axle — the same limit as '
            'the G-G diagram on the Dynamics page.  The lap then never asks for '
            'more.  First run with downforce ~1 min (then remembered until the '
            'car changes).  Standing start with the clutch slipping.\n'
            'old quick method: corner speed from the most-loaded single tyre, '
            'braking and accelerating on a grip circle (it braked harder and '
            'cornered slower than the car can).')
        g.addWidget(self._method, 1, 1)
        g.addWidget(QLabel('Tyre use cap:'), 2, 0)
        self._util_cap = QDoubleSpinBox()
        self._util_cap.setRange(0.30, 1.00); self._util_cap.setSingleStep(0.05)
        self._util_cap.setDecimals(2); self._util_cap.setValue(1.00)
        self._util_cap.setToolTip(
            'Most of its grip any tyre is allowed to use (1.00 = off, the '
            'car runs at the limit).  0.80 = no tyre ever uses more than 80 % '
            'of the grip it has: corner, brake and drive limits all shrink '
            'with it.  Separate from the grip scale: the grip scale is how '
            'much grip the road gives, this is how much of it the driver '
            'uses.  The tyre-use graph is still measured against the full '
            'grip, so it tops out near this cap.')
        g.addWidget(self._util_cap, 2, 1)
        # EVENT (user 2026-11-15): an event is the same lap driven several
        # times (endurance = 22 km of laps).  Total time = laps x lap time.
        g.addWidget(QLabel('Event laps:'), 3, 0)
        self._event_laps = QSpinBox()
        self._event_laps.setRange(1, 500); self._event_laps.setValue(1)
        self._event_laps.setToolTip(
            'How many times the lap is driven in the event.  Used when the '
            'event distance is 0.  Total event time = laps x lap time.')
        g.addWidget(self._event_laps, 3, 1)
        g.addWidget(QLabel('Event distance:'), 4, 0)
        self._event_km = QDoubleSpinBox()
        self._event_km.setRange(0.0, 100.0); self._event_km.setDecimals(2)
        self._event_km.setSingleStep(0.5); self._event_km.setValue(0.0)
        self._event_km.setSuffix(' km')
        self._event_km.setToolTip(
            'Total event length (FSAE endurance = 22 km).  0 = off, use the '
            'lap count instead.  When set, laps = distance / this track\'s lap '
            'length, rounded UP to whole laps (you finish the lap you are on).')
        g.addWidget(self._event_km, 4, 1)
        self._event_laps.valueChanged.connect(lambda *_: self._update_event())
        self._event_km.valueChanged.connect(lambda *_: self._update_event())

        self._build_tyre_temp(side, hdr, grid, spin)

        self._btn = QPushButton('►  Sim Current Car')
        self._btn.setStyleSheet(
            'QPushButton { background:#E23B48; color:#ffffff; padding:10px;'
            ' font-weight:bold; font-size:13px; border-radius:4px; }'
            'QPushButton:hover { background:#ef5867; }'
            'QPushButton:disabled { background:#3a3a3a; color:#777; }')
        self._btn.clicked.connect(self._on_sim)
        side.addWidget(self._btn)

        self._status = QLabel('')
        self._status.setStyleSheet('color:#8a8a92; font-size:11px;')
        self._status.setWordWrap(True)
        side.addWidget(self._status)
        # loads around the lap (gui/lap_loads_page.py, the Loads-page solve per station)
        self._loads_btn = QPushButton('Loads around this lap…')
        self._loads_btn.clicked.connect(lambda: __import__('gui.lap_loads_page', fromlist=['open_window']).open_window(self._main, self))
        side.addWidget(self._loads_btn)
        self._ggmap_btn = QPushButton("This lap on the car's G-G limits…")
        self._ggmap_btn.setToolTip(
            'Every point of the lap (sideways g, tyre forward/braking g) over '
            "the car's limit at a few speeds.  Needs a lap run with the G-G "
            'method.')
        self._ggmap_btn.clicked.connect(self.open_ggmap)
        side.addWidget(self._ggmap_btn)
        # MMD at a corner's speed, with this lap's grip + aero (gui/lap_mmd.py)
        self._mmd_btn = QPushButton('MMD from this lap…')
        self._mmd_btn.setToolTip(
            "The Milliken Moment Diagram at a corner's speed from the lap you "
            "just ran (the lap's grip scale and aero), with the lap's points "
            'in that corner marked on it.')
        self._mmd_btn.clicked.connect(lambda: __import__('gui.lap_mmd', fromlist=['open_window']).open_window(self._main, self))
        side.addWidget(self._mmd_btn)
        # engine torque + power the lap runs on, and drive force per gear (gui/lap_engine.py)
        self._eng_btn = QPushButton('Engine torque && power curves…')
        self._eng_btn.setToolTip(
            'Crank torque and power vs rpm exactly as the lap sim uses them, '
            'wheel power after drivetrain losses, the redline, the drive '
            'force in each gear vs speed, and (after a run) where the lap '
            'spent its rpm.')
        self._eng_btn.clicked.connect(lambda: __import__('gui.lap_engine', fromlist=['open_window']).open_window(self._main, self))
        side.addWidget(self._eng_btn)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet('color:#2a2a2a;')
        side.addWidget(sep)

        self._readout = QLabel('—')
        self._readout.setStyleSheet(
            "color:#e8e8ec; font-family:'Consolas','SF Mono',monospace;"
            'font-size:12px; background:#0e0e10; padding:8px;'
            'border:1px solid #2a2a2a; border-radius:4px;')
        self._readout.setWordWrap(True)
        # EVENT total sits ABOVE the one-lap readout (user 2026-11-15 read the
        # 1:46 lap time as the 22 km total when it sat below the long readout)
        self._event_lbl = QLabel('')
        self._event_lbl.setStyleSheet(
            "color:#FFD600; font-family:'Consolas','SF Mono',monospace;"
            'font-size:13px; font-weight:bold; background:#0e0e10; padding:8px;'
            'border:1px solid #FFD600; border-radius:4px;')
        self._event_lbl.setWordWrap(True)
        self._event_lbl.setVisible(False)
        side.addWidget(self._event_lbl)
        side.addWidget(self._readout)

        # CORNER ISOLATION (user 2026-10-11): every corner the lap sim cut out,
        # numbered on the map above; the pooled line is what the Ackermann-
        # from-laps method designs for.
        from PyQt6.QtWidgets import QTableWidget, QAbstractItemView
        cl = QLabel('Corners (isolated: |lat g| > 0.30 g)')
        cl.setStyleSheet('color:#E23B48;font-weight:bold;margin-top:6px;')
        side.addWidget(cl)
        self._corner_tbl = QTableWidget(0, 9)
        self._corner_tbl.setHorizontalHeaderLabels(['#', 'dir', 'in (m)', 'len (m)', 'time (s)', 'mean g', 'peak g', 'R harm (m)', 'v min (kph)'])
        self._corner_tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._corner_tbl.setStyleSheet("color:#e8e8ec; background:#0e0e10; font-family:'Consolas',monospace; font-size:11px; gridline-color:#2a2a2a;")
        self._corner_tbl.verticalHeader().setVisible(False)
        self._corner_tbl.setMaximumHeight(220)
        side.addWidget(self._corner_tbl)
        self._corner_pool = QLabel('')
        # steering effort over THIS lap: the lap's lateral-g trace through the ONE
        # effort chain (vahan.steering.compute_steering_effort, aero per g)
        self._effort_lbl = QLabel('')
        self._effort_lbl.setStyleSheet("color:#ffd200; font-family:'Consolas',monospace; font-size:11px;")
        self._effort_lbl.setWordWrap(True)
        self._corner_pool.setStyleSheet("color:#cfcfd4; font-family:'Consolas',monospace; font-size:11px;")
        self._corner_pool.setWordWrap(True)
        side.addWidget(self._corner_pool)
        side.addWidget(self._effort_lbl)

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
        # resizable, never clipped (gui/side_panel.py): was a fixed 300 px
        # box narrower than its widest row, with no splitter to widen it
        from gui.side_panel import SideScroll, side_split
        side_scroll = SideScroll(side_inner, 340)
        side_scroll.setStyleSheet('QScrollArea { border:0; background:#000; }')
        self._side_scroll = side_scroll

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
        # any other track JSON dropped into tracks/ (e.g. a local, untracked
        # course built from logged data) — labelled by its own "name" field
        import glob as _glob, json as _json
        known = {f for _, f in _TRACKS}
        for p in sorted(_glob.glob(os.path.join(base, 'tracks', '*.json'))):
            fname = os.path.basename(p)
            if fname in known:
                continue
            try:
                label = str(_json.load(open(p)).get('name', fname))
            except Exception:
                continue
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
        nav_hint = QLabel('hover = read values · scroll = zoom · drag = pan · '
                          'double-click = reset · "expand ↗" = open big')
        nav_hint.setStyleSheet('color:#8a8a92; font-size:10px;')
        nav_hint.setWordWrap(True)   # wraps instead of holding the page 420 px wider
        topbar.addWidget(nav_hint)
        self._popout_btn = QPushButton('Open graphs in a big window ↗')
        self._popout_btn.setToolTip(
            'Every graph shown here (and the track map) in its own large, '
            'resizable window with zoom-box / pan / reset / save tools.  '
            'The distance graphs zoom together.')
        self._popout_btn.setStyleSheet(
            'QPushButton { background:#15151a; color:#FFD600; padding:4px 10px;'
            ' border:1px solid #FFD600; border-radius:4px; font-size:11px; }'
            'QPushButton:hover { background:#2a2410; }')
        self._popout_btn.clicked.connect(lambda: self.open_popout())
        topbar.addWidget(self._popout_btn)
        # Plot anything against anything from this lap (user 2026-11-10) — the
        # same free plotter that sits under the 3-D model, lap source only.
        self._free_btn = QPushButton('Plot anything (X vs Y) ↗')
        self._free_btn.setToolTip('Pick any lap channel for X and any for Y (speed, g, downforce, '
                                  'ride height, rpm, path radius, straight/corner flag, the full-solve '
                                  'channels...), with your choice of units.')
        self._free_btn.setStyleSheet(self._popout_btn.styleSheet())
        self._free_btn.clicked.connect(self._open_free_plot)
        topbar.addWidget(self._free_btn)
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
        self._canvas = ReadableCanvas(self._fig)
        right.addWidget(self._canvas, 1)
        self._side_split = side_split(side_scroll, right, 340, key='laptime')
        root.addWidget(self._side_split, 1)

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
            self._hover = self._canvas.hover
        except Exception:
            pass

    def _open_free_plot(self):
        from gui.free_plot import open_free_plot_window
        open_free_plot_window(self._main, source='lap', parent=self)

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
        key = self._expand_hit(event)            # 'expand ↗' tag on a graph
        if key is not None and event.button == 1:
            self._pan = None
            if key == 'track':
                self.open_popout([], with_track=True)
            else:
                self.open_popout([key], with_track=False)
            return
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
        """Show the active front/rear tire under the dropdowns."""
        if not hasattr(self, '_tire_active'):
            return
        tf = getattr(self._main, '_tire_model', None)
        tr = getattr(self._main, '_tire_model_rear', None)
        fid = getattr(tf, 'tire_id', None) or '(parametric fallback)'
        if tr is None:
            self._tire_active.setText(f'F+R: {fid}')
        else:
            rid = getattr(tr, 'tire_id', None) or '(parametric)'
            self._tire_active.setText(f'SPLIT  F: {fid}\n       R: {rid}')

    def _tire_path(self, idx: int):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base, 'tire_data', self._tire_rounds[idx][1])

    def _after_tire_change(self, msg: str):
        self._refresh_tire_active()
        had = self._result is not None
        self._status.setText(msg + ('  · re-simulating…' if had
                                    else '  · hit Sim to compare'))
        if had:
            self._on_sim()                        # instant A/B

    def _on_tire_front(self, combo_idx: int):
        """Front-axle tire pick.  Index 0 is the '(loaded tire)' placeholder."""
        idx = combo_idx - 1
        if not (0 <= idx < len(self._tire_rounds)):
            return
        try:
            self._main._on_tire_file(self._tire_path(idx))   # sets _tire_model
        except Exception as e:
            self._status.setText(f'Front tire load failed: {e}')
            return
        self._after_tire_change(f'Front tire → {self._tire_rounds[idx][0]}')

    def _on_tire_rear(self, combo_idx: int):
        """Rear-axle tire pick.  Index 0 = '(same as front)' -> no split."""
        try:
            if combo_idx == 0:
                self._main._set_rear_tire(None)
                self._after_tire_change('Rear tire → same as front')
                return
            idx = combo_idx - 1
            if not (0 <= idx < len(self._tire_rounds)):
                return
            self._main._set_rear_tire(self._tire_path(idx))
            self._after_tire_change(f'Rear tire → {self._tire_rounds[idx][0]}')
        except Exception as e:
            self._status.setText(f'Rear tire load failed: {e}')

    def _project_aero(self):
        """(Cl·A m², rear share %) of the car's aero package when Apply Aero
        is on (Dynamics panel F_ref / V_ref / ρ / CoP), else None."""
        try:
            m = self._main
            if not getattr(m, '_aero_active', False):
                return None
            cfg = m._dynamics_panel.get_custom_aero_params()
            v = float(cfg['V_ref_kph']) / 3.6
            rho = float(cfg['air_density'])
            if v <= 0 or rho <= 0 or float(cfg['F_ref_N']) <= 0:
                return None
            return (2.0 * float(cfg['F_ref_N']) / (rho * v * v),
                    float(cfg['cop_rear_pct']))
        except Exception:
            return None

    def aero_source(self) -> str:
        """'car' (the Dynamics panel package) or 'page' (typed boxes)."""
        try:
            return self._aero_src.currentData() or 'car'
        except Exception:
            return 'page'

    def set_aero_source(self, src: str):
        i = self._aero_src.findData(src)
        if i >= 0:
            self._aero_src.setCurrentIndex(i)
        self._apply_aero_source()

    def _apply_aero_source(self):
        """'car': copy the car's package into the (locked) boxes — re-read
        every Sim, so a changed car is always what runs.  Apply Aero off =
        Cl·A 0.  'page': unlock the boxes."""
        car_src = self.aero_source() == 'car'
        if car_src:
            car = self._project_aero()
            for sb, val in ((self._cla, car[0] if car else 0.0),
                            (self._cop, car[1] if car else float(self._cop.value()))):
                sb.blockSignals(True); sb.setValue(float(val)); sb.blockSignals(False)
        self._cla.setEnabled(not car_src)
        self._cop.setEnabled(not car_src)

    def _use_car_aero(self):
        car = self._project_aero()
        if car is None:
            self._status.setText('The car has no aero package switched on '
                                 '(Dynamics panel → Apply Aero).')
            return
        self.set_aero_source('car')
        self._status.setText(f"Aero from the car: Cl·A {car[0]:.2f} m², "
                             f"{car[1]:.0f} % rear.  Press Sim.")

    def _update_rh_cap(self):
        """Check (and size for) the ride-height cap at the reference speed:
        does the current setup hold ride height >= cap, and if not, what ride
        rate / spring rate would?"""
        if not hasattr(self, '_rh_cap_readout'):
            return
        try:
            from vahan.laptime import (downforce_at_speed_N, aero_heave_mm,
                                       required_ride_rate_Npm,
                                       wheel_rate_from_ride_rate_Npm)
            veh = self._main._build_dynamics_solver()._veh
            v = float(self._rh_refspeed.value()) / 3.6
            cla = float(self._cla.value()); cop = float(self._cop.value()) / 100.0
            rho = float(self._rho.value())
            D = downforce_at_speed_N(cla, v, rho)
            lines = [f'@ {self._rh_refspeed.value():.0f} kph: {D:.0f} N down']
            ok_all = True
            for tag, Daxle, kr, rh0, cap, mr, spr in (
                ('F', D * (1 - cop), veh.ride_rate_front_Npm, self._rh_f.value(),
                 self._rh_cap_f.value(), veh.motion_ratio_front,
                 veh.spring_rate_front_Npm),
                ('R', D * cop, veh.ride_rate_rear_Npm, self._rh_r.value(),
                 self._rh_cap_r.value(), veh.motion_ratio_rear,
                 veh.spring_rate_rear_Npm)):
                heave = aero_heave_mm(Daxle, kr)
                rh = rh0 - heave
                if rh + 1e-6 >= cap:
                    lines.append(f'{tag}: {rh:.0f} mm  ✓ (cap {cap:.0f})')
                else:
                    ok_all = False
                    max_heave = max(rh0 - cap, 0.1)
                    rr_need = required_ride_rate_Npm(Daxle, max_heave)
                    wr_need = wheel_rate_from_ride_rate_Npm(
                        rr_need, veh.tire_rate_Npm)
                    if not np.isfinite(wr_need):
                        lines.append(f'{tag}: {rh:.0f} mm  ✗ under cap '
                                     f'{cap:.0f} — TIRE-limited (stiffer tire/'
                                     f'lower aero needed)')
                    else:
                        # two levers to hold the cap: stiffer SPRING (const MR)
                        # OR a higher MR at compression (the IK nonlinear-MR
                        # target_hi — design a progressive MR rising to this).
                        spr_need = wr_need / max(mr * mr, 1e-6)
                        mr_need = (wr_need / max(spr, 1e-6)) ** 0.5
                        lines.append(
                            f'{tag}: {rh:.0f} mm ✗ under cap {cap:.0f} by '
                            f'{cap-rh:.0f}. Hold it via spring ≥ '
                            f'{spr_need/175.127:.0f} lbf/in (now '
                            f'{spr/175.127:.0f}) OR MR→{mr_need:.2f} at '
                            f'compression (now {mr:.2f}) — IK progressive '
                            f'target_hi')
            self._rh_cap_readout.setText('\n'.join(lines))
            self._rh_cap_readout.setStyleSheet(
                'font-size:10px; color:%s;' % ('#7a9a5a' if ok_all else '#c08a3a'))
        except Exception:
            self._rh_cap_readout.setText('')

    def _show_diff_info(self):
        """Popup explaining the differential (same look as the suspension
        panels' ⓘ dialogs)."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextBrowser
        try:
            from gui import section_info
            html = section_info.DIFFERENTIAL
        except Exception:
            html = '<p>Differential info unavailable.</p>'
        dlg = QDialog(self)
        dlg.setWindowTitle('ⓘ  Differential (Drexler FSAE LSD)')
        dlg.resize(640, 720)
        dlg.setStyleSheet('QDialog { background:#0a0a0a; }'
                          'QTextBrowser { background:#0a0a0a; color:#e0e0e0; '
                          'border:none; font-size:12px; }')
        lay = QVBoxLayout(dlg)
        tb = QTextBrowser(); tb.setOpenExternalLinks(False); tb.setHtml(html)
        lay.addWidget(tb)
        btn = QPushButton('Close'); btn.clicked.connect(dlg.accept)
        from gui.panels import BTN_PRIMARY
        btn.setStyleSheet(BTN_PRIMARY)
        lay.addWidget(btn)
        dlg.exec()

    def _on_diff_changed(self, *_):
        """Push the differential config onto the live car, update the readout,
        and (if a lap exists) re-sim so the balance effect is immediate."""
        from vahan.differential import Differential
        kinds = {0: 'salisbury', 1: 'open', 2: 'spool'}
        kind = kinds.get(self._diff_type.currentIndex(), 'salisbury')
        p, c = self._diff_opts[self._diff_ramp.currentIndex()][1]
        d = Differential(kind, p, c, float(self._diff_preload.value()))
        self._main._diff = d
        self._main._engine_braking_Nm = float(self._diff_engbrake.value())
        # gray out ramp/preload for open/spool (no ramp tuning there)
        is_lsd = (kind == 'salisbury')
        self._diff_ramp.setEnabled(is_lsd)
        self._diff_preload.setEnabled(is_lsd)
        self._diff_engbrake.setEnabled(is_lsd)
        # readout: locking % + a representative exit-understeer yaw moment
        try:
            _ss = self._main._build_dynamics_solver()
            veh = _ss._veh
            r_t = max(veh.tire_radius_m, 1e-3)
            track = float(getattr(veh, 'rear_track_m', 1.2))
            # the car's own rear tyre on THE project grip scale (no 1.5 literal)
            mu0 = float(_ss._tire_rear.peak_mu(700.0, 0.0)) * float(_ss._mu_scale)
            cap = mu0 * veh.total_mass_kg * 9.80665 * (1.0 - veh.front_weight_fraction)
            # ~200 Nm axle drive torque is a typical FSAE corner-exit value
            mz_exit = d.yaw_moment_Nm(200.0, track, r_t, True, max_bias_N=cap)
            mz_entry = d.yaw_moment_Nm(0.0, track, r_t, False, max_bias_N=cap)
        except Exception:
            mz_exit = mz_entry = 0.0
        # the DECISION numbers: actual understeer-gradient shift this diff adds
        # vs an open diff, at a power-on and a trailing-throttle corner.
        du_pwr = du_coast = 0.0
        try:
            from vahan.differential import Differential as _D

            def ug(diff, lon):
                self._main._diff = diff
                return self._main._build_dynamics_solver().solve(
                    1.3, lon).understeer_gradient_deg
            u_open_p, u_d_p = ug(_D('open'), 0.4), ug(d, 0.4)
            u_open_c, u_d_c = ug(_D('open'), -0.4), ug(d, -0.4)
            du_pwr, du_coast = u_d_p - u_open_p, u_d_c - u_open_c
        except Exception:
            pass
        finally:
            self._main._diff = d           # restore the chosen diff
        if kind == 'open':
            self._diff_readout.setText('Open — free inner/outer (least '
                                       'understeer, least off-corner traction; '
                                       'inner wheel spins under power).')
        elif kind == 'spool':
            self._diff_readout.setText(
                f'Spool — fully locked.  vs open: {du_pwr:+.2f}° understeer on '
                f'exit, {du_coast:+.2f}° on entry (max stability, draggy).')
        else:
            self._diff_readout.setText(
                f'power {d.power_locking_pct:.0f}% / coast '
                f'{d.coast_locking_pct:.0f}% lock\n'
                f'vs open: {du_pwr:+.2f}° understeer on EXIT · '
                f'{du_coast:+.2f}° on ENTRY (+ = more understeer/stable)')
        had = self._result is not None
        self._status.setText('Diff → ' + d.describe()
                             + ('  · re-simulating…' if had else ''))
        if had:
            self._on_sim()

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
            f'· {t.n} stations ({t.length_m / max(t.n - 1, 1):.1f} m)')
        self._redraw()

    # ── sim ──────────────────────────────────────────────────────────────
    def _on_sim(self):
        if self._track is None:
            self._status.setText('No track loaded.')
            return
        try:
            sim = self.build_sim()
        except Exception as e:
            self._status.setText(f'Could not build the car: {e}')
            return
        self._btn.setEnabled(False)
        self._last_sim = sim               # the lap's car, for 'MMD from this lap'
        self._status.setText('Simulating…')
        self._worker = _SimWorker(sim, self._track,
                                  int(self._ndetail.value()),
                                  method=self.method())
        self._worker.progress.connect(
            lambda m, p: self._status.setText(f'{m}  ({p}%)'))
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    # ── tyre temperature (vahan/tyre_temperature.py) ─────────────────────
    def _build_tyre_temp(self, side, hdr, grid, spin):
        """TYRE TEMPERATURE inputs.  The data (grip vs temperature, the heat
        model's fitted constants) comes from a tyre_temperature_*.json next
        to the tyre data; every number is shown and editable here."""
        from vahan import tyre_temperature as TT
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._tt_files = TT.find_data_files(base)
        self._tt_data = {}
        self._tt_measured = None
        hdr('TYRE TEMPERATURE')
        g = grid()
        g.addWidget(QLabel('Mode:'), 0, 0)
        self._tt_mode = QComboBox()
        self._tt_mode.addItem('off (tyre model as tested)', 'off')
        self._tt_mode.addItem('fixed tyre temperature', 'fixed')
        self._tt_mode.addItem('estimated from ambient (laps)', 'estimated')
        self._tt_mode.setToolTip(
            'off: the tyre grip exactly as the tyre data was tested (the lap '
            'is unchanged).\nfixed: every tyre at the temperature below.\n'
            'estimated: the tyres start at the start temperature, heat up '
            'from the sliding work of each lap and cool to the air and road; '
            "each lap runs on the previous lap's mean temperature.")
        g.addWidget(self._tt_mode, 0, 1, 1, 3)
        g.addWidget(QLabel('Data:'), 1, 0)
        self._tt_file = QComboBox()
        for p in self._tt_files:
            self._tt_file.addItem(os.path.basename(p)[len('tyre_temperature_'):-5], p)
        if not self._tt_files:
            self._tt_file.addItem('— no tyre_temperature_*.json in tire_data/ —', None)
            self._tt_mode.setEnabled(False)
        g.addWidget(self._tt_file, 1, 1, 1, 3)
        self._tt_amb = spin(g, 2, 0, 'Ambient:', -20, 60, 20.0, 1, 1.0, ' °C',
                            'Air temperature (your input).')
        self._tt_track = spin(g, 2, 1, 'Track:', -20, 80, 25.0, 1, 1.0, ' °C',
                              'Track surface temperature (your input).')
        self._tt_fixed = spin(g, 3, 0, 'Fixed tyre:', -20, 140, 50.0, 1, 1.0, ' °C',
                              "'fixed' mode: tyre surface temperature.")
        self._tt_start = spin(g, 3, 1, 'Start tyre:', -20, 140, 20.0, 1, 1.0, ' °C',
                              "'estimated' mode: tyre temperature at the start "
                              '(= ambient for cold tyres, higher with warmers).')
        g.addWidget(QLabel('Laps:'), 4, 0)
        self._tt_laps = QSpinBox()
        self._tt_laps.setRange(1, 10); self._tt_laps.setValue(3)
        self._tt_laps.setToolTip("'estimated': laps run; the LAST one is shown "
                                 '(each lap takes as long as one normal run).')
        g.addWidget(self._tt_laps, 4, 1)
        self._tt_cold = spin(g, 5, 0, 'Cold loss:', 0, 50, 5.0, 1, 0.5, ' %/10°C',
                             'ASSUMED grip lost per 10 °C BELOW the tested '
                             'temperature range (no tyre data there).  Seeded '
                             'from the data file.')
        self._tt_hot = spin(g, 5, 1, 'Hot loss:', 0, 50, 5.0, 1, 0.5, ' %/10°C',
                            'ASSUMED grip lost per 10 °C ABOVE the tested range.')
        self._tt_heat = {}
        for i, (key, lab, dec, tip) in enumerate((
                ('a_K_per_J', 'Heat-in:', 5, 'surface °C per joule of sliding work (fitted)'),
                ('k1_per_s', 'Surf→body:', 4, 'surface-to-body heat flow, 1/s (fitted)'),
                ('h0_per_s', 'Cool still:', 4, 'surface cooling at rest, 1/s (fitted)'),
                ('h1_per_m', 'Cool/speed:', 5, 'extra surface cooling per m/s, 1/m (fitted)'),
                ('k2_per_s', 'Body←surf:', 4, 'body warming from the surface, 1/s (fitted)'),
                ('hb_per_s', 'Body cool:', 5, 'body cooling, 1/s (fitted)'),
                ('road_share', 'Road share:', 2, 'share of the cooling that goes to the '
                 'road (the rest to the air)'),
                ('slip_peak_deg', 'Slip @ peak:', 1, 'slip angle at peak grip, deg (sets '
                 'the sliding speed = heat in)'),
                ('slip_ratio_peak', 'Slip ratio:', 3, 'braking/drive slip ratio at peak '
                 'grip (ASSUMED: the tested 16 in tyres have no braking/drive data)'))):
            sb = spin(g, 6 + i // 2, i % 2, lab, 0.0, 100.0, 0.0, dec,
                      10 ** -dec * 10, '', tip)
            self._tt_heat[key] = sb
        self._tt_note = QLabel('')
        self._tt_note.setWordWrap(True)
        self._tt_note.setStyleSheet('color:#aaa; font-size:10px;')
        side.addWidget(self._tt_note)
        self._tt_curve_btn = QPushButton('Grip vs tyre temperature curve…')
        self._tt_curve_btn.setToolTip('The grip multiplier the lap sim applies at each '
                                      'tyre temperature (measured range + the assumed '
                                      'cold/hot loss boxes above).')
        self._tt_curve_btn.clicked.connect(
            lambda: __import__('gui.tyre_temp_plot', fromlist=['open_window']).open_window(self))
        side.addWidget(self._tt_curve_btn)
        self._tt_file.currentIndexChanged.connect(self._tt_load)
        self._tt_mode.currentIndexChanged.connect(self._tt_enable)
        self._tt_load()
        self._tt_enable()

    def _tt_load(self):
        from vahan import tyre_temperature as TT
        p = self._tt_file.currentData()
        if not p:
            self._tt_note.setText('No tyre-temperature data file: put a '
                                  'tyre_temperature_*.json in tire_data/.')
            return
        try:
            d = TT.load_data(p)
        except Exception as e:
            self._tt_note.setText(f'Tyre-temperature file unreadable: {e}')
            return
        self._tt_data = d
        st = TT.TyreTempSettings.from_data(d)
        self._tt_cold.setValue(st.cold_loss_pct_per_10C)
        self._tt_hot.setValue(st.hot_loss_pct_per_10C)
        sp = d.get('thermal', {}).get('slip_peak_deg')
        if sp is not None:
            st.slip_peak_deg = float(sp)
        for k, sb in self._tt_heat.items():
            sb.setValue(float(getattr(st, k)))
        gc = d['grip_curve']; lo, hi = gc['measured_C']
        self._tt_measured = (float(lo), float(hi))
        win = d.get('maker_window_C')
        th = d['thermal']
        self._tt_note.setText(
            f'Grip vs tyre temperature MEASURED {lo:.0f}-{hi:.0f} °C (grip x1 = '
            f'{gc["ref_T_C"]:.0f} °C, the temperature the tyre data belongs to); '
            f'outside that the cold/hot loss is ASSUMED.'
            + (f'  Maker window {win[0]:.0f}-{win[1]:.0f} °C.' if win else '')
            + f'  Heat model vs the test rig: {th.get("fit_rms_C", float("nan")):.1f} °C '
            f'rms ({th.get("holdout_rms_C", float("nan")):.1f} °C on held-out runs).')

    def _tt_enable(self):
        m = self._tt_mode.currentData()
        self._tt_fixed.setEnabled(m == 'fixed')
        for w in (self._tt_start, self._tt_laps):
            w.setEnabled(m == 'estimated')

    def tyre_temp_settings(self):
        """TyreTempSettings from the boxes, or None when off / no data."""
        from vahan import tyre_temperature as TT
        m = self._tt_mode.currentData() if hasattr(self, '_tt_mode') else 'off'
        if m in (None, 'off') or not self._tt_data:
            return None
        over = {k: float(sb.value()) for k, sb in self._tt_heat.items()}
        return TT.TyreTempSettings.from_data(
            self._tt_data, self._tt_file.currentText(), mode=m,
            ambient_C=float(self._tt_amb.value()), track_C=float(self._tt_track.value()),
            fixed_C=float(self._tt_fixed.value()), start_C=float(self._tt_start.value()),
            laps=int(self._tt_laps.value()),
            cold_loss_pct_per_10C=float(self._tt_cold.value()),
            hot_loss_pct_per_10C=float(self._tt_hot.value()), **over)

    def build_sim(self) -> LapSimulator:
        """The LapSimulator exactly as the Simulate button configures it
        (live car + every box on this page).  Split out of _on_sim so a
        headless caller runs the SAME configuration the user sees."""
        if self.aero_source() == 'car':
            self._apply_aero_source()      # the car as it is NOW
            try:
                self._upd_aero_ref()
            except Exception:
                pass
        ss = self._main._build_dynamics_solver()
        sim = LapSimulator(
            ss,
            cla_m2=float(self._cla.value()),
            cda_m2=float(self._cda.value()),
            air_density=float(self._rho.value()),
            aero_cop_rear_frac=float(self._cop.value()) / 100.0,
            grip_scale=float(self._main.grip_scale()),   # THE project scale
            static_rh_front_mm=float(self._rh_f.value()),
            static_rh_rear_mm=float(self._rh_r.value()),
        )
        # remembered for the aero graph title + front/rear split
        self._last_aero = (float(self._cla.value()), float(self._cda.value()),
                           float(self._cop.value()))
        sim.set_util_cap(float(self._util_cap.value()))
        sim.set_tyre_temperature(self.tyre_temp_settings())
        sim.set_gearbox(
            [sb.value() for sb in self._gear_spins],
            primary_ratio=float(self._primary.value()),
            final_drive=float(self._final.value()),
            redline_rpm=float(self._redline.value()),
        )
        # 1a rotating inertia, 1b shift realism, 1c torque curve — all three
        # are sim-local overrides seeded from the car, exactly like Cd·A.
        sim.set_rotating_inertia(float(self._I_wf.value()),
                                 float(self._I_wr.value()),
                                 float(self._I_eng.value()))
        sim.set_shift_model(float(self._shift_time.value()),
                            float(self._shift_int.value()),
                            float(self._shift_hyst.value()) / 100.0)
        curve = self._torque_curve()
        if curve is not None:
            sim.set_torque_curve(curve[0], curve[1], curve[2])
        else:
            sim.set_torque_curve(None, None)
        return sim

    def method(self) -> str:
        """'ggv' (the car's G-G limits, default) or 'pointmass' (old)."""
        try:
            return self._method.currentData() or 'ggv'
        except Exception:
            return 'ggv'

    def open_ggmap(self):
        """The lap's points over the car's G-G limits (method 'ggv')."""
        res = self._result
        if res is None or getattr(res, 'ggv', None) is None:
            self._status.setText("Run the lap with Method = the car's G-G "
                                 'limits first.')
            return None
        from PyQt6.QtWidgets import QDialog, QVBoxLayout
        dlg = QDialog(self)
        dlg.setWindowTitle("Lap on the car's G-G limits")
        dlg.resize(900, 760)
        lay = QVBoxLayout(dlg)
        fig = Figure(figsize=(9, 7.6))
        draw_ggmap(fig, res, self._track)
        cv = FigureCanvas(fig)
        lay.addWidget(cv)
        cv.draw()
        self._ggmap_dlg = dlg
        dlg.show()
        return dlg

    # ── powertrain-realism helpers (1a / 1b / 1c) ────────────────────────
    def _total_ratio_1st(self) -> float:
        gs = [sb.value() for sb in self._gear_spins if sb.value() > 0]
        return (float(self._primary.value()) * float(self._final.value())
                * (max(gs) if gs else 1.0))

    def _update_inertia_note(self):
        """Show the equivalent mass the inertias amount to in FIRST gear —
        the number that makes the engine term's ratio-squared amplification
        visible instead of buried."""
        try:
            veh = self._main._build_dynamics_solver()._veh
            m, r = float(veh.total_mass_kg), float(veh.tire_radius_m)
        except Exception:
            m, r = 286.0, 0.203
        ratio = self._total_ratio_1st()
        I = 2.0 * (self._I_wf.value() + self._I_wr.value())
        m_eq = (I + self._I_eng.value() * ratio ** 2) / max(r * r, 1e-6)
        self._inertia_readout.setText(
            f'≈ {m_eq:.0f} kg equivalent mass in 1st ({100 * m_eq / m:.0f}% of '
            f'the car), of which the engine is '
            f'{100 * (self._I_eng.value() * ratio ** 2) / max(I + self._I_eng.value() * ratio ** 2, 1e-9):.0f}% '
            f'(ratio {ratio:.2f}, squared = {ratio ** 2:.0f}).  ASSUMED until '
            f'measured.')

    def _torque_curve(self):
        """(rpm[], Nm[], label) or None — parsed from the box, or the ASSUMED
        generic shape scaled to this car's own peak power."""
        if not self._tq_use.isChecked():
            return None
        txt = self._tq_text.text().strip()
        pairs = []
        for tok in txt.replace(';', ',').split(','):
            if ':' not in tok:
                continue
            a, b = tok.split(':', 1)
            try:
                pairs.append((float(a), float(b)))
            except ValueError:
                continue
        if len(pairs) >= 2:
            pairs.sort()
            import vahan.engine as _VE
            _cap = getattr(self._main, '_car', {}).get('engine_power_cap_hp')
            r_, t_ = _VE.apply_power_cap([p[0] for p in pairs], [p[1] for p in pairs],
                                         float(_cap) if _cap not in (None, '') else None)
            return (list(r_), list(t_), 'user-entered dyno points'
                    + (f', capped at {float(_cap):g} hp' if len(r_) != len(pairs) else ''))
        # Default engine = the SDM26 curve with the CALIBRATION the user chose
        # on the Engine page (saved in the project car dict; 'corrected' if
        # unset).  ONE engine source.  Only if the data files are missing do we
        # fall back to the ASSUMED generic shape scaled to peak power.
        import vahan.engine as _VE
        car = getattr(self._main, '_car', {})
        # every knob passed EXPLICITLY (no module-global mutation — the old
        # global writes never reached ve_target_curve; audit item 21)
        # ONE source (vahan.engine.car_engine_curve): method + knobs, the chosen
        # named table, and the user's power cap applied at use time
        c = _VE.car_engine_curve(car)
        if c is not None:
            return list(c[0]), list(c[1]), c[2]
        try:
            veh = self._main._build_dynamics_solver()._veh
            P_eng = float(veh.wheel_power_W) / max(
                float(veh.drivetrain_efficiency), 1e-3)
            rpm, tq = assumed_torque_curve(P_eng, float(veh.engine_rpm))
            return list(rpm), list(tq), 'CBR600RR 20mm-restricted — ASSUMED shape, scaled to this car'
        except Exception:
            return None

    def _update_tq_note(self):
        c = self._torque_curve()
        if c is None:
            self._tq_note.setText(
                'IN USE: torque plateau then constant power (ASSUMED shape — '
                'no measured curve).')
            return
        rpm, tq, lbl = c
        self._tq_note.setText(
            f'IN USE: {lbl} — {len(rpm)} points, {min(rpm):.0f}–{max(rpm):.0f} '
            f'rpm, peak {max(tq):.1f} N·m at '
            f'{rpm[int(np.argmax(tq))]:.0f} rpm.')

    def _on_fail(self, msg: str):
        self._btn.setEnabled(True)
        self._status.setText(f'Sim failed: {msg}')

    def _on_done(self, res):
        self._btn.setEnabled(True)
        self._result = res
        try:    # the free plotter(s) listen for new lap data
            self._main._notify_free_plot('lap')
        except Exception:
            pass
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
        # straights vs corners (user 2026-11-10, for aero): the lap split into
        # its corners and the straights between them, each pooled as
        # distance / time, plus the share of the lap time above each speed.
        _ss = getattr(res, 'straight_summary', None) or {}
        _bands = _ss.get('speed_bands_pct', {}) or {}
        _straight_txt = (
            f'  straights-only (corners excluded):   avg speed {_ss.get("v_mean_kph", 0.0):5.1f} kph   '
            f'{_ss.get("n_straights", 0)} straights = {_ss.get("time_straight_pct", 0.0):4.1f}% of lap time, '
            f'peak {_ss.get("v_max_kph", 0.0):5.1f} kph\n'
            f'  on the same split, corners avg {_ss.get("v_mean_corner_kph", 0.0):5.1f} kph over '
            f'{_ss.get("time_corner_pct", 0.0):4.1f}% of lap time\n'
            + ('  lap time above ' + '   '.join(f'{k:.0f} kph: {p:4.1f}%' for k, p in sorted(_bands.items())) + '\n'
               if _bands else '')) if _ss and 'error' not in _ss else ''
        self._readout.setText(
            f'LAP TIME   {int(m)}:{s:05.2f}{delta}\n'
            f'sectors    {sect}\n'
            f'avg speed  {res.avg_speed_kph:6.1f} kph   min {res.min_speed_kph:5.1f} / max {res.max_speed_kph:5.1f} kph\n'
            f'corners    avg {res.avg_corner_lat_g:4.2f} g   peak {res.peak_lat_g:4.2f} g   {res.time_cornering_pct:4.1f}% of lap\n'
            f'  cornering-only (straights excluded):  avg speed {res.avg_corner_speed_kph:5.1f} kph   avg radius {res.avg_corner_radius_m:5.1f} m\n'
            + _straight_txt +
            f'long accel peak {res.peak_accel_g:4.2f} g   brake peak {res.peak_brake_g:4.2f} g\n'
            f'max speed  {res.max_speed_kph:6.1f} kph\n'
            f'peak lat   {np.nanmax(res.lat_g):6.2f} g\n'
            f'peak brake {np.nanmin(res.lon_g):6.2f} g\n'
            f'peak accel {np.nanmax(res.lon_g):6.2f} g\n'
            f'top gear   {top_gear}   (geared top {v_top:.0f} kph '
            f'@ {self._redline.value():.0f})\n'
            f'shifts     {res.n_upshifts} up / {res.n_downshifts} down, '
            f'{res.shift_time_lost_s:.2f} s of torque cut\n'
            f'rot inertia{max(res.equiv_mass_kg.values()) if res.equiv_mass_kg else 0.0:7.0f} kg equivalent (ASSUMED)\n'
            + (f"method     the car's G-G limits (full solve, {len(res.ggv.v)} speed(s))\n"
               f'launch     standing start, clutch slips at {res.launch_rpm:.0f} rpm'
               if getattr(res, 'method', '') == 'ggv' and res.ggv is not None
               else 'method     old quick method (point mass)'))
        # min ride height (aero squat) — the "don't bottom out" number
        try:
            rf, rr = res.rh_front_mm, res.rh_rear_mm
            if rf is not None and rr is not None:
                self._readout.setText(self._readout.text() +
                    f'\nmin RH     F {np.nanmin(rf):.0f} / R {np.nanmin(rr):.0f} mm'
                    f'  (from {self._rh_f.value():.0f}/{self._rh_r.value():.0f})')
        except Exception:
            pass
        self._fill_corners(res)
        self._fill_effort(res)
        self._event_res = res
        self._update_event()
        self._redraw()

    def event_totals(self, res=None):
        """(laps, lap_length_m, total_time_s, total_km) for the EVENT inputs,
        or None without a lap.  Distance > 0 wins: laps = ceil(distance / lap
        length).  Every lap counted as THIS lap's time (the sim's lap, incl.
        its standing start if the track is open)."""
        res = res if res is not None else getattr(self, '_event_res', None)
        if res is None:
            return None
        try:
            lap_m = float(np.asarray(res.s_m, float)[-1])
            lap_s = float(res.lap_time_s)
        except Exception:
            return None
        if not (lap_m > 0 and np.isfinite(lap_s)):
            return None
        km = float(self._event_km.value())
        laps = int(np.ceil(km * 1000.0 / lap_m - 1e-9)) if km > 0 else int(self._event_laps.value())
        laps = max(laps, 1)
        return laps, lap_m, laps * lap_s, laps * lap_m / 1000.0

    def _update_event(self):
        lbl = getattr(self, '_event_lbl', None)
        if lbl is None:
            return
        t = self.event_totals()
        if t is None:
            lbl.setVisible(False)
            return
        laps, lap_m, tot_s, tot_km = t
        if laps <= 1 and float(self._event_km.value()) <= 0:
            lbl.setVisible(False)
            return
        mm, ss = divmod(tot_s, 60.0)
        src = (f'{self._event_km.value():.2f} km asked / {lap_m:.0f} m lap, rounded up'
               if self._event_km.value() > 0 else 'laps typed')
        lbl.setText(f'EVENT TOTAL  {int(mm)}:{ss:05.2f}  ({tot_s:.1f} s)\n'
                    f'           = {laps} laps x {tot_s / laps:.2f} s per lap\n'
                    f'distance   {tot_km:.2f} km   ({src})\n'
                    f'every lap counted as this lap (incl. its start)')
        lbl.setVisible(True)

    def _fill_effort(self, res):
        """Hand torque over the lap: time-weighted average (whole lap / corners only), lowest in a corner, peak,
        and the force at the rim when the project names a steering-wheel OD.  Same chain as the steering plots."""
        lbl = getattr(self, '_effort_lbl', None)
        if lbl is None:
            return
        try:
            import numpy as np
            from vahan.steering import compute_steering_effort
            m = self._main
            ss = m._build_dynamics_solver()
            was = m._aero_active; m._aero_active = True
            try:
                e = compute_steering_effort(ss, m._front_hp, m._topology.front.damper_mount.value, m._tire_model,
                                            steer_cfg=dict(m._steer), front_solver=m._solvers['FL'],
                                            aero_Fz_per_g=m._get_aero_Fz_per_g(),
                                            lat_grid=(0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.2, 2.3))
            finally:
                m._aero_active = was
            tv = e.get('T_vs_latg') or []
            if not tv:
                lbl.setText('steering effort: no hand torque (set steer.rack_travel_per_rev_mm)'); return
            gx = np.array([0.0] + [a for a, _ in tv], float); ty = np.array([0.0] + [b for _, b in tv], float)
            g = np.abs(np.asarray(res.lat_g, float)); s = np.asarray(res.s_m, float); v = np.asarray(res.v_ms, float)
            dt = np.diff(s) / np.maximum(0.5 * (v[1:] + v[:-1]), 0.1); gm = 0.5 * (g[1:] + g[:-1]); T = np.interp(gm, gx, ty)
            corner = gm > 0.30
            avg = float(np.sum(T * dt) / np.sum(dt)); cavg = float(np.sum(T[corner] * dt[corner]) / max(np.sum(dt[corner]), 1e-9)) if corner.any() else float('nan')
            low = float(T[corner].min()) if corner.any() else float('nan'); pk = float(T.max()); gpk = float(gm[int(T.argmax())])
            above = float(100 * np.sum(dt[T > 15.0]) / np.sum(dt))
            od = e.get('wheel_od_mm')
            rim = (f'  |  at the rim (Ø{od:.0f} mm): avg {avg / (od / 2000):.0f} N, peak {pk / (od / 2000):.0f} N' if od else '  |  rim force: no steering-wheel OD in the project')
            lbl.setText(f'STEERING EFFORT over this lap (aero per station): avg {avg:.1f} N·m whole lap, {cavg:.1f} in corners, lowest in a corner {low:.1f}, '
                        f'peak {pk:.1f} N·m at {gpk:.2f} g; above 15 N·m for {above:.0f} % of the lap{rim}')
        except Exception as ex:
            lbl.setText(f'steering effort: not computed ({ex})')

    def _fill_corners(self, res):
        """Corner table + pooled line from res.corners / res.corner_summary."""
        from PyQt6.QtWidgets import QTableWidgetItem
        tbl = getattr(self, '_corner_tbl', None)
        if tbl is None:
            return
        cs = list(getattr(res, 'corners', []) or [])
        tbl.setRowCount(len(cs))
        for r, c in enumerate(cs):
            cells = (str(c['index']), c['direction'][0].upper(), f'{c["s_in_m"]:.0f}', f'{c["length_m"]:.0f}', f'{c["time_s"]:.2f}',
                     f'{c["g_mean"]:.2f}', f'{c["g_peak"]:.2f}', f'{c["R_harmonic_m"]:.1f}', f'{c["v_min_kph"]:.0f}')
            for k, txt in enumerate(cells):
                tbl.setItem(r, k, QTableWidgetItem(txt))
        tbl.resizeColumnsToContents()
        sm = getattr(res, 'corner_summary', {}) or {}
        if sm.get('n_corners'):
            self._corner_pool.setText(
                f'{sm["n_corners"]} corners, {sm["corner_time_s"]:.1f} s = {sm["time_cornering_pct"]:.0f} % of the lap  |  pooled (time-weighted): '
                f'{sm["g_mean"]:.2f} g, radius mean {sm["R_mean_m"]:.0f} / median {sm["R_median_m"]:.0f} / harmonic {sm["R_harmonic_m"]:.0f} m  '
                f'(harmonic = the one the steer angle sees)')
        else:
            self._corner_pool.setText('no corners above 0.30 g' + (f'  ({sm["error"]})' if sm.get('error') else ''))

    # ── drawing ──────────────────────────────────────────────────────────
    # ONE drawing routine (_draw_into) feeds both the page canvas and the
    # big pop-out window, so the two can never show different numbers.
    @staticmethod
    def _style(ax, xlabel, ylabel, title, big=False):
        fs_t, fs_l, fs_k = (12, 11, 10) if big else (9.5, 8.5, 8)
        ax.set_facecolor(_BG_AX)
        ax.tick_params(colors=_TXT_DIM, labelsize=fs_k)
        for sp in ax.spines.values():
            sp.set_edgecolor(_SPINE)
        ax.set_xlabel(xlabel, color=_TXT_DIM, fontsize=fs_l)
        ax.set_ylabel(ylabel, color=_TXT_DIM, fontsize=fs_l)
        if getattr(ax, '_lap_compact', False) and ylabel:
            # graph too short for a rotated label: unit goes in the title
            ax.yaxis.label.set_visible(False)
            title = f'{title}   [{ylabel}]'
        ax.set_title(title, color=_TXT, fontsize=fs_t, loc='left')
        ax.grid(True, color=_GRID, lw=0.6)
        ax.set_axisbelow(True)

    @staticmethod
    def _legend(ax, big=False, ncol=4):
        if not [h for h in ax.get_legend_handles_labels()[1]
                if not h.startswith('_')]:
            return
        if big:
            ax.legend(fontsize=9, facecolor='#101014', edgecolor=_SPINE,
                      labelcolor=_TXT, framealpha=0.9,
                      loc='upper left', bbox_to_anchor=(1.005, 1.0),
                      borderaxespad=0.0, handlelength=2.2)
        else:
            # outside the plot, right of it: the data stays readable
            ax.legend(fontsize=7.5, facecolor='#101014', edgecolor=_SPINE,
                      labelcolor=_TXT, framealpha=0.85, ncol=1,
                      loc='upper left', bbox_to_anchor=(1.005, 1.0),
                      borderaxespad=0.0, handlelength=2.0)

    def _checked(self):
        return [k for k, _l, _d in self._GRAPHS
                if self._graph_cbs[k].isChecked()]

    def _shown_keys(self):
        """The graphs on screen right now (Solo wins over the checkboxes)."""
        if self._result is None:
            return []
        solo_key = self._solo.currentData()
        return [solo_key] if solo_key is not None else self._checked()

    def _draw_track(self, fig, tax, big=False):
        t, res = self._track, self._result
        if res is None:
            self._style(tax, 'x (m)', 'y (m)',
                        f'{t.name} — {t.length_m:.0f} m  (press Sim to '
                        f'colour it by speed)', big)
            tax.plot(t.x_m, t.y_m, color='#8a8a92', lw=2.5)
        else:
            self._style(tax, 'x (m)', 'y (m)',
                        f'{t.name} — speed around the lap  '
                        f'(lap {res.lap_time_s:.2f} s)', big)
            kph = np.asarray(res.v_ms, float) * 3.6
            pts = np.column_stack([t.x_m, t.y_m]).reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap=_SPEED_CMAP, linewidths=4.0 if big else 3.2,
                                capstyle='round', joinstyle='round', zorder=3)
            lc.set_array(0.5 * (kph[1:] + kph[:-1]))
            lc.set_clim(float(np.nanmin(kph)), float(np.nanmax(kph)))
            tax.add_collection(lc)
            cb = fig.colorbar(lc, ax=tax, fraction=0.025, pad=0.01)
            cb.set_label('speed (km/h)', color=_TXT_DIM,
                         fontsize=10 if big else 8)
            cb.ax.tick_params(colors=_TXT_DIM, labelsize=9 if big else 7)
            cb.outline.set_edgecolor(_SPINE)
            # isolated corners: numbered at the apex
            for c in getattr(res, 'corners', []) or []:
                if np.isfinite(c.get('x_apex_m', np.nan)):
                    tax.annotate(str(c['index']), (c['x_apex_m'], c['y_apex_m']),
                                 xytext=(0, 8), textcoords='offset points',
                                 ha='center', fontsize=8 if big else 7,
                                 color='#ffd200', zorder=6,
                                 bbox=dict(boxstyle='round,pad=0.15', fc='#000000',
                                           ec='#ffd200', lw=0.5, alpha=0.85))
        tax.plot(t.x_m[0], t.y_m[0], marker='^', color='#FFFFFF', ms=11,
                 ls='none', zorder=7)
        tax.annotate('start', (t.x_m[0], t.y_m[0]), xytext=(8, -12),
                     textcoords='offset points', color='#FFFFFF',
                     fontsize=8 if big else 7)
        tax.set_aspect('equal', adjustable='datalim')
        tax.autoscale_view()

    def _draw_graph(self, ax, key, big=False):
        """Draw one lap graph into `ax`.  Every graph's x axis is the
        distance along the lap; the y label carries the unit."""
        res = self._result
        s = np.asarray(res.s_m, float)
        d = res.det_channels
        ds_ = res.det_s_m
        n_det = len(ds_) if ds_ is not None else 0
        lw = 1.9 if big else 1.5
        X = _XLABEL
        if key == 'speed':
            self._style(ax, X, 'speed (km/h)', 'Speed', big)
            kph = np.asarray(res.v_ms, float) * 3.6
            top = float(np.nanmax(kph)) * 1.12
            cap = np.asarray(res.v_corner_cap_ms, float) * 3.6
            ax.plot(s, np.where(cap < top, cap, np.nan), color='#9a9aa2',
                    lw=1.0, ls='--', label='fastest the grip allows here')
            ax.plot(s, kph, color='#FFD600', lw=lw + 0.2, label='speed')
            ax.set_ylim(0, top)
            self._legend(ax, big)
        elif key == 'gg':
            self._style(ax, X, 'acceleration (g)',
                        'Sideways g (+ left-hand turn, − right-hand) and '
                        'forward g (+ speeding up, − braking)', big)
            k = np.asarray(self._track.kappa_1pm, float)
            lat_s = np.sign(k) * np.asarray(res.lat_g, float)
            cap_g = getattr(res, 'ay_cap_local_ms2', None)
            if cap_g is not None:
                cap_g = np.asarray(cap_g, float) / 9.80665
                ax.plot(s, cap_g, color='#FFFFFF', lw=0.8, ls=':', alpha=0.7,
                        label='sideways grip limit')
                ax.plot(s, -cap_g, color='#FFFFFF', lw=0.8, ls=':',
                        alpha=0.7, label='_limit_neg')
            ax.plot(s, lat_s, color='#42A5F5', lw=lw, label='sideways g')
            ax.plot(s, res.lon_g, color='#E53935', lw=lw - 0.2,
                    label='forward g')
            ax.axhline(0, color='#55555c', lw=0.7)
            self._legend(ax, big)
        elif key == 'util':
            self._style(ax, X, 'grip used (1.0 = at the limit)',
                        f'Tyre grip used, each tyre  (full suspension solve '
                        f'at {n_det} points)', big)
            for c in ('FL', 'FR', 'RL', 'RR'):
                ax.plot(ds_, d[f'util_{c}'], color=_CC[c], ls=_LS[c],
                        lw=lw - 0.2, label=_CORNER_WORDS[c])
            # the PAIR of tyres on each axle = the limit the car's G-G
            # limits are built on (one tyre can be over while its partner
            # has grip to spare)
            if 'util_axle_F' in d:
                ax.plot(ds_, d['util_axle_F'], color='#FFD600', lw=lw + 1.2,
                        alpha=0.35, label='front pair')
                ax.plot(ds_, d['util_axle_R'], color='#42A5F5', lw=lw + 1.2,
                        alpha=0.35, label='rear pair')
            ax.axhline(1.0, color='#FFFFFF', lw=0.9, ls='--', alpha=0.6,
                       label='at the limit')
            ax.set_ylim(bottom=0)
            self._legend(ax, big, ncol=4)
        elif key == 'ttemp':
            tf = getattr(res, 'tyre_temp_front_C', None)
            if tf is None:
                self._style(ax, X, 'tyre surface (°C)',
                            'Tyre temperature — off (turn it on in TYRE '
                            'TEMPERATURE and run the lap)', big)
            else:
                kf, kr = res.tyre_grip_factor or (1.0, 1.0)
                self._style(ax, X, 'tyre surface (°C)',
                            f'Tyre surface temperature (grip x{kf:.3f} front, '
                            f'x{kr:.3f} rear on this lap)', big)
                ax.plot(s, tf, color='#FFD600', lw=lw, label='front tyres')
                ax.plot(s, res.tyre_temp_rear_C, color='#42A5F5', lw=lw,
                        label='rear tyres')
                rng = getattr(self, '_tt_measured', None)
                if rng:
                    ax.axhspan(rng[0], rng[1], color='#FFFFFF', alpha=0.06,
                               label=f'grip measured {rng[0]:.0f}-{rng[1]:.0f} °C')
                self._legend(ax, big)
        elif key == 'lltd':
            self._style(ax, X, 'front share (%)',
                        'LLTD — share of the cornering weight shift carried '
                        'by the front axle', big)
            ax.ticklabel_format(axis='y', useOffset=False)
            # only meaningful while cornering: on straights there is no
            # weight shift to share (the ratio is 0/0 noise) — same 0.30 g
            # corner rule as the corner table
            lat_d = np.interp(ds_, s, np.asarray(res.lat_g, float))
            ll = np.where(lat_d >= 0.30, d['lltd_pct'], np.nan)
            ax.plot(ds_, ll, color='#FFD600', lw=lw,
                    label='front share (while cornering > 0.3 g)')
            self._legend(ax, big)
        elif key == 'roll':
            self._style(ax, X, 'roll (deg)',
                        'Body roll  (+ in right-hand turns, − in left-hand)',
                        big)
            ax.plot(ds_, d['roll_deg'], color='#42A5F5', lw=lw, label='roll')
            ax.axhline(0, color='#55555c', lw=0.7)
            self._legend(ax, big)
        elif key == 'susp':
            self._style(ax, X, 'travel (mm, + = compressed)',
                        'Wheel travel each corner, and shock travel '
                        '(= wheel travel × motion ratio)', big)
            for c in ('FL', 'FR', 'RL', 'RR'):
                ax.plot(ds_, d[f'travel_{c}'], color=_CC[c], ls=_LS[c],
                        lw=lw - 0.4, label=f'wheel {_CORNER_WORDS[c]}')
            ax.plot(ds_, d['shock_F_mm'], color='#FFD600', lw=lw + 0.6,
                    alpha=0.55, label='shock, front left')
            ax.plot(ds_, d['shock_R_mm'], color='#FFFFFF', lw=lw + 0.6,
                    alpha=0.55, label='shock, rear left')
            ax.axhline(0, color='#55555c', lw=0.7)
            self._legend(ax, big, ncol=3)
        elif key == 'aero':
            cla, cda, cop = getattr(self, '_last_aero', (0.0, 0.0, 50.0))
            self._style(ax, X, 'force (N)',
                        f'Aero forces — Cl·A {cla:+.2f} m², Cd·A {cda:.2f} m², '
                        f'{cop:.0f} % of the downforce on the rear', big)
            ax.plot(s, res.aero_down_N, color='#42A5F5', lw=lw + 0.4,
                    label='downforce, total')
            ax.plot(s, res.aero_down_N * (1.0 - cop / 100.0),
                    color='#FFFFFF', lw=lw - 0.5, ls='--',
                    label='downforce, front axle')
            ax.plot(s, res.aero_down_N * (cop / 100.0),
                    color='#FFD600', lw=lw - 0.5, ls='-.',
                    label='downforce, rear axle')
            ax.plot(s, res.aero_drag_N, color='#E53935', lw=lw - 0.2,
                    ls=':', label='drag')
            ax.axhline(0, color='#55555c', lw=0.7)
            if abs(np.nanmax(np.abs(res.aero_down_N))) < 1.0:
                ax.text(0.5, 0.7, 'Cl·A = 0 — no downforce in this run',
                        transform=ax.transAxes, ha='center',
                        color=_TXT_DIM, fontsize=10 if big else 9)
            self._legend(ax, big)
        elif key == 'rpm':
            self._style(ax, X, 'engine speed (rpm)',
                        'Engine speed and gear  (red bands = torque cut '
                        'while shifting)', big)
            ax.plot(s, res.engine_rpm, color='#FFD600', lw=lw,
                    label='engine rpm')
            ax.axhline(float(self._redline.value()), color='#E53935',
                       lw=0.9, ls='--', alpha=0.7, label='redline')
            if res.gear is not None and np.nanmax(res.gear) > 0:
                ax2 = ax.twinx()
                ax2.plot(s, res.gear, color='#42A5F5', lw=lw - 0.3,
                         drawstyle='steps-post', label='gear')
                ax2.set_ylabel('gear', color='#42A5F5',
                               fontsize=11 if big else 8.5)
                ax2.tick_params(colors='#42A5F5', labelsize=10 if big else 8)
                gmax = int(np.nanmax(res.gear))
                ax2.set_yticks(range(1, gmax + 1))
                ax2.set_ylim(0.5, gmax + 0.8)
                for sp in ax2.spines.values():
                    sp.set_edgecolor(_SPINE)
                # rpm axis on top so the hover reads rpm; gear drawn behind
                ax2.set_facecolor(_BG_AX)
                ax2.patch.set_visible(True)
                ax.set_zorder(ax2.get_zorder() + 1)
                ax.patch.set_visible(False)
                ax.plot([], [], color='#42A5F5', lw=lw - 0.3, label='gear')
            sh = getattr(res, 'shifting', None)
            if sh is not None and np.any(sh):
                first = True
                for i in np.where(np.asarray(sh) > 0)[0]:
                    ax.axvspan(s[max(i - 1, 0)], s[min(i + 1, len(s) - 1)],
                               color='#E53935', alpha=0.30, lw=0,
                               label='torque cut (shift)' if first else None)
                    first = False
            self._legend(ax, big)
        elif key == 'power':
            self._style(ax, X, 'power (hp)', 'Engine power used to drive',
                        big)
            ax.plot(s, res.power_used_W / 745.7, color='#FFD600', lw=lw,
                    label='power used')
            try:
                pk = float(self._main._dynamics_panel._power_hp.value())
                ax.axhline(pk, color='#E53935', lw=0.9, ls='--',
                           alpha=0.7, label='engine peak')
            except Exception:
                pass
            ax.set_ylim(bottom=0)
            self._legend(ax, big)
        elif key == 'diff':
            self._style(ax, X, 'yaw moment (N·m)',
                        'Differential turning moment  (+ = pushes the car '
                        'straight: understeer on exit, steadier on entry)',
                        big)
            dy = getattr(res, 'diff_yaw_Nm', None)
            if dy is not None:
                ax.fill_between(s, 0, dy, color='#FFD600', alpha=0.20, lw=0)
                ax.plot(s, dy, color='#FFD600', lw=lw, label='diff moment')
            ax.axhline(0, color='#55555c', lw=0.7)
            self._legend(ax, big)
        elif key == 'rideh':
            self._style(ax, X, 'ride height (mm)',
                        'Ride height  (drops as downforce builds)', big)
            rf = getattr(res, 'rh_front_mm', None)
            rr = getattr(res, 'rh_rear_mm', None)
            if rf is not None:
                ax.plot(s, rf, color='#FFD600', lw=lw, label='front')
                ax.axhline(float(np.nanmin(rf)), color='#FFD600', lw=0.8,
                           ls=':', alpha=0.6,
                           label=f'lowest front {np.nanmin(rf):.0f} mm')
            if rr is not None:
                ax.plot(s, rr, color='#42A5F5', lw=lw, label='rear')
                ax.axhline(float(np.nanmin(rr)), color='#42A5F5', lw=0.8,
                           ls=':', alpha=0.6,
                           label=f'lowest rear {np.nanmin(rr):.0f} mm')
            try:
                cf = float(self._rh_cap_f.value())
                cr = float(self._rh_cap_r.value())
                ax.axhline(cf, color='#E53935', lw=1.0, ls='--',
                           label=f'limit front {cf:.0f} mm')
                if abs(cr - cf) > 0.5:
                    ax.axhline(cr, color='#E53935', lw=1.0, ls='-.',
                               label=f'limit rear {cr:.0f} mm')
                lo = min(np.nanmin(rf) if rf is not None else cf,
                         np.nanmin(rr) if rr is not None else cr)
                if lo < max(cf, cr):
                    ax.axhspan(lo - 1, max(cf, cr), color='#E53935',
                               alpha=0.08)
            except Exception:
                pass
            self._legend(ax, big)
        elif key == 'segs':
            # straights (yellow) vs corners (blue) with each segment's mean
            # speed as a dashed level — user 2026-11-10, "give something to aero"
            ss = getattr(res, 'straight_summary', None) or {}
            kph = np.asarray(res.v_ms, float) * 3.6
            self._style(ax, X, 'speed (km/h)',
                        f'Straights vs corners — straights average {ss.get("v_mean_kph", float("nan")):.1f} km/h '
                        f'({ss.get("time_straight_pct", float("nan")):.0f} % of the lap time), corners average '
                        f'{ss.get("v_mean_corner_kph", float("nan")):.1f} km/h', big)
            seg = np.zeros(len(s))
            for c in getattr(res, 'corners', None) or []:
                seg[int(c['i_in']):int(c['i_out']) + 1] = 1.0
            ax.plot(s, kph, color='#55555c', lw=0.8)
            ax.plot(s, np.where(seg < 0.5, kph, np.nan), color='#FFD600', lw=lw + 0.3, label='straight')
            ax.plot(s, np.where(seg > 0.5, kph, np.nan), color='#42A5F5', lw=lw + 0.3, label='corner')
            for st in getattr(res, 'straights', None) or []:
                ax.hlines(st['v_mean_kph'], st['s_in_m'], st['s_out_m'], color='#FFD600', ls='--', lw=1.1, alpha=0.9)
            for c in getattr(res, 'corners', None) or []:
                ax.hlines(c['v_mean_kph'], c['s_in_m'], c['s_out_m'], color='#42A5F5', ls='--', lw=1.1, alpha=0.9)
            ax.plot([], [], color='#9a9aa2', ls='--', lw=1.1, label='mean speed of each segment')
            ax.set_ylim(0, float(np.nanmax(kph)) * 1.12)
            self._legend(ax, big)

    def _draw_into(self, fig, keys, with_track=True, big=False):
        """Lay out [track map] + the `keys` graphs stacked on ONE shared
        distance axis in `fig`.  Returns [(key, ax), …] ('track' first)."""
        fig.clf()
        fig.set_facecolor(_BG_FIG)
        out = []
        n = len(keys) + (1 if with_track else 0)
        if n == 0:
            return out
        if with_track:
            # the map keeps a usable share of the height however many
            # graphs are stacked under it
            tr = 1.6 if len(keys) <= 1 else max(1.2, 0.25 * len(keys))
            ratios = [tr] + [1.0] * len(keys)
        else:
            ratios = [1.0] * len(keys)
        gs = fig.add_gridspec(n, 1, height_ratios=ratios)
        row = 0
        if with_track:
            tax = fig.add_subplot(gs[0])
            self._draw_track(fig, tax, big)
            out.append(('track', tax))
            row = 1
        ax0 = None
        for gi, key in enumerate(keys):
            ax = fig.add_subplot(gs[row + gi], sharex=ax0)
            if ax0 is None:
                ax0 = ax
            ax._lap_compact = (not big) and len(keys) > 4
            self._draw_graph(ax, key, big)
            if gi < len(keys) - 1:
                # shared distance axis: only the bottom graph shows the tick
                # numbers; the label text stays set (the hover reads it)
                ax.tick_params(labelbottom=False)
                ax.xaxis.label.set_visible(False)
            out.append((key, ax))
        if ax0 is not None:
            ax0.set_xlim(float(self._result.s_m[0]), float(self._result.s_m[-1]))
        return out

    def _redraw(self, *_):
        """ONE big grid in a single canvas: the TRACK map always on top
        (equal-aspect, speed-coloured), then the checked graphs (or the Solo
        graph) stacked below on one shared distance axis.  Navigate per
        subplot with scroll (zoom), drag (pan), double-click (reset); click
        'expand ↗' on any graph for a big window with zoom tools."""
        if self._track is None:
            return
        self._home_lims = {}
        self._expand_tags = []
        drawn = self._draw_into(self._fig, self._shown_keys(),
                                with_track=True, big=False)
        for key, ax in drawn:
            tag = ax.text(1.0, 1.005, 'expand ↗', transform=ax.transAxes,
                          ha='right', va='bottom', fontsize=8,
                          color='#FFD600', zorder=20,
                          bbox=dict(boxstyle='round,pad=0.2', fc='#15151a',
                                    ec='#FFD600', lw=0.6))
            self._expand_tags.append((tag, key))
        self._canvas.draw()
        for _key, ax in drawn:
            self._home_lims[ax] = (ax.get_xlim(), ax.get_ylim())

    # ── pop-out ──────────────────────────────────────────────────────────
    def _expand_hit(self, event) -> str | None:
        for tag, key in getattr(self, '_expand_tags', []):
            try:
                if tag.contains(event)[0]:
                    return key
            except Exception:
                pass
        return None

    def open_popout(self, keys=None, with_track=None):
        """Open graphs large in their own window (zoom box, pan, reset, save
        from the toolbar; hover to read).  keys=None -> every graph on the
        page right now, plus the map."""
        if self._track is None:
            return None
        if keys is None:
            keys = self._shown_keys()
            with_track = True if with_track is None else with_track
        with_track = bool(with_track)
        if not keys and not with_track:
            return None
        win = _LapPlotWindow(self, list(keys), with_track)
        keep = []
        for w in getattr(self, '_popouts', []):
            try:
                if w.isVisible():
                    keep.append(w)
            except RuntimeError:          # closed (WA_DeleteOnClose)
                pass
        self._popouts = keep + [win]
        win.show()
        return win


def draw_ggmap(fig, res, track=None):
    """The lap (sideways g signed by turn direction, TYRE forward/braking g)
    over the car's G-G limit at three speeds.  Points coloured by speed
    (blue slow -> white -> yellow -> red fast, the page's speed colours)."""
    fig.clf()
    fig.set_facecolor(_BG_FIG)
    ax = fig.add_subplot(111)
    ax.set_facecolor('#0e0e10')
    gm = res.ggv
    v = np.asarray(res.v_ms, float)
    lat = np.asarray(res.lat_g, float)
    if track is not None and len(track.kappa_1pm) == len(lat):
        lat = np.sign(np.asarray(track.kappa_1pm, float)) * lat
    lon = np.asarray(res.lon_tyre_g if res.lon_tyre_g is not None else res.lon_g, float)
    kph = v * 3.6
    sc = ax.scatter(lat, lon, c=kph, cmap=_SPEED_CMAP, s=6, zorder=3)
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label('speed (km/h)', color='#e8e8ec')
    cb.ax.tick_params(colors='#e8e8ec')
    vs = [float(np.percentile(v, 10)), float(np.median(v)), float(np.max(v))]
    for vv, ls in zip(vs, (':', '--', '-')):
        bx, by = gm.border(vv)
        ax.plot(bx, by, color=_SPEED_CMAP(min(max((vv * 3.6 - kph.min()) / max(np.ptp(kph), 1e-6), 0), 1)),
                lw=1.6, ls=ls, label=f'limit at {vv * 3.6:.0f} km/h')
    ax.axhline(0, color='#55555c', lw=0.7); ax.axvline(0, color='#55555c', lw=0.7)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('sideways g (+ left-hand turn, − right-hand)', color='#e8e8ec')
    ax.set_ylabel('tyre forward g (+ speeding up, − braking)', color='#e8e8ec')
    ax.set_title("This lap on the car's G-G limits (full suspension solve; "
                 'drag taken out of the forward g)', color='#e8e8ec', fontsize=10)
    ax.tick_params(colors='#e8e8ec')
    for sp in ax.spines.values():
        sp.set_color('#3a3a40')
    leg = ax.legend(loc='upper left', bbox_to_anchor=(0.0, -0.10), ncol=3,
                    borderaxespad=0.0, fontsize=8, facecolor='#15151a',
                    edgecolor='#3a3a40')
    for t in leg.get_texts():
        t.set_color('#e8e8ec')
    return ax


class _LapPlotWindow(QDialog):
    """A lap graph (or the whole stack) large, in its own resizable window,
    with matplotlib's zoom/pan/reset/save toolbar and the hover readout.
    Distance graphs share one x axis, so zooming one zooms them all."""

    _PX_PER_GRAPH = 300           # beyond 4 graphs the window scrolls

    def __init__(self, page, keys, with_track):
        super().__init__(page)
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        names = dict((k, l) for k, l, _d in page._GRAPHS)
        names['track'] = 'Track map'
        title = ', '.join(names.get(k, k) for k in
                          ((['track'] if with_track else []) + keys))
        self.setWindowTitle(f'Lap graphs — {title}')
        self.resize(1500, 950)
        self.setStyleSheet('QDialog { background:#000000; }')
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        self.fig = Figure(facecolor=_BG_FIG, layout='constrained')
        self.canvas = ReadableCanvas(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        lay.addWidget(self.toolbar)
        hint = QLabel('Hover over a line to read its value.  Toolbar: '
                      'magnifier = drag a box to zoom in · cross arrows = '
                      'drag to move · house = back to the full lap · disk = '
                      'save.  Distance graphs zoom together.')
        hint.setStyleSheet('color:#9a9aa2; font-size:11px;')
        hint.setWordWrap(True)
        lay.addWidget(hint)
        n = len(keys) + (1 if with_track else 0)
        if n > 4:
            sc = QScrollArea()
            sc.setWidgetResizable(True)
            sc.setStyleSheet('QScrollArea { border:0; background:#000; }')
            self.canvas.setMinimumHeight(self._PX_PER_GRAPH * n)
            sc.setWidget(self.canvas)
            lay.addWidget(sc, 1)
        else:
            lay.addWidget(self.canvas, 1)
            self.canvas.mpl_connect('scroll_event', self._on_scroll)
        page._draw_into(self.fig, keys, with_track=with_track, big=True)
        self.canvas.draw_idle()       # laid out once the window has a size

    def _on_scroll(self, event):
        ax = event.inaxes
        if ax is None or event.xdata is None:
            return
        f = (1.0 / 1.25) if event.button == 'up' else 1.25
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        ax.set_xlim(event.xdata - (event.xdata - x0) * f,
                    event.xdata + (x1 - event.xdata) * f)
        ax.set_ylim(event.ydata - (event.ydata - y0) * f,
                    event.ydata + (y1 - event.ydata) * f)
        self.canvas.draw_idle()
