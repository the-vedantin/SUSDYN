"""gui/corner_speed_page.py — the CORNER SPEED page (Ctrl+9).

WHY IT EXISTS.  Two analyses lived as scripts + PNGs (the speed-vs-radius
study and the per-corner grip-budget tables).  The rule is that ANY method,
plot or table must be usable as a FEATURE in the app, driven by the loaded
model — so here they are, as two tabs on one page.

WHAT IT COMPUTES.  Nothing.  Every number comes out of vahan:

    tab 1  TIGHTEST CORNER vs SPEED
           vahan.corner_speed.corner_speed_rows  ->  vahan.ymd.trim_sweep_ackermann
           (the ONE yaw-moment engine) at the car's as-built Ackermann, with and
           without the Dynamics panel's aero package; the steering-lock radius
           from the kinematic solvers with the rack at its physical stop.
    tab 2  PER-CORNER GRIP BUDGET  ("all four tyres inside their budget")
           vahan.corner_speed.grip_budget_study  ->  vahan.dynamics.SteadyStateSolver
           (its own per-corner utilization), bisected to the first g at which any
           corner exceeds 1.0; the axle-aggregate limit alongside for reference.

This page ARRANGES, THREADS and PLOTS.  Heavy calls run in a QThread with a
progress line; the large radii are solved first so the table fills from the
easy end.  run_corner_speed(blocking=True) / run_grip_budget(blocking=True)
run inline for tests and return the compute function's own result.

PALETTE.  Graph lines: yellow = no aero, red = aero, white = the steering-lock
marker (in-app palette; the user is colourblind — never red/green).  The
per-corner curves on tab 2 use the app's corner convention FL yellow / FR red /
RL white / RR blue, solid = no aero, dashed = aero.  Text follows
docs/DESIGN.md (no yellow text; accent red for emphasis).
"""
from __future__ import annotations

import math
import traceback

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QScrollArea, QCheckBox, QLineEdit, QComboBox, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QApplication, QGroupBox,
    QSplitter, QDoubleSpinBox,
)

from vahan import corner_speed as CS
from vahan.kinematics import KinematicMetrics

# ── docs/DESIGN.md dark tokens (text / UI) ──────────────────────────────────
_INK, _MUTED, _ACCENT = '#ECECEE', '#9A9AA2', '#E23B48'
_PANEL, _LINE, _PAPER = '#141416', '#2A2A2E', '#0B0B0D'
# ── graph LINE colours only ─────────────────────────────────────────────────
_YEL, _RED, _WHT, _BLU = '#FFD600', '#E53935', '#FFFFFF', '#42A5F5'
_CC = {'FL': _YEL, 'FR': _RED, 'RL': _WHT, 'RR': _BLU}
_MONO = ("font-family:'Consolas','SF Mono',monospace; font-size:11px; "
         f"color:{_INK}; background:#0e0e10; padding:6px; "
         f"border:1px solid {_LINE}; border-radius:4px;")
_G_EARTH = 9.80665     # the constant inside MainWindow._custom_aero_Fz_per_g


# ═══════════════════════════════════════════════════════════════════════════
#  workers — every heavy call off the GUI thread
# ═══════════════════════════════════════════════════════════════════════════
class _SpeedWorker(QThread):
    row = pyqtSignal(int, int, object)      # (i_done, n_total, row dict)
    done = pyqtSignal(object)               # the full row list
    failed = pyqtSignal(str)

    def __init__(self, tire, solver, cfg):
        super().__init__()
        self._tire, self._solver, self._cfg = tire, solver, cfg
        self.cancel = False

    def run(self):
        try:
            c = self._cfg
            rows = CS.corner_speed_rows(
                self._tire, self._solver, c['radii'],
                ackermann_pct=c['ackermann_pct'], grip_multiplier=c['mu_scale'],
                aero_per_g_by_radius=c['aero_by_radius'],
                progress=lambda i, n, r: self.row.emit(i, n, r),
                cancelled=lambda: self.cancel)
            self.done.emit(rows)
        except Exception as e:                  # noqa: BLE001 — surfaced on the page
            traceback.print_exc()
            self.failed.emit(f'{type(e).__name__}: {e}')


class _GripWorker(QThread):
    note = pyqtSignal(str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, solver, cfg):
        super().__init__()
        self._solver, self._cfg = solver, cfg
        self.cancel = False

    def run(self):
        try:
            c = self._cfg
            study = CS.grip_budget_study(
                self._solver, lateral_g=c['lateral_g'], aero_cases=c['aero_cases'],
                sweep_g=c.get('sweep_g'), iters=c['iters'],
                progress=self.note.emit, cancelled=lambda: self.cancel,
                grip_scales=c.get('grip_scales'))
            self.done.emit(study)
        except Exception as e:                  # noqa: BLE001
            traceback.print_exc()
            self.failed.emit(f'{type(e).__name__}: {e}')


# ═══════════════════════════════════════════════════════════════════════════
#  the page
# ═══════════════════════════════════════════════════════════════════════════
class CornerSpeedPage(QWidget):
    """Corner speed + per-corner grip budget.  `main` is the MainWindow, which
    owns the ONE solved car, its steering rack and the ONE tyre model."""

    SPEED_COLS = ('radius m', 'aero', 'lateral g', 'speed km/h', 'stability N_beta Nm/deg',
                  'stable?', 'downforce at limit N', 'body slip deg', 'front steer deg',
                  'converged', 'note')
    GRIP_COLS = ('case', 'point', 'lateral g', 'corner', 'Fz N', 'inclination deg',
                 'budget N', 'demand N', 'utilization', 'below tyre-data floor')

    def __init__(self, main):
        super().__init__()
        self._main = main
        self._worker = None
        self._grip_worker = None
        self.last_rows: list | None = None      # tab 1 result (compute function's rows)
        self.last_speed_cfg: dict | None = None
        self.last_study: dict | None = None     # tab 2 result (compute function's study)
        self.last_grip_cfg: dict | None = None
        self._rows_live: list = []
        self._build()
        self._seed_grip_defaults()
        self._refresh_car_note()

    # ── UI ──────────────────────────────────────────────────────────────
    def _build(self):
        from gui.ackermann_page import _Slot
        from gui.panels import _spin, BTN_PRIMARY, BTN_SECONDARY
        self.setStyleSheet(
            f'QWidget{{background:{_PAPER};color:{_INK};font-size:13px;}}'
            f'QGroupBox{{border:1px solid {_LINE};border-radius:4px;margin-top:14px;padding-top:6px;}}'
            f'QGroupBox::title{{subcontrol-origin:margin;left:8px;color:{_ACCENT};'
            'font-weight:650;letter-spacing:1px;font-size:11px;}'
            'QDoubleSpinBox,QSpinBox,QComboBox,QLineEdit{background:#1e1e24;border:1px solid #444;'
            'border-radius:3px;padding:2px 4px;}'
            f'QTableWidget{{background:{_PANEL};gridline-color:{_LINE};font-size:11px;}}'
            f'QHeaderView::section{{background:#1e1e24;color:{_MUTED};padding:4px;border:0;font-size:10px;}}')
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8); root.setSpacing(8)

        side = QVBoxLayout(); side.setSpacing(6)
        title = QLabel('CORNER SPEED')
        title.setStyleSheet(f'color:{_ACCENT}; font-size:15px; font-weight:bold; letter-spacing:1px;')
        side.addWidget(title)
        sub = QLabel('How fast this car can take each corner radius, and which tyre '
                     'runs out of grip first.  Every number is the loaded model.')
        sub.setStyleSheet(f'color:{_MUTED}; font-size:11px;'); sub.setWordWrap(True)
        side.addWidget(sub)

        # THE CAR — what the two tabs read from the model
        self._car_note = QLabel('')
        self._car_note.setStyleSheet(_MONO); self._car_note.setWordWrap(True)
        self._car_note.setToolTip(
            'Config, tyre and pressure, belt-to-track grip scale and the aero '
            'package are set on the Dynamics panel (Suspension page).  Ackermann '
            'is probed from the steering linkage at full lock; the lock radius is '
            'solved from the front corners with the rack at its physical stop.')
        side.addWidget(self._car_note)

        # ── tab-1 inputs ───────────────────────────────────────────────
        g1 = QGroupBox('TIGHTEST CORNER vs SPEED'); l1 = QGridLayout(g1); l1.setSpacing(4)
        r = 0
        lb = QLabel('Corner radii (m):'); lb.setStyleSheet('font-size:11px;')
        l1.addWidget(lb, r, 0, 1, 2); r += 1
        self._radii_txt = QLineEdit(', '.join(f'{x:g}' for x in CS.DEFAULT_RADII_M))
        self._radii_txt.setToolTip(
            'Comma-separated radii.  Solved largest first (they trim fast; the '
            'hairpin last).  FSAE: 50 m sweeper down to a 4.5 m hairpin.')
        l1.addWidget(self._radii_txt, r, 0, 1, 2); r += 1
        self._add_lock = QCheckBox('add the steering-lock radius')
        self._add_lock.setChecked(True)
        self._add_lock.setToolTip('Append the kinematic minimum radius (rack at its stop, '
                                  'mean of both front wheel angles) as the last row.')
        l1.addWidget(self._add_lock, r, 0, 1, 2); r += 1
        self._aero_chk = QCheckBox('include aero (Dynamics panel package)')
        self._aero_chk.setChecked(True)
        self._aero_chk.setToolTip(
            'A second row per radius with the panel\'s aero package.  On a fixed '
            'radius the downforce scales with lateral g (F = ½ρ·ClA·g·R·Ay), so it '
            'enters the trim engine as N-per-g at that radius — the same path the '
            'Ackermann page and the loads table use.')
        l1.addWidget(self._aero_chk, r, 0, 1, 2); r += 1
        self._aero_note = QLabel(''); self._aero_note.setStyleSheet(f'color:{_MUTED}; font-size:10px;')
        self._aero_note.setWordWrap(True)
        l1.addWidget(self._aero_note, r, 0, 1, 2); r += 1
        # grip multiplier = a MIRROR of THE project grip scale (every page's grip box shows/sets it)
        _lbg = QLabel('Grip multiplier:'); _lbg.setStyleSheet('font-size:11px;')
        self._grip = QDoubleSpinBox(); self._grip.setRange(0.10, 1.50); self._grip.setDecimals(2); self._grip.setSingleStep(0.01)
        self._grip.setToolTip('Tyre grip multiplier (belt-to-track).  Same value as the Dynamics panel / Lap Time / every '
                              'page grip box — editing it here changes it everywhere.')
        l1.addWidget(_lbg, r, 0); l1.addWidget(self._grip, r, 1); r += 1
        self._btn_speed = QPushButton('►  ANALYSE CORNER SPEED')
        self._btn_speed.setStyleSheet(BTN_PRIMARY)
        self._btn_speed.clicked.connect(lambda: self.run_corner_speed(blocking=False))
        l1.addWidget(self._btn_speed, r, 0, 1, 2); r += 1
        self._btn_speed_copy = QPushButton('Copy table')
        self._btn_speed_copy.setStyleSheet(BTN_SECONDARY)
        self._btn_speed_copy.setToolTip('Copy the rows as tab-separated text (Excel / Sheets paste).')
        self._btn_speed_copy.clicked.connect(self._copy_speed_table)
        self._btn_speed_cancel = QPushButton('Cancel')
        self._btn_speed_cancel.setStyleSheet(BTN_SECONDARY); self._btn_speed_cancel.setEnabled(False)
        self._btn_speed_cancel.clicked.connect(self._cancel_speed)
        l1.addWidget(self._btn_speed_copy, r, 0); l1.addWidget(self._btn_speed_cancel, r, 1); r += 1
        self._speed_progress = QLabel('press ANALYSE')
        self._speed_progress.setStyleSheet(f'color:{_MUTED}; font-size:11px;'); self._speed_progress.setWordWrap(True)
        l1.addWidget(self._speed_progress, r, 0, 1, 2); r += 1
        side.addWidget(g1)

        # ── tab-2 inputs ───────────────────────────────────────────────
        g2 = QGroupBox('PER-CORNER GRIP BUDGET'); l2 = QGridLayout(g2); l2.setSpacing(4)
        r = 0

        def row2(label, w, tip=''):
            nonlocal r
            lb = QLabel(label); lb.setStyleSheet('font-size:11px;')
            if tip:
                lb.setToolTip(tip); w.setToolTip(tip)
            l2.addWidget(lb, r, 0); l2.addWidget(w, r, 1); r += 1
            return w

        self._grip_g = row2('Table at:', _spin(0.1, 4.0, 1.5, ' g', dec=2, step=0.1),
                            'Lateral g for the four-corner table.  The limits below '
                            'are bisected independently of this value.')
        self._grip_mode = QComboBox()
        self._grip_mode.addItem('fixed downforce at a speed', 'speed')
        self._grip_mode.addItem('constant radius (V² scaling with g)', 'radius')
        row2('Aero case:', self._grip_mode,
             'How the panel\'s aero package enters the aero rows.  "fixed at a '
             'speed": the downforce the package makes at that speed, held constant '
             'while g rises (the binder tables).  "constant radius": downforce '
             'grows with g on that radius (the Dynamics sweep convention).')
        self._grip_speed = row2('at speed:', _spin(1.0, 400.0, 60.0, ' km/h', dec=1, step=5.0),
                                'Speed for the fixed-downforce case.  Defaults to the '
                                'panel\'s aero reference speed, so the aero row is the '
                                'panel\'s own downforce number.')
        self._grip_radius = row2('at radius:', _spin(1.0, 500.0, 10.0, ' m', dec=1, step=1.0),
                                 'Radius for the constant-radius case (defaults to the '
                                 'Dynamics panel turn radius).')
        self._grip_mode.currentIndexChanged.connect(self._sync_grip_mode)
        self._btn_grip = QPushButton('►  RUN GRIP BUDGET')
        self._btn_grip.setStyleSheet(BTN_PRIMARY)
        self._btn_grip.clicked.connect(lambda: self.run_grip_budget(blocking=False))
        l2.addWidget(self._btn_grip, r, 0, 1, 2); r += 1
        self._btn_grip_copy = QPushButton('Copy table')
        self._btn_grip_copy.setStyleSheet(BTN_SECONDARY)
        self._btn_grip_copy.setToolTip('Copy every row (chosen g + both limits) as tab-separated text.')
        self._btn_grip_copy.clicked.connect(self._copy_grip_table)
        self._btn_grip_cancel = QPushButton('Cancel')
        self._btn_grip_cancel.setStyleSheet(BTN_SECONDARY); self._btn_grip_cancel.setEnabled(False)
        self._btn_grip_cancel.clicked.connect(self._cancel_grip)
        l2.addWidget(self._btn_grip_copy, r, 0); l2.addWidget(self._btn_grip_cancel, r, 1); r += 1
        self._grip_progress = QLabel('press RUN')
        self._grip_progress.setStyleSheet(f'color:{_MUTED}; font-size:11px;'); self._grip_progress.setWordWrap(True)
        l2.addWidget(self._grip_progress, r, 0, 1, 2); r += 1
        side.addWidget(g2)

        self._status = QLabel('')
        self._status.setStyleSheet(f'color:{_MUTED}; font-size:11px;'); self._status.setWordWrap(True)
        side.addWidget(self._status)
        side.addStretch(1)

        inner = QWidget(); inner.setLayout(side)
        scroll = QScrollArea(); scroll.setWidget(inner); scroll.setWidgetResizable(True)
        scroll.setFixedWidth(330); scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        root.addWidget(scroll)

        # ── right: two tabs ────────────────────────────────────────────
        self._tabs = QTabWidget()
        # tab 1
        t1 = QSplitter(Qt.Orientation.Vertical)
        self.speed_slot = _Slot('TIGHTEST CORNER vs SPEED — the highest trimmed (yaw moment = 0) '
                                'lateral g the car holds on each radius, and the speed that goes '
                                'with it.  Yellow = no aero, red = aero, white dashed = steering lock.')
        t1.addWidget(self.speed_slot)
        self.speed_table = self._make_table(self.SPEED_COLS)
        t1.addWidget(self.speed_table)
        t1.setStretchFactor(0, 3); t1.setStretchFactor(1, 2)
        self._tabs.addTab(t1, 'Tightest corner vs speed')
        # tab 2
        t2w = QWidget(); t2 = QVBoxLayout(t2w); t2.setContentsMargins(0, 0, 0, 0); t2.setSpacing(6)
        self.grip_readout = QLabel('press RUN GRIP BUDGET')
        self.grip_readout.setStyleSheet(_MONO); self.grip_readout.setWordWrap(True)
        t2.addWidget(self.grip_readout)
        t2s = QSplitter(Qt.Orientation.Vertical)
        self.grip_slot = _Slot('PER-CORNER UTILIZATION vs LATERAL g — demand / (μ_peak(Fz, IA) × grip '
                               'scale × Fz) for each tyre; 1.0 = the tyre is out of budget.  '
                               'FL yellow · FR red · RL white · RR blue; solid = no aero, dashed = aero.')
        t2s.addWidget(self.grip_slot)
        self.grip_table = self._make_table(self.GRIP_COLS)
        t2s.addWidget(self.grip_table)
        t2s.setStretchFactor(0, 3); t2s.setStretchFactor(1, 2)
        t2.addWidget(t2s, 1)
        self._tabs.addTab(t2w, 'Per-corner grip budget')
        root.addWidget(self._tabs, 1)

        self.speed_slot.busy('press ANALYSE CORNER SPEED')
        self.grip_slot.busy('press RUN GRIP BUDGET')
        self._sync_grip_mode()

    @staticmethod
    def _make_table(cols):
        t = QTableWidget(0, len(cols))
        t.setHorizontalHeaderLabels(list(cols))
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        t.horizontalHeader().setStretchLastSection(True)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.setAlternatingRowColors(False)
        return t

    def _sync_grip_mode(self, *_):
        mode = self._grip_mode.currentData()
        self._grip_speed.setEnabled(mode == 'speed')
        self._grip_radius.setEnabled(mode == 'radius')

    # ── what the page reads from the model ──────────────────────────────
    def _lock_solve(self) -> dict:
        """Front toe angles with the rack at its physical stop, from FRESH corner
        solvers (MainWindow._build_corner_solvers is pure w.r.t. the GUI — the
        live solvers are not touched), and the mean-steer bicycle radius."""
        mw = self._main
        out = {'handwheel_deg': float('nan'), 'toe_FL_deg': float('nan'),
               'toe_FR_deg': float('nan'), 'lock_radius_m': float('nan'),
               'inner_wheel_radius_m': float('nan')}
        try:
            hand = CS.full_lock_handwheel_deg(mw._steer)
            if not np.isfinite(hand):
                return out
            solv = type(mw)._build_corner_solvers(mw._all_corner_hp(), mw._steer,
                                                  mw._topology, hand)
            tl = float(KinematicMetrics(solv['FL'].solve(0.0), 'left').toe)
            tr = float(KinematicMetrics(solv['FR'].solve(0.0), 'right').toe)
            # the same car-dict key _build_dynamics_solver feeds VehicleParams
            wb = float(mw._car['wheelbase_mm']) / 1000.0
            out.update(handwheel_deg=hand, toe_FL_deg=tl, toe_FR_deg=tr,
                       lock_radius_m=CS.lock_radius_m(tl, tr, wb))
            inner = max(abs(tl), abs(tr))
            if inner > 0.1:
                out['inner_wheel_radius_m'] = wb / math.tan(math.radians(inner))
        except Exception:                        # noqa: BLE001
            traceback.print_exc()
        return out

    def _aero_per_g(self, radius_m: float):
        """The car's aero package at this radius as N per lateral g, through the
        ONE app path (MainWindow._get_aero_Fz_per_g) with the Apply-Aero gate
        lifted for the read — same as the Ackermann page."""
        mw = self._main
        was = getattr(mw, '_aero_active', False)
        try:
            mw._aero_active = True
            aero = mw._get_aero_Fz_per_g(radius_m=float(radius_m))
        except Exception:                        # noqa: BLE001
            aero = None
        finally:
            mw._aero_active = was
        return aero or None

    def _aero_fixed_at_speed(self, speed_kph: float):
        """The panel's aero package (F_ref at V_ref, CoP) as the downforce AT
        one speed, through the app's ONE conversion (MainWindow.
        _custom_aero_Fz_per_g): at 1 g on the radius R = V²/g the speed is
        exactly V, so the per-g dict at that radius IS the load at V.  Read
        from the F_ref/V_ref package directly — the 'solved deficit' source is
        defined per g, not per speed, and belongs to the constant-radius case."""
        v = float(speed_kph) / 3.6
        if v <= 0.0:
            return None
        try:
            aero = self._main._custom_aero_Fz_per_g(radius_m=v * v / _G_EARTH)
        except Exception:                        # noqa: BLE001
            aero = None
        return aero or None

    def _aero_description(self) -> str:
        mw = self._main
        try:
            panel = mw._dynamics_panel
            src = panel.get_aero_source() if hasattr(panel, 'get_aero_source') else 'solved'
            if src == 'solved' and getattr(mw, '_last_aero_result', None) is not None:
                r = mw._last_aero_result
                return (f'aero source: solved deficit from the Aero panel — '
                        f'{r.front_axle_need_N + r.rear_axle_need_N:.0f} N at {r.lateral_g:.2f} g')
            c = panel.get_custom_aero_params()
            v = float(c['V_ref_kph']) / 3.6
            cla = (2.0 * float(c['F_ref_N']) / (float(c['air_density']) * v * v)
                   if v > 0 and float(c['air_density']) > 0 else float('nan'))
            return (f'aero package (Dynamics panel): {c["F_ref_N"]:.0f} N at {c["V_ref_kph"]:.1f} km/h '
                    f'= ClA-equivalent {cla:.3f} m², CoP {c["cop_rear_pct"]:.0f} % rear')
        except Exception:                        # noqa: BLE001
            return 'aero package: not readable'

    def _aero_short(self) -> str:
        """Legend / note form of the aero package.  Short ON PURPOSE: a legend
        label wider than the axes makes constrained layout collapse the axes
        to zero width (seen 2026-09-14 with the long description above)."""
        mw = self._main
        try:
            panel = mw._dynamics_panel
            src = panel.get_aero_source() if hasattr(panel, 'get_aero_source') else 'solved'
            if src == 'solved' and getattr(mw, '_last_aero_result', None) is not None:
                r = mw._last_aero_result
                return (f'solved deficit {r.front_axle_need_N + r.rear_axle_need_N:.0f} N '
                        f'at {r.lateral_g:.2f} g')
            c = panel.get_custom_aero_params()
            return (f'{c["F_ref_N"]:.0f} N at {c["V_ref_kph"]:g} km/h, '
                    f'CoP {c["cop_rear_pct"]:.0f} % rear')
        except Exception:                        # noqa: BLE001
            return 'panel package'

    def _refresh_car_note(self):
        mw = self._main
        lines = []
        try:
            tm = getattr(mw, '_tire_model', None)
            if tm is None:
                lines.append('tyre     NONE LOADED — load a TTC file on the Dynamics panel')
            else:
                psi = float(getattr(tm, 'pressure_psi', 0.0) or 0.0)
                lines.append(f'tyre     {getattr(tm, "tire_id", "?")}'
                             + (f'  {psi:.0f} psi' if psi > 0 else '  pressure BLENDED'))
            if hasattr(mw, 'grip_scale'):
                lines.append(f'grip     belt-to-track scale {mw.grip_scale():.2f} (project grip scale; '
                             f'limits also at {", ".join(f"{x:.2f}" for x in mw.grip_scale_list())})')
            lk = self._lock_solve()
            if np.isfinite(lk['lock_radius_m']):
                lines.append(f'lock     handwheel {lk["handwheel_deg"]:.0f} deg -> front wheels '
                             f'{abs(lk["toe_FL_deg"]):.1f} / {abs(lk["toe_FR_deg"]):.1f} deg')
                lines.append(f'         min radius {lk["lock_radius_m"]:.2f} m (mean steer; inner wheel '
                             f'alone {lk["inner_wheel_radius_m"]:.2f} m)')
            else:
                lines.append('lock     steering-lock radius not solvable (check the rack)')
            lines.append(self._aero_description())
        except Exception as e:                   # noqa: BLE001
            lines.append(f'model readout failed: {e}')
        self._car_note.setText('\n'.join(lines))
        self._aero_note.setText('aero package: ' + self._aero_short())

    def _seed_grip_defaults(self):
        """ONCE, at construction: the tab-2 aero case starts at the panel's own
        aero reference speed and turn radius.  Never re-applied on RUN — a
        value the user typed must survive the run."""
        mw = self._main
        try:
            c = mw._dynamics_panel.get_custom_aero_params()
            self._grip_speed.setValue(float(c['V_ref_kph']))
            self._grip_radius.setValue(float(mw._dynamics_panel._turn_radius.value()))
        except Exception:                        # noqa: BLE001
            pass

    def _solver(self):
        tire = getattr(self._main, '_tire_model', None)
        solver = self._main._build_dynamics_solver()
        if tire is None:
            tire = solver._tire                  # LinearTireModel fallback, announced
        return tire, solver

    # ═══════════════════════════════════════════════════════════════════
    #  tab 1 — tightest corner vs speed
    # ═══════════════════════════════════════════════════════════════════
    def speed_config(self) -> dict:
        """Everything the compute function needs, read on the GUI thread."""
        from gui.ackermann_page import _parse_list
        radii = sorted({round(float(x), 4) for x in _parse_list(self._radii_txt.text(), CS.DEFAULT_RADII_M)
                        if x > 0.0}, reverse=True)
        lk = self._lock_solve()
        if self._add_lock.isChecked() and np.isfinite(lk['lock_radius_m']):
            rl = round(float(lk['lock_radius_m']), 2)
            if all(abs(rl - x) > 1e-6 for x in radii):
                radii.append(rl)
            radii = sorted(set(radii), reverse=True)
        try:
            ack = float(self._main._probe_static_ackermann())
        except Exception:                        # noqa: BLE001
            ack = float('nan')
        ack_ok = bool(np.isfinite(ack))
        if not ack_ok:
            ack = 0.0
        aero_by_radius = None
        if self._aero_chk.isChecked():
            aero_by_radius = {R: self._aero_per_g(R) for R in radii}
            if not any(aero_by_radius.values()):
                aero_by_radius = None
        return {'radii': radii, 'ackermann_pct': ack, 'ackermann_probed': ack_ok,
                'aero_by_radius': aero_by_radius, 'lock': lk,
                # legend label: the SHORT form (a long one collapses the axes)
                'aero_text': self._aero_short()}

    def run_corner_speed(self, blocking: bool = False):
        """Tab 1.  blocking=True runs inline (tests) and returns the rows the
        compute function produced; otherwise a worker thread fills the page."""
        if self._worker is not None and self._worker.isRunning():
            self._speed_progress.setText('already analysing — wait or cancel')
            return None
        self._refresh_car_note()
        try:
            tire, solver = self._solver()
        except Exception as e:                   # noqa: BLE001
            self._speed_progress.setText(f'could not build the car: {e}')
            return None
        cfg = self.speed_config()
        cfg['mu_scale'] = float(solver._mu_scale)
        cfg['tire_id'] = str(getattr(tire, 'tire_id', 'linear tyre model'))
        self.last_speed_cfg = cfg
        self._rows_live = []
        self.speed_table.setRowCount(0)
        self.speed_slot.busy('solving the large radii first…')
        self.speed_slot.summary('—')
        n_rows = len(cfg['radii']) * (2 if cfg['aero_by_radius'] else 1)
        self._speed_progress.setText(f'0 / {n_rows} rows')
        if blocking:
            rows = CS.corner_speed_rows(
                tire, solver, cfg['radii'], ackermann_pct=cfg['ackermann_pct'],
                grip_multiplier=cfg['mu_scale'], aero_per_g_by_radius=cfg['aero_by_radius'],
                progress=self._on_speed_row)
            self._finish_speed(rows)
            return rows
        self._btn_speed.setEnabled(False); self._btn_speed_cancel.setEnabled(True)
        self._worker = _SpeedWorker(tire, solver, cfg)
        self._worker.row.connect(self._on_speed_row)
        self._worker.done.connect(self._finish_speed)
        self._worker.failed.connect(self._fail_speed)
        self._worker.start()
        return None

    def _cancel_speed(self):
        if self._worker is not None:
            self._worker.cancel = True
            self._speed_progress.setText('cancelling after the current radius…')

    def _fail_speed(self, msg):
        self._btn_speed.setEnabled(True); self._btn_speed_cancel.setEnabled(False)
        self._speed_progress.setText(f'analysis failed: {msg}')

    def _on_speed_row(self, i, n, row):
        self._rows_live.append(row)
        self._speed_progress.setText(
            f'{i} / {n} rows — R {row["radius_m"]:.2f} m {"aero" if row["aero"] else "no aero"}: '
            + (f'{row["ay_g"]:.3f} g, {row["speed_kph"]:.1f} km/h' if np.isfinite(row['ay_g'])
               else f'no trim ({row.get("error") or "not converged"})'))
        self._draw_speed(self._rows_live)
        self._fill_speed_table(self._rows_live)
        QApplication.processEvents()

    def _finish_speed(self, rows):
        self._btn_speed.setEnabled(True); self._btn_speed_cancel.setEnabled(False)
        self.last_rows = list(rows)
        self._draw_speed(self.last_rows)
        self._fill_speed_table(self.last_rows)
        cfg = self.last_speed_cfg or {}
        n_ok = sum(1 for r in rows if np.isfinite(r['ay_g']))
        ack_txt = (f'{cfg.get("ackermann_pct", 0.0):+.1f} % Ackermann (probed at full lock)'
                   if cfg.get('ackermann_probed', True) else
                   'Ackermann probe FAILED — assumed 0 %')
        self._speed_progress.setText(
            f'done: {n_ok} of {len(rows)} rows trimmed; grip scale {cfg.get("mu_scale", float("nan")):.2f}, '
            f'{ack_txt}; tyre {cfg.get("tire_id", "?")}')

    def _draw_speed(self, rows):
        s = self.speed_slot
        ax1, ax2 = s.axes(2)
        cfg = self.last_speed_cfg or {}
        lock = float((cfg.get('lock') or {}).get('lock_radius_m', float('nan')))
        for aero, col, lab in ((False, _YEL, 'no aero'),
                               (True, _RED, 'aero: ' + (cfg.get('aero_text') or 'panel package'))):
            # (label kept short on purpose — see _aero_short)
            R, V, A = CS.corner_speed_series(rows, aero)
            if len(R) == 0:
                continue
            ax1.plot(R, V, 'o-', color=col, lw=2, ms=5, label=lab)
            ax2.plot(R, A, 'o-', color=col, lw=2, ms=5, label=lab)
        for ax, yl, ttl in ((ax1, 'max speed the car can hold the radius at (km/h)',
                             'speed vs corner radius'),
                            (ax2, 'trimmed lateral g at that speed',
                             'lateral g vs corner radius')):
            ax.set_xlabel('corner radius (m)', fontsize=8.5)
            ax.set_ylabel(yl, fontsize=8.5)
            ax.set_title(ttl, fontsize=9)
            if np.isfinite(lock):
                ax.axvline(lock, color=_WHT, ls='--', lw=1, label='_lock')
                ax.text(lock, 0.98, f' steering lock {lock:.2f} m', color=_WHT, fontsize=7.5,
                        va='top', ha='left', transform=ax.get_xaxis_transform())
            if ax.get_lines():
                s.legend(ax, loc='lower right')
        s.finish()
        # plain-English line: the hairpin and the fastest corner
        fin = [r for r in rows if np.isfinite(r['ay_g']) and not r['aero']]
        fin_a = [r for r in rows if np.isfinite(r['ay_g']) and r['aero']]
        if fin:
            tight = min(fin, key=lambda r: r['radius_m']); wide = max(fin, key=lambda r: r['radius_m'])
            txt = (f'WHAT IT MEANS: without aero the car holds a {tight["radius_m"]:.2f} m corner at '
                   f'{tight["speed_kph"]:.1f} km/h ({tight["ay_g"]:.2f} g) and a {wide["radius_m"]:.0f} m '
                   f'corner at {wide["speed_kph"]:.1f} km/h ({wide["ay_g"]:.2f} g).')
            if fin_a:
                wa = max(fin_a, key=lambda r: r['radius_m'])
                txt += (f'  With aero the {wa["radius_m"]:.0f} m corner goes to {wa["speed_kph"]:.1f} km/h '
                        f'({wa["ay_g"]:.2f} g, {wa["downforce_N"]:.0f} N of downforce at that speed).')
            unstable = [r for r in rows if r['stable'] is False]
            if unstable:
                txt += ('  Rows marked stable = NO have dN/dbeta > 0 at the trim: the car does not '
                        'self-correct there (' + ', '.join(f'{r["radius_m"]:.0f} m {"aero" if r["aero"] else "no aero"}'
                                                            for r in unstable) + ').')
            s.summary(txt)
        else:
            s.summary('no trimmed point yet')

    def _fill_speed_table(self, rows):
        t = self.speed_table
        cfg = self.last_speed_cfg or {}
        lock = float((cfg.get('lock') or {}).get('lock_radius_m', float('nan')))
        t.setRowCount(len(rows))
        for i, r in enumerate(rows):
            note = r.get('error') or ''
            if np.isfinite(lock) and abs(r['radius_m'] - round(lock, 2)) < 1e-6:
                note = 'steering lock' + (' — ' + note if note else '')
            vals = (f'{r["radius_m"]:.2f}', 'aero' if r['aero'] else 'no aero', f'{r["ay_g"]:.3f}',
                    f'{r["speed_kph"]:.1f}', f'{r["N_beta_Nm_per_deg"]:+.1f}',
                    {True: 'yes', False: 'NO', None: '?'}[r['stable']],
                    f'{r["downforce_N"]:.0f}', f'{r["beta_deg"]:+.2f}', f'{r["delta_deg"]:.2f}',
                    'yes' if r['converged'] else 'no', note)
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                                    if j not in (1, 5, 9, 10) else Qt.AlignmentFlag.AlignCenter)
                if j == 5 and r['stable'] is False:
                    it.setForeground(Qt.GlobalColor.red)
                t.setItem(i, j, it)

    def _copy_speed_table(self):
        rows = self.last_rows or self._rows_live
        if not rows:
            self._status.setText('nothing to copy — run the analysis first')
            return
        cfg = self.last_speed_cfg or {}
        lock = float((cfg.get('lock') or {}).get('lock_radius_m', float('nan')))
        txt = CS.corner_speed_table_text(rows, round(lock, 2) if np.isfinite(lock) else float('nan'))
        QApplication.clipboard().setText(txt)
        self._status.setText(f'{len(rows)} rows copied (tab-separated)')

    # ═══════════════════════════════════════════════════════════════════
    #  tab 2 — per-corner grip budget
    # ═══════════════════════════════════════════════════════════════════
    def grip_config(self) -> dict:
        mode = self._grip_mode.currentData()
        cases = {'no aero': None}
        if mode == 'speed':
            v = float(self._grip_speed.value())
            af = self._aero_fixed_at_speed(v)
            if af:
                cases[f'aero {sum(af.values()):.0f} N at {v:.1f} km/h'] = {'aero_Fz': af}
        else:
            R = float(self._grip_radius.value())
            ag = self._aero_per_g(R)
            if ag:
                cases[f'aero {sum(ag.values()):.0f} N per g on {R:.1f} m'] = {'aero_Fz_per_g': ag}
        return {'lateral_g': float(self._grip_g.value()), 'aero_cases': cases,
                'mode': mode, 'iters': 24, 'sweep_g': None,
                'aero_text': self._aero_description()}

    def run_grip_budget(self, blocking: bool = False, iters: int | None = None,
                        sweep_g=None):
        """Tab 2.  blocking=True runs inline (tests) and returns the study dict
        the compute function produced.  iters / sweep_g override the bisection
        depth and the utilization-curve g list (tests keep them small)."""
        if self._grip_worker is not None and self._grip_worker.isRunning():
            self._grip_progress.setText('already running — wait or cancel')
            return None
        self._refresh_car_note()
        try:
            _tire, solver = self._solver()
        except Exception as e:                   # noqa: BLE001
            self._grip_progress.setText(f'could not build the car: {e}')
            return None
        cfg = self.grip_config()
        if iters is not None:
            cfg['iters'] = int(iters)
        if sweep_g is not None:
            cfg['sweep_g'] = [float(g) for g in sweep_g]
        cfg['mu_scale'] = float(solver._mu_scale)
        # limits at every grip scale in the user's list (Dynamics panel)
        if cfg.get('grip_scales') is None and hasattr(self._main, 'grip_scale_list'):
            cfg['grip_scales'] = list(self._main.grip_scale_list())
        self.last_grip_cfg = cfg
        self.grip_slot.busy('solving…'); self.grip_slot.summary('—')
        self.grip_table.setRowCount(0)
        if blocking:
            study = CS.grip_budget_study(
                solver, lateral_g=cfg['lateral_g'], aero_cases=cfg['aero_cases'],
                sweep_g=cfg['sweep_g'], iters=cfg['iters'],
                progress=self._grip_progress.setText,
                grip_scales=cfg.get('grip_scales'))
            self._finish_grip(study)
            return study
        self._btn_grip.setEnabled(False); self._btn_grip_cancel.setEnabled(True)
        self._grip_worker = _GripWorker(solver, cfg)
        self._grip_worker.note.connect(self._grip_progress.setText)
        self._grip_worker.done.connect(self._finish_grip)
        self._grip_worker.failed.connect(self._fail_grip)
        self._grip_worker.start()
        return None

    def _cancel_grip(self):
        if self._grip_worker is not None:
            self._grip_worker.cancel = True
            self._grip_progress.setText('cancelling after the current case…')

    def _fail_grip(self, msg):
        self._btn_grip.setEnabled(True); self._btn_grip_cancel.setEnabled(False)
        self._grip_progress.setText(f'run failed: {msg}')

    def _finish_grip(self, study):
        self._btn_grip.setEnabled(True); self._btn_grip_cancel.setEnabled(False)
        self.last_study = study
        self._draw_grip(study)
        self._fill_grip_table(study)
        self._show_grip_readout(study)
        self._grip_progress.setText(f'done — {len(study["cases"])} case(s), grip scale {study["mu_scale"]:.2f}')

    def _show_grip_readout(self, study):
        lines = [f'PER-CORNER LIMIT — the first lateral g at which ANY tyre is out of budget '
                 f'(utilization = demand / (μ_peak(Fz, IA) × {study["mu_scale"]:.2f} × Fz)):']
        for label, c in study['cases'].items():
            lc, la = c['corner_limit'], c['axle_limit']
            flag = '  [search ceiling — no corner ever exceeded 1.0]' if lc['hit_upper_bound'] else ''
            lines.append(f'  {label:<34s} {lc["limit_g"]:.3f} g  binding {lc["binding"]}'
                         f'   |   axle-aggregate reference (app criterion) {la["limit_g"]:.3f} g '
                         f'binding {la["binding"]}{flag}')
        for label, c in study['cases'].items():
            rows = c.get('limits_by_grip_scale') or []
            if rows:
                lines.append(f'LIMITS BY GRIP SCALE — {label}:')
                for r_ in rows:
                    lines.append(
                        f'  grip x{r_["grip_scale"]:.2f}:  per-corner {r_["corner_limit_g"]:.3f} g '
                        f'({r_["corner_binding"]})   axle {r_["axle_limit_g"]:.3f} g '
                        f'({r_["axle_binding"]})   traction {r_["traction_g"]:.3f} g   '
                        f'braking {r_["braking_g"]:.3f} g')
        g0 = study['lateral_g']
        lines.append(f'AT {g0:.2f} g:')
        for label, c in study['cases'].items():
            at = c['at_g']
            if 'corners' not in at:
                lines.append(f'  {label:<34s} solve failed: {at.get("error")}')
                continue
            cells = '  '.join(f'{k} {at["corners"][k]["utilization"]:.3f}'
                              + ('*' if at['corners'][k]['below_data_floor'] else '') for k in CS.CORNERS)
            lines.append(f'  {label:<34s} {cells}   worst {at["worst_corner"]} {at["worst_utilization"]:.3f}')
        lines.append('* = Fz below the tyre file\'s lowest tested load (μ clamped at the floor).  '
                     'IA < 0 = leaning into the turn.')
        self.grip_readout.setText('\n'.join(lines))

    def _draw_grip(self, study):
        s = self.grip_slot
        ax = s.axes(1)
        styles = ['-', '--', ':', '-.']
        for k, (label, c) in enumerate(study['cases'].items()):
            sw = c['sweep']
            if not sw['g']:
                continue
            ls = styles[k % len(styles)]
            for corner in CS.CORNERS:
                ax.plot(sw['g'], sw['utilization'][corner], ls=ls, color=_CC[corner], lw=1.8,
                        label=f'{corner} {label}')
            lim = c['corner_limit']['limit_g']
            ax.axvline(lim, color=_WHT, ls=ls, lw=0.9, label='_lim')
            ax.text(lim, 0.02, f' {label}: all four inside budget to {lim:.3f} g',
                    color=_WHT, fontsize=7, rotation=90, va='bottom', ha='left',
                    transform=ax.get_xaxis_transform())
        ax.axhline(1.0, color=_WHT, lw=0.8, label='_one')
        ax.set_xlabel('lateral acceleration (g)', fontsize=8.5)
        ax.set_ylabel('tyre utilization = demand / budget', fontsize=8.5)
        ax.set_title('per-corner utilization vs lateral g', fontsize=9)
        if ax.get_lines():
            s.legend(ax, ncol=2, loc='upper left')
        s.finish()
        parts = []
        for label, c in study['cases'].items():
            lc = c['corner_limit']
            parts.append(f'{label}: {lc["limit_g"]:.3f} g, {lc["binding"]} runs out first')
        s.summary('WHAT IT MEANS: the car stays inside every tyre\'s budget up to — '
                  + '; '.join(parts)
                  + '.  The inner tyres bind, not the loaded outer ones: they carry little load and '
                    'lean out of the turn.  The axle-aggregate number (pair budget) is the app\'s '
                    'canonical limit and is always higher.')

    def _fill_grip_table(self, study):
        t = self.grip_table
        rows = []
        for label, c in study['cases'].items():
            for point, tab in (('chosen g', c['at_g']),
                               ('per-corner limit', (c['corner_limit'] or {}).get('at_limit'))):
                if not tab or 'corners' not in tab:
                    continue
                for corner in CS.CORNERS:
                    d = tab['corners'][corner]
                    rows.append((label, point, f'{tab["lateral_g"]:.3f}', corner, f'{d["Fz_N"]:.0f}',
                                 f'{d["inclination_deg"]:+.2f}', f'{d["budget_N"]:.0f}',
                                 f'{d["demand_N"]:.0f}', f'{d["utilization"]:.3f}',
                                 'yes' if d['below_data_floor'] else 'no', d['utilization']))
        t.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, v in enumerate(r[:-1]):
                it = QTableWidgetItem(v)
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                                    if j >= 2 and j != 3 and j != 9 else Qt.AlignmentFlag.AlignCenter)
                if j == 8 and np.isfinite(r[-1]) and r[-1] > 1.0:
                    it.setForeground(Qt.GlobalColor.red)
                t.setItem(i, j, it)

    def _copy_grip_table(self):
        if not self.last_study:
            self._status.setText('nothing to copy — run the grip budget first')
            return
        QApplication.clipboard().setText(CS.grip_budget_table_text(self.last_study))
        self._status.setText('grip-budget table copied (tab-separated)')
