"""gui/bearings_page.py — CONTROL-ARM SPHERICAL BEARINGS page (Ctrl+0).

WHY IT EXISTS.  The user sizes the inboard control-arm spherical plain bearings
in the SKF plain-bearing calculator (productselect.skf.com, "size-lubrication /
plain bearing"), which asks per load case for: radial force Fr (kN), axial force
Fa (kN), oscillation time (s, one full 4-beta cycle), half the angle of
oscillation beta (deg), load direction, temperature.  This page produces exactly
those rows from the loaded model, for two bore orientations:
  1  bore NORMAL to the arm plane       -> the arm swing is TILT of the ball
                                           (changes with travel), turning ~ 0
  2  bore ALONG the front-rear pickup line -> the arm swing is TURNING about the
                                           bore, tilt stays 0

WHAT IT COMPUTES.  Nothing itself (ONE MODEL):
  * pickup forces  : gui.wheel_package.compute_case -> vahan.loads.compute_all_corners
                     (the Loads page path), ComponentLoads.chassis_forces
  * poses          : the same solve's ComponentLoads.state; static and full
                     bump/droop from the kinematic solvers
  * angles, force split, load-direction class : vahan.spherical_bearings
  * oscillation time default: 1 / ride frequency of that axle (vahan.ride_solve),
    an explicit, editable choice — the road decides the real value.

MOTION CYCLES (one SKF load case each), per wheel-package case:
  lateral cases       : this corner as the OUTER wheel <-> as the INNER wheel
                        (left turn <-> right turn, same longitudinal g)
  straight-line cases : static <-> the case (braking / acceleration)
The larger of the two ends' forces is reported (both ends shown in the detail).

Text follows docs/DESIGN.md (no yellow text; accent red for eyebrows)."""
from __future__ import annotations

import traceback

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QDoubleSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QApplication, QCheckBox,
)

from vahan import spherical_bearings as SB

_INK, _MUTED, _ACCENT = '#ECECEE', '#9A9AA2', '#E23B48'
_PANEL, _LINE, _PAPER = '#141416', '#2A2A2E', '#0B0B0D'

_PICKUP_NAME = {'uca_front': 'Upper arm, front pickup', 'uca_rear': 'Upper arm, rear pickup',
                'lca_front': 'Lower arm, front pickup', 'lca_rear': 'Lower arm, rear pickup'}
_ORIENT_NAME = {'normal': '1  normal to arm plane', 'pivot': '2  along pickup line'}


def ride_period_s(win) -> dict:
    """{'F': s, 'R': s} = 1 / ride natural frequency per axle (the app's own ride model)."""
    from vahan import ride_solve as RS
    veh = win._build_dynamics_solver()._veh
    mf, mr = RS.sprung_corner_masses_kg(veh)
    ff = RS.ride_frequency_Hz(veh.ride_rate_front_Npm, mf)
    fr = RS.ride_frequency_Hz(veh.ride_rate_rear_Npm, mr)
    return {'F': 1.0 / ff, 'R': 1.0 / fr, 'F_Hz': ff, 'R_Hz': fr}


def compute_bearing_rows(win, temperature_C: float = 20.0, osc_time_s: dict | None = None,
                         cases=None, swivel_limit_deg: float = SB.SWIVEL_LIMIT_DEG) -> dict:
    """All SKF rows for the LEFT front and LEFT rear corner (the right side is the
    mirror; both turn directions are already in every lateral cycle).
    osc_time_s: {'F': s, 'R': s} override; None = 1 / ride frequency.
    swivel_limit_deg: the rod end's swivel limit on its built-in angle; rows carry over_limit.
    Every case is solved at its case SPEED with the aero load of the Dynamics
    panel package (wheel_package.compute_case); 'cases_aero' lists them."""
    from gui import wheel_package as WP
    cases = cases or WP.CASES
    per = ride_period_s(win)
    osc = osc_time_s or {'F': per['F'], 'R': per['R']}
    cache = {}

    def solve(lat, lon):
        k = (round(lat, 6), round(lon, 6))
        if k not in cache:
            cache[k] = WP.compute_case(win, lat, lon)[0]
        return cache[k]

    cases_aero = [(c[0], WP.case_aero(win, c[1], c[2])) for c in cases]
    rows = []
    for corner in ('FL', 'RL'):
        solver = win._solvers[corner]
        st0 = solver.solve(0.0)
        lo, hi = win._spring_travel_range(solver, corner)
        full = [solver.solve(float(lo)), solver.solve(float(hi))]
        L0 = solve(0.0, 0.0)[corner]
        for name, lat, lon, *_rule in cases:
            if abs(lat) > 1e-9:
                A, B = solve(lat, lon)[corner], solve(-lat, lon)[corner]
                ends = f'{name}: this wheel outside <-> inside'
            else:
                A, B = L0, solve(lat, lon)[corner]
                ends = f'straight, 0 g <-> {name}'
                if abs(float(A.travel_m) - float(B.travel_m)) < 1e-6:
                    # the dynamics solve reports pitch but puts no pitch travel on
                    # the corners, so both ends are the static pose (KNOWN GAP)
                    ends += '  [swing 0: no pitch travel in load model]'
            if not (A.valid and B.valid and A.state is not None and B.state is not None):
                rows.append(dict(corner=corner, case=name, invalid=True,
                                 reason=(A.invalid_reason or B.invalid_reason or 'load solve invalid')))
                continue
            rows += SB.case_rows(st0, A.state, B.state, A.chassis_forces, B.chassis_forces,
                                 corner=corner, case=name, ends=ends,
                                 osc_time_s=osc[corner[0]], temperature_C=temperature_C,
                                 full_travel_states=full, swivel_limit_deg=swivel_limit_deg)
    return dict(rows=rows, ride=per, osc=osc, temperature_C=temperature_C,
                swivel_limit_deg=float(swivel_limit_deg), cases_aero=cases_aero)


def swivel_summary(rows) -> dict:
    """{(corner, pickup, orientation): (built-in angle deg, over_limit, worst
    tilt deg)} — the per-pickup picture the net reports; a pickup that is over
    the limit in BOTH bore orientations has no rod-end orientation that works."""
    out = {}
    for r in rows:
        if r.get('invalid'):
            continue
        k = (r['corner'], r['pickup'], r['orientation'])
        t_in, t_w = float(r['tilt_installed_deg']), float(r['tilt_worst_deg'])
        if k not in out:
            out[k] = (t_in, bool(r['over_limit']), t_w)
        else:
            out[k] = (max(out[k][0], t_in), out[k][1] or bool(r['over_limit']), max(out[k][2], t_w))
    return out


class BearingsPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self._main = main
        self._data = None
        self.setStyleSheet(f'background:{_PAPER}; color:{_INK};')
        v = QVBoxLayout(self); v.setContentsMargins(16, 12, 16, 12); v.setSpacing(8)

        eb = QLabel('CONTROL-ARM SPHERICAL BEARINGS')
        eb.setStyleSheet(f'color:{_ACCENT}; font-size:11px; font-weight:650; letter-spacing:2px;')
        v.addWidget(eb)
        t = QLabel('SKF plain-bearing calculator inputs, per inboard pickup and load case')
        t.setStyleSheet(f'color:{_INK}; font-size:15px; font-weight:600;')
        v.addWidget(t)
        cap = QLabel(
            'Rod ends are INLINE with their arm leg (eye square to the leg).  Orientation 1 (bolt normal to the '
            'arm plane): no built-in tilt; the arm swing TILTS the ball, so "Half angle" is ~0 and the swing is '
            'tilt.  Orientation 2 (bolt along the front-rear pickup line): the arm swing TURNS the ball, and the '
            'bolt sits at a built-in tilt of 90° minus the leg-to-pickup-line angle, constant through travel.  '
            'Fr / Fa are the larger of the two ends of each motion cycle.  Oscillation time is one '
            'full cycle; default = 1 / ride frequency of that axle (an assumed road input — edit it).  '
            'A rod end swivels about 27°, so its BUILT-IN angle (90° minus the leg-to-pickup-line angle; '
            '0 for the plane-normal bolt) must stay under the swivel limit; rows over it are marked OVER LIMIT.  '
            'The worst tilt over full travel is shown for information.  '
            'Each load case is solved at its speed with the Dynamics-panel aero package (Loads page path).')
        cap.setWordWrap(True); cap.setStyleSheet(f'color:{_MUTED}; font-size:12px;')
        v.addWidget(cap)

        h = QHBoxLayout(); h.setSpacing(10)
        h.addWidget(self._lab('Axle'))
        self._axle = QComboBox(); self._axle.addItems(['Front + rear', 'Front', 'Rear'])
        h.addWidget(self._axle)
        h.addWidget(self._lab('Bore'))
        self._orient = QComboBox(); self._orient.addItems(['Both orientations', '1  normal to arm plane', '2  along pickup line'])
        h.addWidget(self._orient)
        h.addWidget(self._lab('Temperature °C'))
        self._temp = QDoubleSpinBox(); self._temp.setRange(-40, 250); self._temp.setValue(20.0); self._temp.setDecimals(0)
        h.addWidget(self._temp)
        self._use_ride = QCheckBox('Oscillation time = 1 / ride frequency'); self._use_ride.setChecked(True)
        h.addWidget(self._use_ride)
        h.addWidget(self._lab('else F / R s'))
        self._tf = QDoubleSpinBox(); self._tf.setRange(0.01, 60); self._tf.setDecimals(3); self._tf.setValue(0.33)
        self._tr = QDoubleSpinBox(); self._tr.setRange(0.01, 60); self._tr.setDecimals(3); self._tr.setValue(0.33)
        h.addWidget(self._tf); h.addWidget(self._tr)
        h.addWidget(self._lab('Rod-end swivel limit (built-in angle) °'))
        self._swivel = QDoubleSpinBox(); self._swivel.setRange(0.5, 90.0); self._swivel.setDecimals(1)
        self._swivel.setSingleStep(0.5); self._swivel.setSuffix(' °'); self._swivel.setValue(SB.SWIVEL_LIMIT_DEG)
        self._swivel.setToolTip('How far the rod end can swivel (about 27°).  Each row\'s BUILT-IN angle '
                                '(90° minus the leg-to-pickup-line angle) is compared against it.')
        h.addWidget(self._swivel)
        self._go = QPushButton('Calculate'); self._go.clicked.connect(self.refresh)
        self._copy = QPushButton('Copy for SKF'); self._copy.clicked.connect(self._copy_rows)
        h.addWidget(self._go); h.addWidget(self._copy); h.addStretch(1)
        v.addLayout(h)
        for w_ in (self._axle, self._orient):
            w_.currentIndexChanged.connect(self._fill)

        self._status = QLabel(''); self._status.setStyleSheet(f'color:{_MUTED}; font-size:12px;')
        v.addWidget(self._status)

        self._cols = ['Corner', 'Pickup', 'Bore', 'Load case (motion cycle)', 'Fr  radial kN', 'Fa  axial kN',
                      'Oscillation time s', 'Half angle of oscillation °', 'Tilt in this cycle ±°',
                      'Built-in angle (rod end inline) °', 'Worst tilt, full travel °', 'Swivel limit °',
                      'Arm swing droop→bump °', 'Load direction', 'Temp °C', 'Why this load direction']
        self._tab = QTableWidget(0, len(self._cols))
        self._tab.setHorizontalHeaderLabels(self._cols)
        self._tab.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._tab.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._tab.verticalHeader().setVisible(False)
        self._tab.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._tab.horizontalHeader().setStretchLastSection(True)
        self._tab.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        self._tab.setColumnWidth(3, 300)
        self._tab.setWordWrap(False)
        self._tab.setStyleSheet(
            f'QTableWidget {{ background:{_PANEL}; color:{_INK}; gridline-color:{_LINE}; '
            f"font-family:'Consolas','SF Mono',monospace; font-size:12px; border:1px solid {_LINE}; }} "
            f'QHeaderView::section {{ background:{_PAPER}; color:{_MUTED}; font-size:11px; '
            f'text-transform:uppercase; letter-spacing:1px; padding:4px; border:none; '
            f'border-bottom:2px solid {_LINE}; }}')
        v.addWidget(self._tab, 1)

    def _lab(self, s):
        l = QLabel(s); l.setStyleSheet(f'color:{_MUTED}; font-size:12px;'); return l

    # ── compute ──────────────────────────────────────────────────────────
    def refresh(self, blocking: bool = True):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            osc = None if self._use_ride.isChecked() else {'F': self._tf.value(), 'R': self._tr.value()}
            self._status.setText('Solving the load cases (Loads page model)…'); QApplication.processEvents()
            self._data = compute_bearing_rows(self._main, float(self._temp.value()), osc,
                                              swivel_limit_deg=float(self._swivel.value()))
            p = self._data['ride']
            if self._use_ride.isChecked():
                self._tf.setValue(p['F']); self._tr.setValue(p['R'])
            n_over = sum(1 for r in self._data['rows'] if r.get('over_limit'))
            aero_txt = '; '.join(f"{nm}: {ca['speed_kph']:.0f} km/h, aero {ca['total_N']:.0f} N"
                                 for nm, ca in self._data.get('cases_aero', []))
            self._status.setText(
                f"Ride frequency front {p['F_Hz']:.2f} Hz / rear {p['R_Hz']:.2f} Hz → oscillation time "
                f"{self._data['osc']['F']:.3f} / {self._data['osc']['R']:.3f} s.  Left corners shown; right = mirror.  "
                f"Rod-end swivel limit {self._data['swivel_limit_deg']:.1f}° on the built-in angle: {n_over} of "
                f"{len(self._data['rows'])} rows OVER LIMIT.  Case speeds (aero from the Dynamics panel package"
                f"{'' if any(ca['aero_Fz'] for _, ca in self._data.get('cases_aero', [])) else ', Apply Aero OFF'}): "
                + aero_txt)
            self._fill()
        except Exception as e:
            traceback.print_exc()
            self._status.setText(f'Failed: {e}')
        finally:
            QApplication.restoreOverrideCursor()
        return self._data

    def _visible_rows(self):
        if not self._data:
            return []
        ax = self._axle.currentIndex(); o = self._orient.currentIndex()
        out = []
        for r in self._data['rows']:
            if ax == 1 and r['corner'][0] != 'F': continue
            if ax == 2 and r['corner'][0] != 'R': continue
            if r.get('invalid'):
                out.append(r); continue
            if o == 1 and r['orientation'] != 'normal': continue
            if o == 2 and r['orientation'] != 'pivot': continue
            out.append(r)
        return out

    def _fill(self):
        rows = self._visible_rows()
        self._tab.setRowCount(len(rows))
        for i, r in enumerate(rows):
            over = bool(r.get('over_limit')) and not r.get('invalid')
            if r.get('invalid'):
                vals = [r['corner'], '—', '—', r['case'], '—', '—', '—', '—', '—', '—', '—', '—', '—',
                        'load solve invalid', '—', r['reason']]
            else:
                lim = (f"OVER LIMIT ({r['swivel_limit_deg']:.1f})" if over
                       else f"within {r['swivel_limit_deg']:.1f}")
                vals = [r['corner'], _PICKUP_NAME[r['pickup']], _ORIENT_NAME[r['orientation']], r['ends'],
                        f"{r['Fr_kN']:.2f}", f"{r['Fa_kN']:.2f}", f"{r['osc_time_s']:.3f}",
                        f"{r['half_angle_deg']:.2f}", f"{r['tilt_half_deg']:.2f}", f"{r['tilt_installed_deg']:.1f}",
                        f"{r['tilt_worst_deg']:.1f}", lim, f"{r['arm_swing_full_deg']:.1f}",
                        r['load_direction'], f"{r['temperature_C']:.0f}", r['load_direction_reason']]
            for j, s in enumerate(vals):
                it = QTableWidgetItem(str(s))
                if j in (3, 15):
                    it.setToolTip(str(s))
                if 4 <= j <= 12 or j == 14:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if over:
                    # docs/DESIGN.md: emphasis = accent red text (no yellow, no green)
                    it.setForeground(QColor(_ACCENT))
                self._tab.setItem(i, j, it)

    def _copy_rows(self):
        rows = [r for r in self._visible_rows() if not r.get('invalid')]
        head = ['corner', 'pickup', 'bore', 'load case', 'Fr kN', 'Fa kN', 'oscillation time s',
                'half angle deg', 'load direction', 'temperature C', 'built-in angle deg', 'worst tilt deg',
                'swivel limit deg', 'swivel']
        lines = ['\t'.join(head)]
        for r in rows:
            lines.append('\t'.join([r['corner'], r['pickup'], r['orientation'], r['ends'], f"{r['Fr_kN']:.3f}",
                                    f"{r['Fa_kN']:.3f}", f"{r['osc_time_s']:.3f}", f"{r['half_angle_deg']:.2f}",
                                    r['load_direction'], f"{r['temperature_C']:.0f}", f"{r['tilt_installed_deg']:.1f}",
                                    f"{r['tilt_worst_deg']:.1f}", f"{r['swivel_limit_deg']:.1f}",
                                    'OVER LIMIT' if r.get('over_limit') else 'within limit']))
        QApplication.clipboard().setText('\n'.join(lines))
        self._status.setText(f'Copied {len(rows)} rows (tab-separated) to the clipboard.')
