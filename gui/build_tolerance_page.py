"""gui/build_tolerance_page.py — BUILD TOLERANCE page (Ctrl+Shift+1).

Two tabs (user 2026-09-26):
  AERO HEAVE    "ride height shrink over velocity caused by downforce in
                straights and corners" — per-corner ride-height drop vs speed on
                the straight and in steady corners (vahan.aero_ride: the ONE
                nonlinear wheel-rate curve + the steady-state solver's roll,
                jacking and tyre load), aero = the Dynamics panel package.  No
                lap simulation.
  CG TOLERANCE  "irl [the CG] won't be [what it is] in sim ... what our CG
                tolerance is, height and front to rear" — the car re-solved
                with the CG moved (height, fore-aft), its direct metrics
                measured by vahan.cg_tolerance.car_metrics, and for each metric
                the CG band inside which it stays within an allowance.

WHAT IT COMPUTES.  Nothing itself (ONE MODEL): every number is the app's own
steady-state solver / lap simulator, rebuilt through MainWindow's own builders
with the car dict's cg_z_mm / cg_y_mm temporarily moved (restored after, even
on error).  Allowances are editable engineering choices (defaults in
vahan.cg_tolerance.METRICS).  Text follows docs/DESIGN.md."""
from __future__ import annotations

import traceback

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTabWidget, QDoubleSpinBox,
    QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QApplication,
    QSplitter, QComboBox, QLineEdit,
)

from vahan import cg_tolerance as CT
from gui.plot_dialog import ReadableCanvas

_INK, _MUTED, _ACCENT = '#ECECEE', '#9A9AA2', '#E23B48'
_PANEL, _LINE, _PAPER = '#141416', '#2A2A2E', '#0B0B0D'
_YEL, _RED, _WHT, _BLU = '#FFD600', '#E53935', '#FFFFFF', '#42A5F5'     # graph lines only


def laptime_page(main):
    """The Lap Time page (built lazily without leaving the current page)."""
    if getattr(main, '_laptime_page', None) is None:
        cur = main._pages.currentIndex()
        main._switch_page(1)
        main._switch_page(cur)
    return main._laptime_page


def aero_package(main) -> dict:
    """The car's aero package = the Dynamics panel's (reference downforce at a
    reference speed + centre of pressure), as Cl·A."""
    p = main._dynamics_panel.get_custom_aero_params()
    v = float(p['V_ref_kph']) / 3.6; rho = float(p['air_density'])
    cla = 2.0 * float(p['F_ref_N']) / (rho * v * v) if v > 0 else 0.0
    return dict(cla_m2=cla, cop_rear=float(p['cop_rear_pct']) / 100.0, rho=rho,
                label=f"{p['F_ref_N']:.0f} N at {p['V_ref_kph']:.1f} km/h, {p['cop_rear_pct']:.0f} % rear")


def package_from(F_ref_N: float, V_ref_kph: float, cop_rear_pct: float, rho: float = 1.225) -> dict:
    v = float(V_ref_kph) / 3.6
    return dict(cla_m2=2.0 * float(F_ref_N) / (rho * v * v) if v > 0 else 0.0, cop_rear=float(cop_rear_pct) / 100.0, rho=rho,
                label=f"{F_ref_N:.0f} N at {V_ref_kph:.1f} km/h, {cop_rear_pct:.0f} % rear")


def car_long_limits(main) -> tuple:
    """(braking g, acceleration g) the car can do (grip / power, no aero grip),
    from the dynamics solver - the defaults of the page's two boxes."""
    ss = main._build_dynamics_solver(); m = ss.max_accel_g(20.0, 0.0)
    return float(m['braking_g']), float(m['effective_g'])


def aero_ride(main, radii_m=(9.0, 15.0, 30.0), speeds_kph=None, package=None,
              brake_g: float | None = None, accel_g: float | None = None) -> dict:
    """Wheel travel + ride height vs speed from the ONE steady-state solver (its
    corner travel includes the aero sink): straight line + each corner radius
    up to the grip limit on that radius (aero included in the limit).  Also
    the damper travel limits of each axle (the app's own spring-travel range)."""
    from vahan import aero_ride as AR
    from vahan.corner_speed import per_corner_limit_g
    ss = main._build_dynamics_solver(); pk = package or aero_package(main)
    sp = np.linspace(0.0, 130.0, 53) if speeds_kph is None else np.asarray(speeds_kph, float)
    lim_f = main._spring_travel_range(main._solvers['FL'], 'FL'); lim_r = main._spring_travel_range(main._solvers['RL'], 'RL')
    bg, ag = car_long_limits(main)
    bg = bg if brake_g is None else float(brake_g); ag = ag if accel_g is None else float(accel_g)
    out = {'package': pk, 'straight': AR.ride_sweep(ss, pk['cla_m2'], pk['cop_rear'], pk['rho'], sp), 'corners': [],
           'brake': AR.ride_sweep(ss, pk['cla_m2'], pk['cop_rear'], pk['rho'], sp, longitudinal_g=-abs(bg)),
           'accel': AR.ride_sweep(ss, pk['cla_m2'], pk['cop_rear'], pk['rho'], sp, longitudinal_g=abs(ag)),
           'limits_mm': {'F': (lim_f[0] * 1000, lim_f[1] * 1000), 'R': (lim_r[0] * 1000, lim_r[1] * 1000)}}
    for R in radii_m:
        per_g = AR.aero_per_corner_N(pk['cla_m2'], pk['cop_rear'], pk['rho'], np.sqrt(G_EARTH * float(R)))
        lim = float(per_corner_limit_g(ss, aero_Fz_per_g=per_g)['limit_g'])
        vmax = np.sqrt(lim * G_EARTH * float(R)) * 3.6
        c = AR.ride_sweep(ss, pk['cla_m2'], pk['cop_rear'], pk['rho'], np.linspace(0.0, vmax, 40),
                          radius_m=float(R), grip_limit_g=lim + 1e-9)
        c['v_limit_kph'] = vmax
        out['corners'].append(c)
    return out


G_EARTH = 9.81


def cg_builds(main, axis: str, offsets_mm, with_lap: bool = True) -> dict:
    """Build (on the GUI thread, through the app's own builders) one solver +
    lap simulator per CG offset.  axis 'height' (cg_z_mm) or 'fore_aft'
    (cg_y_mm, + = rearward).  The design CG is restored before returning."""
    key, default = ('cg_z_mm', 280.0) if axis == 'height' else ('cg_y_mm', 845.0)
    x0 = float(main._car.get(key, default))
    lp = laptime_page(main) if with_lap else None
    track = getattr(lp, '_track', None) if lp is not None else None
    nd = int(lp._ndetail.value()) if lp is not None else 400
    jobs = []
    try:
        for d in offsets_mm:
            main._car[key] = x0 + float(d)
            ss = main._build_dynamics_solver()
            sim = lp.build_sim() if (lp is not None and track is not None) else None
            jobs.append((ss, sim))
    finally:
        main._car[key] = x0
        main._build_dynamics_solver()
    return dict(axis=axis, key=key, x0_mm=x0, x_mm=[x0 + float(d) for d in offsets_mm], jobs=jobs, track=track, n_detail=nd)


def run_builds(b: dict, progress=None) -> dict:
    """Solve every built car (pure computation — safe off the GUI thread)."""
    out = []
    for i, (ss, sim) in enumerate(b['jobs']):
        out.append(CT.car_metrics(ss, sim, b['track'], n_detail=b['n_detail']))
        if progress:
            progress(i + 1, len(b['jobs']))
    return dict(axis=b['axis'], key=b['key'], x0_mm=b['x0_mm'], x_mm=b['x_mm'], metrics=out)


def cg_sweep(main, axis: str, offsets_mm, with_lap: bool = True, progress=None) -> dict:
    """Blocking build + solve (tests / headless)."""
    return run_builds(cg_builds(main, axis, offsets_mm, with_lap), progress)


class _CGWorker(QThread):
    step = pyqtSignal(str, int, int)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, builds):
        super().__init__(); self._builds = builds

    def run(self):
        try:
            res = {}
            for b in self._builds:
                res[b['axis']] = run_builds(b, lambda i, n, a=b['axis']: self.step.emit(a, i, n))
            self.done.emit(res)
        except Exception:
            self.failed.emit(traceback.format_exc())


class BuildTolerancePage(QWidget):
    def __init__(self, main):
        super().__init__()
        self._main = main
        self._aero = None
        self._sweeps = {}
        self.setStyleSheet(f'QWidget{{background:{_PAPER};color:{_INK};font-size:13px;}}'
                           'QDoubleSpinBox,QPushButton{background:#1e1e24;border:1px solid #444;'
                           'border-radius:3px;padding:2px 4px;} QPushButton:hover{background:#2a2a32;}'
                           f'QTableWidget{{background:{_PANEL};gridline-color:{_LINE};'
                           "font-family:'Consolas','SF Mono',monospace;font-size:12px;}"
                           f'QHeaderView::section{{background:{_PAPER};color:{_MUTED};padding:4px;border:0;'
                           f'border-bottom:2px solid {_LINE};font-size:11px;}}')
        v = QVBoxLayout(self); v.setContentsMargins(14, 10, 14, 10); v.setSpacing(6)
        top = QHBoxLayout()
        eb = QLabel('BUILD TOLERANCE'); eb.setStyleSheet(f'color:{_ACCENT};font-size:11px;font-weight:650;letter-spacing:2px;')
        top.addWidget(eb); top.addStretch(1)
        top.addWidget(QLabel('Grip multiplier'))
        self._grip = QDoubleSpinBox(); self._grip.setRange(0.10, 1.50); self._grip.setDecimals(2); self._grip.setSingleStep(0.01)
        self._grip.setToolTip('Tyre grip multiplier (belt-to-track).  Same value as every grip box in the app — editing it '
                              'here changes it everywhere.  Re-run the tab to apply.')
        top.addWidget(self._grip)
        v.addLayout(top)
        self._tabs = QTabWidget(); v.addWidget(self._tabs, 1)
        self._tabs.addTab(self._build_aero_tab(), 'Aero heave')
        self._tabs.addTab(self._build_cg_tab(), 'CG tolerance')
        self._status = QLabel(''); self._status.setWordWrap(True); self._status.setStyleSheet(f'color:{_MUTED};font-size:12px;')
        v.addWidget(self._status)

    # ── aero heave ───────────────────────────────────────────────────────
    def _build_aero_tab(self):
        w = QWidget(); v = QVBoxLayout(w)
        cap = QLabel('Suspension travel of each wheel caused by downforce: on the straight (the car sinks) and in steady '
                     'corners of the radii below, up to the car\'s grip limit (downforce + body roll).  Dashed lines = '
                     'where the dampers bottom out / fully extend.  Aero = the Dynamics panel package.')
        cap.setWordWrap(True); cap.setStyleSheet(f'color:{_MUTED};font-size:12px;'); v.addWidget(cap)
        h = QHBoxLayout()

        def spin(lbl, lo, hi, val, dec, suf, tip):
            h.addWidget(QLabel(lbl)); sb = QDoubleSpinBox(); sb.setRange(lo, hi); sb.setDecimals(dec); sb.setValue(val)
            sb.setSuffix(suf); sb.setToolTip(tip); h.addWidget(sb); return sb
        self._aF = spin('Downforce', 0, 20000, 350, 0, ' N', 'Total downforce at the reference speed')
        self._aV = spin('at', 1, 400, 88.5, 1, ' km/h', 'Reference speed of that downforce')
        self._aC = spin('Rear share', 0, 100, 54, 0, ' %', 'Share of the downforce on the rear axle (centre of pressure)')
        self._aVmax = spin('Straight to', 20, 400, 130, 0, ' km/h', 'Top speed plotted on the straight')
        self._bG = spin('Braking', 0.0, 4.0, 1.6, 2, ' g', 'Straight-line braking g (default = the car\'s braking limit)')
        self._aG = spin('Accel', 0.0, 3.0, 1.0, 2, ' g', 'Straight-line acceleration g (default = the car\'s traction/power limit)')
        rb = QPushButton("Car's values"); rb.setToolTip('Reload downforce / speed / rear share (Dynamics panel) and the braking / acceleration limits')
        rb.clicked.connect(self._load_car_aero); h.addWidget(rb)
        h.addWidget(QLabel('Corner radii (m)'))
        self._radii = QLineEdit('9, 15, 30'); self._radii.setMaximumWidth(120); h.addWidget(self._radii)
        b = QPushButton('Calculate'); b.clicked.connect(self.refresh_aero); h.addWidget(b); h.addStretch(1)
        v.addLayout(h)
        self._afig = Figure(figsize=(12, 8), facecolor='#1b1b1e'); self._acanvas = ReadableCanvas(self._afig)
        v.addWidget(self._acanvas, 1)
        self._atab = QTableWidget(0, 9); self._atab.setMaximumHeight(210)
        self._atab.setHorizontalHeaderLabels(['Case', 'Speed km/h', 'g (lateral / longitudinal)', 'Downforce N', 'Front outside travel mm',
                                              'Front inside travel mm', 'Rear outside travel mm', 'Rear inside travel mm',
                                              'Ride height lost mm (incl. tyre)'])
        self._atab.verticalHeader().setVisible(False); self._atab.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._atab.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        v.addWidget(self._atab)
        return w

    def _load_car_aero(self):
        p = self._main._dynamics_panel.get_custom_aero_params()
        self._aF.setValue(float(p['F_ref_N'])); self._aV.setValue(float(p['V_ref_kph'])); self._aC.setValue(float(p['cop_rear_pct']))
        bg, ag = car_long_limits(self._main); self._bG.setValue(bg); self._aG.setValue(ag)

    def showEvent(self, e):
        super().showEvent(e)
        if not getattr(self, '_aero_loaded', False):
            try:
                self._load_car_aero(); self._aero_loaded = True
            except Exception:
                pass

    def refresh_aero(self):
        try:
            radii = [float(x) for x in self._radii.text().replace(';', ',').split(',') if x.strip()]
            pk = package_from(self._aF.value(), self._aV.value(), self._aC.value(), aero_package(self._main)['rho'])
            t = aero_ride(self._main, radii, np.linspace(0.0, self._aVmax.value(), 53), package=pk,
                          brake_g=self._bG.value(), accel_g=self._aG.value()); self._aero = t
        except Exception as e:
            traceback.print_exc(); self._status.setText(f'Aero ride height failed: {e}'); return None
        f = self._afig; f.clf()
        lim = t['limits_mm']
        st = t['straight']
        ncol = max(3, len(t['corners']))
        wheels = (('FL', 'front, outside', _YEL, '-'), ('FR', 'front, inside', _YEL, ':'),
                  ('RL', 'rear, outside', _RED, '-'), ('RR', 'rear, inside', _RED, ':'))

        def limits(a):
            for ax_, col in (('F', _YEL), ('R', _RED)):
                a.axhline(lim[ax_][1], color=col, lw=0.9, ls='--', alpha=0.6)
                a.axhline(lim[ax_][0], color=col, lw=0.9, ls='--', alpha=0.6)
            a.text(0.01, max(lim['F'][1], lim['R'][1]), f" dampers bottom out (F {lim['F'][1]:.0f} / R {lim['R'][1]:.0f} mm)",
                   color=_MUTED, fontsize=7, va='bottom', transform=a.get_yaxis_transform())
            a.text(0.01, min(lim['F'][0], lim['R'][0]), f" dampers fully extended (F {lim['F'][0]:.0f} / R {lim['R'][0]:.0f} mm)",
                   color=_MUTED, fontsize=7, va='top', transform=a.get_yaxis_transform())

        axes = []
        for k, (key, title) in enumerate((('straight', 'Straight, steady speed'),
                                          ('brake', f"Straight, braking {abs(t['brake']['longitudinal_g']):.2f} g"),
                                          ('accel', f"Straight, accelerating {t['accel']['longitudinal_g']:.2f} g"))):
            d = t[key]; a = f.add_subplot(2, ncol, 1 + k)
            a.plot(d['speed_kph'], d['travel_mm']['FL'], color=_YEL, lw=2.2, label='front wheels')
            a.plot(d['speed_kph'], d['travel_mm']['RL'], color=_RED, lw=2.2, ls='--', label='rear wheels')
            for cn in ('FL', 'RL'):
                a.annotate(f"{d['travel_mm'][cn][-1]:+.1f}", (d['speed_kph'][-1], d['travel_mm'][cn][-1]), xytext=(4, 0),
                           textcoords='offset points', color=_INK, fontsize=8, va='center')
            a.set_title(title, color=_INK, fontsize=10); axes.append(a)
        for k, c in enumerate(t['corners']):
            a = f.add_subplot(2, ncol, ncol + 1 + k, sharey=axes[0])
            for key, lab, col, ls in wheels:
                y = c['travel_mm'][key]; a.plot(c['speed_kph'], y, color=col, ls=ls, lw=2.0, label=lab)
                j = int(np.nanargmax(np.where(np.isfinite(y), c['speed_kph'], -1)))
                a.annotate(f'{y[j]:+.0f}', (c['speed_kph'][j], y[j]), xytext=(4, 0), textcoords='offset points',
                           color=_INK, fontsize=8, va='center')
            a.set_title(f"{c['radius_m']:.0f} m corner, up to its grip limit ({c['v_limit_kph']:.0f} km/h = {c['grip_limit_g']:.2f} g)",
                        color=_INK, fontsize=9)
            axes.append(a)
        for a in axes[1:3]:
            a.sharey(axes[0])
        for a in axes:
            limits(a)
            a.set_facecolor('#111114'); a.set_xlabel('Speed (km/h)', color=_MUTED, fontsize=8); a.tick_params(colors=_MUTED, labelsize=8)
            a.grid(color='#2a2a2e', lw=0.6); a.axhline(0, color='#777', lw=0.9)
            for sp in a.spines.values(): sp.set_color('#333')
            a.legend(facecolor='#1b1b1e', edgecolor='#333', labelcolor=_INK, fontsize=7, loc='upper left', bbox_to_anchor=(0.0, 0.9))
            a.set_ylabel('Wheel travel (mm), + = bump', color=_MUTED, fontsize=8)
        pk = t['package']
        f.suptitle(f"Suspension travel from downforce + braking / accelerating / cornering — aero {pk['label']}, Cl·A {pk['cla_m2']:.2f} m²",
                   color=_MUTED, fontsize=9)
        f.tight_layout(); self._acanvas.draw()
        rows = []
        for sv in (60.0, 100.0, 120.0):
            i = int(np.argmin(abs(st['speed_kph'] - sv)))
            rows.append(('straight', st['speed_kph'][i], 0.0, st['downforce_N'][i], *[st['travel_mm'][cn][i] for cn in ('FL', 'FR', 'RL', 'RR')],
                         0.5 * (st['ride_drop_mm']['FL'][i] + st['ride_drop_mm']['RL'][i])))
        for key, nm in (('brake', 'braking'), ('accel', 'accelerating')):
            d = t[key]; i = int(np.argmin(abs(d['speed_kph'] - 100.0)))
            rows.append((f"straight, {nm} {abs(d['longitudinal_g']):.2f} g", d['speed_kph'][i], d['longitudinal_g'], d['downforce_N'][i],
                         *[d['travel_mm'][cn][i] for cn in ('FL', 'FR', 'RL', 'RR')], 0.5 * (d['ride_drop_mm']['FL'][i] + d['ride_drop_mm']['RL'][i])))
        for c in t['corners']:
            i = int(np.nanargmax(np.where(np.isfinite(c['travel_mm']['FL']), c['speed_kph'], -1)))
            rows.append((f"{c['radius_m']:.0f} m at the grip limit", c['speed_kph'][i], c['lateral_g'][i], c['downforce_N'][i],
                         *[c['travel_mm'][cn][i] for cn in ('FL', 'FR', 'RL', 'RR')], c['ride_drop_mm']['FL'][i]))
        self._atab.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for cix, val in enumerate(row):
                it = QTableWidgetItem(val if cix == 0 else f'{val:.0f}' if cix in (1, 3) else f'{val:.2f}' if cix == 2 else f'{val:+.1f}')
                if cix: it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._atab.setItem(r, cix, it)
        self._status.setText('Wheel travel = the dynamics solver\'s own corner travel (it now includes the aero sink) — '
                             'the same travel the Dynamics page uses.')
        return t

    # ── CG tolerance ─────────────────────────────────────────────────────
    def _build_cg_tab(self):
        w = QWidget(); v = QVBoxLayout(w)
        cap = QLabel('The car is re-solved with its CG moved (the model everything else uses).  "Band" = how far the '
                     'built CG may sit from the design CG before that metric leaves its allowance; the tightest band '
                     'is the build tolerance.  Allowances are choices — edit them in the table (Allowance column).')
        cap.setWordWrap(True); cap.setStyleSheet(f'color:{_MUTED};font-size:12px;'); v.addWidget(cap)
        h = QHBoxLayout()
        def spin(lbl, lo, hi, val, suf):
            h.addWidget(QLabel(lbl)); s = QDoubleSpinBox(); s.setRange(lo, hi); s.setValue(val); s.setDecimals(0); s.setSuffix(suf); h.addWidget(s); return s
        self._hspan = spin('Height sweep ±', 5, 150, 40, ' mm'); self._yspan = spin('Front-rear sweep ±', 5, 200, 50, ' mm')
        self._step = spin('Step', 2, 50, 10, ' mm')
        self._with_lap = QCheckBox('Include lap time (slower)'); self._with_lap.setChecked(True); h.addWidget(self._with_lap)
        b = QPushButton('Run CG sweep'); b.clicked.connect(self.refresh_cg); h.addWidget(b); h.addStretch(1)
        v.addLayout(h)
        sp = QSplitter(Qt.Orientation.Vertical)
        self._cgfig = Figure(figsize=(10, 5), facecolor='#1b1b1e'); self._cgcanvas = ReadableCanvas(self._cgfig)
        sp.addWidget(self._cgcanvas)
        self._cgtab = QTableWidget(0, 9)
        self._cgtab.setHorizontalHeaderLabels(['Metric', 'Design value', 'Allowance', 'Per 10 mm higher CG', 'Height band mm',
                                               'Per 10 mm rearward CG', 'Front-rear band mm', 'Counts', 'Unit'])
        self._cgtab.verticalHeader().setVisible(False)
        self._cgtab.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._cgtab.horizontalHeader().setStretchLastSection(True)
        self._cgtab.itemChanged.connect(self._allowance_edited)
        sp.addWidget(self._cgtab)
        self._cgsum = QLabel(''); self._cgsum.setWordWrap(True); self._cgsum.setStyleSheet(f'color:{_INK};font-size:13px;font-weight:600;')
        v.addWidget(sp, 1); v.addWidget(self._cgsum)
        self._allow = dict(CT.DEFAULT_ALLOWANCE)
        return w

    def refresh_cg(self, blocking: bool = False):
        st = float(self._step.value())
        try:
            builds = []
            for axis, span in (('height', self._hspan.value()), ('fore_aft', self._yspan.value())):
                offs = np.arange(-span, span + 0.5 * st, st)
                self._status.setText(f'Building the car at {len(offs)} CG {axis} positions…'); QApplication.processEvents()
                builds.append(cg_builds(self._main, axis, offs, self._with_lap.isChecked()))
        except Exception as e:
            traceback.print_exc(); self._status.setText(f'CG sweep failed while building: {e}'); return None
        if blocking:
            self._sweeps = {b['axis']: run_builds(b) for b in builds}; self._fill_cg()
            self._status.setText('CG sweep done (the design CG was never changed while solving).'); return self._sweeps
        self._cg_worker = _CGWorker(builds)
        self._cg_worker.step.connect(lambda a, i, n: self._status.setText(
            f'Solving CG {a}: {i}/{n} (lap sims run in the background — the app stays usable)'))
        self._cg_worker.done.connect(self._cg_done)
        self._cg_worker.failed.connect(lambda m: self._status.setText('CG sweep failed: ' + m.splitlines()[-1]))
        self._cg_worker.start()
        return None

    def _cg_done(self, res):
        self._sweeps = res; self._fill_cg()
        self._status.setText('CG sweep done (the design CG was never changed while solving).')

    def _allowance_edited(self, it):
        if it.column() != 2 or getattr(self, '_filling', False):
            return
        try:
            self._allow[CT.METRIC_KEYS[it.row()]] = abs(float(it.text()))
            self._fill_cg()
        except (ValueError, IndexError):
            pass

    def _fill_cg(self):
        if not self._sweeps:
            return
        th = CT.tolerance_table(self._sweeps['height'], self._allow)
        ty = CT.tolerance_table(self._sweeps['fore_aft'], self._allow)
        fmt_b = lambda lo, hi: f"{'beyond sweep' if not np.isfinite(lo) else f'{lo:+.0f}'} / {'beyond sweep' if not np.isfinite(hi) else f'{hi:+.0f}'}"
        self._filling = True
        self._cgtab.setRowCount(len(th))
        cnt = {'both': 'either way', 'down': 'a loss', 'up': 'a rise'}
        for r, (a, b) in enumerate(zip(th, ty)):
            vals = [a['label'], f"{a['design']:.3f}", f"{a['allowance']:g}", f"{a['per_10mm']:+.4f}", fmt_b(a['lo'], a['hi']),
                    f"{b['per_10mm']:+.4f}", fmt_b(b['lo'], b['hi']), cnt[a['direction']] + ('' if a['headline'] else ' (not in headline)'), a['unit']]
            for c, s in enumerate(vals):
                it = QTableWidgetItem(s)
                if c != 2:
                    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if 1 <= c <= 6:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._cgtab.setItem(r, c, it)
        self._filling = False
        hl, hh, khl, khh = CT.overall_band(th); yl, yh, kyl, kyh = CT.overall_band(ty)
        f = lambda x: 'beyond the sweep' if not np.isfinite(x) else f'{x:+.0f} mm'
        self._cgsum.setText(f"CG height tolerance: {f(hl)} (lower, set by {khl or '—'})  to  {f(hh)} (higher, set by {khh or '—'}).   "
                            f"CG front-rear tolerance: {f(yl)} (forward, set by {kyl or '—'})  to  {f(yh)} (rearward, set by {kyh or '—'}).")
        # plots: each metric's change vs offset, allowance shaded
        fig = self._cgfig; fig.clf()
        keys = [k for k in CT.METRIC_KEYS if any(np.isfinite(m.get(k, np.nan)) for m in self._sweeps['height']['metrics'])]
        n = len(keys)
        for i, k in enumerate(keys):
            ax = fig.add_subplot(2, (n + 1) // 2, i + 1); ax.set_facecolor('#111114')
            for sw, col, ls, lab in ((self._sweeps['height'], _YEL, '-', 'height'), (self._sweeps['fore_aft'], _RED, '--', 'front-rear (+ rearward)')):
                x = np.asarray(sw['x_mm']) - sw['x0_mm']; y = np.array([m.get(k, np.nan) for m in sw['metrics']])
                y0 = y[int(np.argmin(abs(x)))]; ax.plot(x, y - y0, color=col, ls=ls, lw=1.8, label=lab)
            al = self._allow[k]; d = CT.DIRECTION[k]
            ys = [np.array([m.get(k, np.nan) for m in sw['metrics']]) for sw in (self._sweeps['height'], self._sweeps['fore_aft'])]
            dev = np.concatenate([y - y[int(np.argmin(abs(np.asarray(sw['x_mm']) - sw['x0_mm'])))] for y, sw in
                                  zip(ys, (self._sweeps['height'], self._sweeps['fore_aft']))])
            dev = dev[np.isfinite(dev)]
            span = max(float(np.max(np.abs(dev))) if dev.size else 0.0, 1.5 * al)
            ax.set_ylim(-1.1 * span, 1.1 * span)                 # limits from the DATA (+ the allowance)
            ax.axhspan(-al if d in ('both', 'down') else -1.1 * span, al if d in ('both', 'up') else 1.1 * span,
                       color='#ffffff', alpha=0.07)
            ax.axhline(0, color='#555', lw=0.8)
            ax.set_title(dict((m[0], m[1]) for m in CT.METRICS)[k], color=_INK, fontsize=8)
            ax.set_xlabel('CG offset from design (mm)', color=_MUTED, fontsize=7); ax.tick_params(colors=_MUTED, labelsize=7)
            ax.grid(color='#2a2a2e', lw=0.5)
            for sp in ax.spines.values(): sp.set_color('#333')
            if i == 0: ax.legend(facecolor='#1b1b1e', edgecolor='#333', labelcolor=_INK, fontsize=7)
        fig.suptitle('Change from the design value (shaded = allowance)', color=_MUTED, fontsize=9)
        fig.tight_layout(); self._cgcanvas.draw()
