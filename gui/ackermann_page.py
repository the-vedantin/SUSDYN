"""
gui/ackermann_page.py — the ACKERMANN page.

WHY IT EXISTS.  The Ackermann work used to be four buttons on the Analysis
panel that answered in TEXT POPUPS.  The user's brief: *"since this is kind of
hefty, lets make it its own page to analyze ackermann using various methods"*
and *"show me graphs, show me what the numbers mean, stop throwing me a word
salad thats all vomit"*.  So: one page, five graphs, one caption and one
plain-English line per graph, and nothing else.

WHAT IT COMPUTES.  Nothing.  Every number on this page comes out of vahan:

    graph 1 + 2   vahan.ackermann.ackermann_bucket
    graph 3       vahan.analysis_plots.ackermann_fz_fy_data
    graph 4       vahan.analysis_plots.ackermann_pair_potential
    graph 5       vahan.laptime.AckermannStationModel (which calls
                  vahan.ackermann.solve_ackermann_force)  ->  LapSimulator,
                  PER STATION, on every track, averaged

This page ARRANGES and PLOTS.  Duplicated physics is the mechanism that
produced rival Ackermann answers before; there is one implementation of each
method and it lives in vahan/.

PALETTE.  The user is colourblind and has said he hates blue: this page's
chrome and every accent on these five figures are yellow / red / white / warm
grey, and where more than three series share an axis they are separated by
LINE STYLE as well as colour.  (The FL-yellow / FR-red / RL-white / RR-blue
CORNER convention used elsewhere in the app is a different thing and is
untouched — no per-corner data is drawn here.)
"""
from __future__ import annotations

import math
import os

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QCheckBox, QLineEdit, QSizePolicy,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ── in-app figure palette: NO BLUE ──────────────────────────────────────────
# The user confuses YELLOW with ORANGE, and red with orange — so amber/orange
# is not a data colour anywhere on this page.  Four data colours only, and
# where more than four series share an axis they separate by LINE STYLE.
_YEL = '#FFD600'        # yellow
_RED = '#E53935'        # red — also the ALERT colour (limits, warnings)
_WHT = '#FFFFFF'        # white
_WGY = '#C8BCA6'        # warm grey
_DIM = '#8a8177'        # dim warm grey (background lines, iso-slip)
_GRID = '#1c1a17'
_TICK = '#8a8177'
_TEXT = '#d8d2c8'
_FIG_BG = '#0b0b0c'
_AX_BG = '#111014'

# Fixed radii for the per-corner demand plot: FSAE hairpins are 3-9 m, the
# skidpad path is about 8 m, and 12 m stands in for a fast sweeper.  Colour
# AND line style differ on every one (4 series > 3).
_RADII = [(3.0, _YEL, '-'), (5.0, _RED, '--'),
          (8.0, _WHT, '-.'), (12.0, _WGY, ':')]

# Style matrix for the Ackermann sweep families (up to ~10 settings on one
# axis — four colours cannot carry that, so colour x LINE STYLE, and the
# styles inside one colour are made as different from each other as dashes
# allow: solid, long-dash, fine-dot.
_SWEEP_STYLES = [
    (_YEL, '-'),               (_RED, '--'),              (_WHT, '-.'),
    (_WGY, ':'),               (_YEL, (0, (6, 2))),       (_RED, (0, (1, 1.6))),
    (_WHT, (0, (3, 1, 1, 1))), (_WGY, '-'),               (_YEL, (0, (1, 1.6))),
    (_RED, '-'),               (_WHT, (0, (6, 2))),       (_WGY, '--'),
]

# The percentage's denominator (the kinematic travel spread) is what makes the
# percentage explode.  Below this many degrees the ratio is not a number worth
# reading — graph 2 shades that band.
_DENOM_FLOOR_DEG = 1.0

# Graph 5's noise band is MEASURED, not assumed: AckermannStationModel.
# binning_error_g() re-solves the front-axle cap at radii deliberately chosen
# to fall between the table's bins, and the worst disagreement is priced at
# the lap sim's own grip sensitivity (one extra lap per track at grip x1.01).
# Nothing here is a hard-coded noise floor any more.


def _parse_list(text: str, fallback) -> list:
    """Read a comma-separated number list from a QLineEdit, tolerantly."""
    out = []
    for tok in str(text).replace(';', ',').split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return out or list(fallback)


def _zero_crossing(x, y):
    """First x where y crosses zero (linear interp), or None.  Skips NaNs."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    for i in range(1, len(y)):
        if y[i - 1] == 0.0:
            return float(x[i - 1])
        if y[i - 1] * y[i] < 0.0:
            t = y[i - 1] / (y[i - 1] - y[i])
            return float(x[i - 1] + t * (x[i] - x[i - 1]))
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  one graph cell
# ═══════════════════════════════════════════════════════════════════════════
class _Slot(QFrame):
    """One graph cell: short caption, canvas, one plain-English summary line.

    The canvas carries ``_plot_data`` in the same
    ``[(ax, [(x, y, label, colour), ...]), ...]`` shape the kinematics canvas
    uses, so hover value-readout works here identically — the tyre-grip plots
    were unhoverable for months precisely because that list was never filled.
    """

    def __init__(self, caption: str):
        super().__init__()
        self.setFrameShape(QFrame.Shape.NoFrame)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)

        self._cap = QLabel(caption)
        self._cap.setStyleSheet('color:#FFD600; font-size:11px; '
                                'font-weight:bold;')
        self._cap.setWordWrap(True)
        lay.addWidget(self._cap)

        self.fig = Figure(facecolor=_FIG_BG, layout='constrained')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        self.canvas.setMinimumHeight(300)
        self.canvas._plot_data = []
        lay.addWidget(self.canvas, 1)

        self._sum = QLabel('—')
        self._sum.setStyleSheet(
            "color:#d8d2c8; font-family:'Consolas','SF Mono',monospace;"
            'font-size:11px; background:#0e0d10; padding:5px;'
            'border:1px solid #2a2823; border-radius:3px;')
        self._sum.setWordWrap(True)
        lay.addWidget(self._sum)

        try:
            from gui.main_window import HoverAnnotator
            self._hover = HoverAnnotator(self.canvas)
        except Exception:
            self._hover = None

    # ── drawing helpers ─────────────────────────────────────────────────
    def axes(self, n: int = 1):
        """Clear and hand back n fresh, styled axes."""
        self.fig.clf()
        self.canvas._plot_data = []
        axs = [self.fig.add_subplot(1, n, i + 1) for i in range(n)]
        for ax in axs:
            ax.set_facecolor(_AX_BG)
            ax.tick_params(colors=_TICK, labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor('#2a2823')
            ax.grid(True, color=_GRID, lw=0.6)
            ax.xaxis.label.set_color(_TICK)
            ax.yaxis.label.set_color(_TICK)
            ax.title.set_color(_TEXT)
        return axs[0] if n == 1 else axs

    def legend(self, ax, **kw):
        kw.setdefault('fontsize', 7)
        kw.setdefault('loc', 'best')
        ax.legend(facecolor='#0e0d10', labelcolor=_TEXT, edgecolor='#2a2823',
                  framealpha=0.85, **kw)

    def summary(self, text: str):
        self._sum.setText(text)

    def busy(self, text: str = 'computing…'):
        ax = self.axes()
        ax.text(0.5, 0.5, text, transform=ax.transAxes, ha='center',
                va='center', color=_TICK, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        self.canvas.draw_idle()

    def finish(self):
        """Register every drawn curve for hover readout, then draw."""
        data = []
        for ax in self.fig.get_axes():
            series = []
            for ln in ax.get_lines():
                lbl = ln.get_label() or ''
                if not lbl or lbl.startswith('_'):
                    continue
                xd = np.asarray(ln.get_xdata(), float)
                yd = np.asarray(ln.get_ydata(), float)
                if len(xd) < 2:
                    continue          # markers / peak dots, not curves
                series.append((xd, yd, lbl, ln.get_color()))
            if series:
                data.append((ax, series))
        self.canvas._plot_data = data
        self.canvas.draw_idle()

    def save(self, path: str):
        self.fig.savefig(path, dpi=130, facecolor=self.fig.get_facecolor())


# ═══════════════════════════════════════════════════════════════════════════
#  worker — every heavy call, off the GUI thread, emitting stage by stage
# ═══════════════════════════════════════════════════════════════════════════
class _AckWorker(QThread):
    stage = pyqtSignal(str, object)     # (stage key, payload)
    note = pyqtSignal(str)              # status line
    done = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, tire, solver, cfg, tracks):
        super().__init__()
        self._tire, self._solver, self._cfg = tire, solver, cfg
        self._tracks = tracks

    def run(self):
        try:
            self._bucket()
            self._fzfy()
            self._pair()
            if self._cfg['do_lap']:
                self._lap()
            else:
                self.stage.emit('lap', None)
            self.done.emit()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.failed.emit(str(e))

    # ── graphs 1 + 2 ────────────────────────────────────────────────────
    def _bucket(self):
        from vahan.ackermann import ackermann_bucket
        c = self._cfg
        out = {}
        for R in c['radii']:
            self.note.emit(f'per-corner toe demand at {R:.0f} m…')
            out[R] = ackermann_bucket(
                self._tire, self._solver, radius_m=float(R),
                lat_g_list=tuple(c['g_list']),
                grip_multiplier=c['grip'], aero=c['aero'] or False)
        self.stage.emit('bucket', out)

    # ── graph 3 ─────────────────────────────────────────────────────────
    def _fzfy(self):
        from vahan.analysis_plots import ackermann_fz_fy_data
        c = self._cfg
        self.note.emit('Fz-Fy capability vs delivered…')
        self.stage.emit('fzfy', ackermann_fz_fy_data(
            self._tire, self._solver, radius_m=c['radius'],
            ackermann_pct=c['as_built_pct'], grip_multiplier=c['grip'],
            lat_g_list=tuple(c['g_list'])))

    # ── graph 4 ─────────────────────────────────────────────────────────
    def _pair(self):
        from vahan.analysis_plots import ackermann_pair_potential
        c = self._cfg
        self.note.emit('pair analysis (RCVD ch.7)…')
        self.stage.emit('pair', ackermann_pair_potential(
            self._tire, self._solver, lat_g=c['pair_g'],
            ackermann_list=tuple(c['sweep'])))

    # ── graph 5 — the lap-time chain, PER STATION ────────────────────────
    def _lap(self):
        """Ackermann -> FRONT-AXLE CAPABILITY AND SCRUB AT EVERY STATION'S OWN
        RADIUS -> lap time, on every track, averaged.

        WHAT CHANGED AND WHY.  The previous version of this chain turned one
        Ackermann setting into ONE GLOBAL grip scale (from the YMD trim sweep
        at ONE radius) and re-ran the lap.  That threw away the entire effect,
        because the Ackermann benefit is RADIUS-DEPENDENT — measured on this
        car, the front-axle force ceiling swings 23% between -100% and +100%
        on a 2.5 m corner, 5.7% on a 5 m corner, 1.3% on a 10 m corner and
        0.03% on a 50 m sweeper.  A single number cannot say that, and the
        thing the user actually wants to weigh is exactly how much of a lap is
        spent at each of those radii.

        So now the lap sim carries an AckermannStationModel: at every station
        it looks up that station's own path radius, asks
        vahan.ackermann.solve_ackermann_force what the front axle can hold
        there at this setting, caps the local lateral limit with it, and
        charges the front tyres' SCRUB as longitudinal drag at the steer angle
        that just holds the circle.  No tyre physics is written here or in
        laptime.py — both only call and combine.

        Still true, and still worth saying: the RAW lap sim is blind to
        Ackermann (moving this car's rack +/-25 mm swung it +40.8% -> +64.5%
        and the lap time did not move a digit, 41.488176 s).  That is why a
        chain is needed at all.
        """
        from vahan.laptime import Track, AckermannStationModel
        c = self._cfg
        pcts = [float(p) for p in c['sweep']]
        base = float(c['lap_grip'])
        self.note.emit('building the per-radius Ackermann capability table '
                       '(~1 min)…')
        mdl = AckermannStationModel(self._tire, self._solver, pcts,
                                    grip_multiplier=base,
                                    aero=c['aero'] or False)
        mdl.build(progress_cb=lambda m, d, t: self.note.emit(m))
        if not np.isfinite(mdl._ay_cap).any():
            self.stage.emit('lap', {'tracks': {}, 'why':
                                    'the front-axle capability table did not '
                                    'solve anywhere'})
            return
        self.note.emit('measuring the method’s own noise floor…')
        err = mdl.binning_error(progress_cb=lambda m: self.note.emit(m))
        err_g, err_sc = err['ay_cap_g'], err['scrub_frac']
        # The perturbed table: front-axle cap pushed DOWN by its own error (so
        # it binds harder, the direction that actually costs lap time) and the
        # scrub pushed UP by its own.  Both worst-case and both correlated
        # across radii — the pessimistic reading on purpose.
        mdl_err = mdl.perturbed(-err_g, 1.0 + err_sc)

        tracks = {}
        n_lap = len(pcts) + 2
        for label, fname in self._tracks:
            path = os.path.join(_repo_root(), 'tracks', fname)
            if not os.path.isfile(path):
                continue
            tr = Track.from_json(path)
            self.note.emit(f'{label}: control lap 1/{n_lap} (Ackermann off)…')
            l_ctrl = self._lap_time(tr, base)
            laps, scrub, capped = [], [], []
            for i, p in enumerate(pcts):
                self.note.emit(f'{label}: lap {i + 2}/{n_lap} at {p:+.0f}% '
                               f'Ackermann…')
                r = self._lap_run(tr, base, mdl, p)
                laps.append(float(r.lap_time_s))
                scrub.append(float(np.mean(r.scrub_N))
                             if r.scrub_N is not None else 0.0)
                capped.append(int(np.sum(np.asarray(r.ay_cap_local_ms2)
                                         < np.max(r.ay_cap_local_ms2) - 1e-9))
                              if r.ay_cap_local_ms2 is not None else 0)
            # THE NOISE BAND, MEASURED, NOT CONVERTED.  Re-run ONE lap against
            # a table moved by exactly this table's own binning error, at the
            # setting where the mechanism is MOST ACTIVE on this track (most
            # front-capped stations; ties broken by the slowest lap).  A band
            # measured where nothing binds is a band of zero and tells you
            # nothing — that is exactly what the first version of this did.
            k_band = int(np.argmax(np.asarray(capped) * 1e6
                                   + np.asarray(laps)))
            self.note.emit(f'{label}: noise-band lap {n_lap}/{n_lap} '
                           f'(cap {-err_g:+.4f} g, scrub '
                           f'x{1 + err_sc:.3f}, at {pcts[k_band]:+.0f}%)…')
            l_band = self._lap_time_ack(tr, base, mdl_err, pcts[k_band])
            band = abs(l_band - laps[k_band])
            tracks[label] = {
                'pct': np.asarray(pcts, float),
                'lap_s': np.asarray(laps, float),
                'scrub_mean_N': np.asarray(scrub, float),
                'capped_stations': np.asarray(capped, int),
                'control_lap_s': float(l_ctrl),
                'band_s': float(band),
                'band_pct_ref': float(pcts[k_band]),
            }
        self.stage.emit('lap', {'tracks': tracks, 'why': None,
                                'err_g': float(err_g),
                                'err_scrub_frac': float(err_sc),
                                'base_grip': base,
                                'spread': mdl.spread_report(),
                                'failed_cells': int(mdl.n_failed_cells)})

    def _lap_run(self, track, grip_scale, ack_model=None, ack_pct=0.0):
        from vahan.laptime import LapSimulator
        veh = self._solver._veh
        sim = LapSimulator(
            self._solver, cla_m2=0.0, cda_m2=float(veh.cda_m2),
            air_density=float(veh.air_density_kg_m3),
            aero_cop_rear_frac=0.5, grip_scale=float(grip_scale),
            static_rh_front_mm=50.0, static_rh_rear_mm=50.0)
        sim.set_gearbox([2.750, 2.000, 1.667, 1.444, 1.304, 1.208],
                        primary_ratio=1.0, final_drive=3.545,
                        redline_rpm=11000.0)
        if ack_model is not None:
            sim.set_ackermann(ack_model, ack_pct)
        # n_detail=8: the per-corner suspension detail pass is diagnostic and
        # does not enter the lap time, so it is trimmed to the minimum here.
        return sim.simulate(track, n_detail=8)

    def _lap_time(self, track, grip_scale) -> float:
        return float(self._lap_run(track, grip_scale).lap_time_s)

    def _lap_time_ack(self, track, grip_scale, model, pct) -> float:
        return float(self._lap_run(track, grip_scale, model, pct).lap_time_s)


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════════════
#  the page
# ═══════════════════════════════════════════════════════════════════════════
class AckermannPage(QWidget):
    """Ackermann analysis — five graphs, five one-line answers.  `main` is the
    MainWindow, which owns the ONE solved car and the ONE tyre model."""

    def __init__(self, main):
        super().__init__()
        self._main = main
        self._worker: _AckWorker | None = None
        self._data: dict = {}
        self._build()
        self._refresh_tire_note()

    # ── UI ──────────────────────────────────────────────────────────────
    def _build(self):
        from gui.panels import _spin
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        side = QVBoxLayout(); side.setSpacing(6)

        title = QLabel('ACKERMANN')
        title.setStyleSheet('color:#FFD600; font-size:15px; font-weight:bold;')
        side.addWidget(title)

        sub = QLabel('Five methods, one page.  Degrees are the answer; the '
                     'percentage is only shown while its reference means '
                     'anything.')
        sub.setStyleSheet('color:#8a8177; font-size:10px;')
        sub.setWordWrap(True)
        side.addWidget(sub)

        def hdr(text):
            l = QLabel(text)
            l.setStyleSheet('color:#FFD600; font-size:11px; font-weight:bold;'
                            'padding-top:6px;')
            side.addWidget(l)

        def grid():
            g = QGridLayout(); g.setSpacing(4)
            side.addLayout(g)
            return g

        def row(g, r, label, w, tip=''):
            lb = QLabel(label)
            lb.setStyleSheet('font-size:11px;')
            g.addWidget(lb, r, 0)
            if tip:
                w.setToolTip(tip); lb.setToolTip(tip)
            g.addWidget(w, r, 1)
            return w

        hdr('CORNER')
        g = grid()
        self._radius = row(g, 0, 'Corner radius:',
                           _spin(1.5, 200.0, 8.0, ' m', dec=1, step=0.5),
                           'Radius the Fz–Fy map, the pair loads and the YMD '
                           'trim sweep are solved at.  The required toe '
                           'difference CHANGES SIGN with radius, so solve at '
                           'the corners this car actually drives (FSAE '
                           'hairpins 3–9 m, skidpad path about 8 m).')
        self._grip = row(g, 1, 'Asphalt grip ×:',
                         _spin(0.40, 1.00, 0.70, ' ×', dec=2, step=0.05),
                         'Belt-rig-to-asphalt derate.  The tyre data is belt '
                         '(Calspan TIRF) and belt grip runs 0.65–0.75× '
                         'asphalt.  1.00 = raw belt curves: every slip angle '
                         'reads low and the limit sits near 2.3 g instead of '
                         '~1.6 g.  The lateral-g axis is always ROAD g.')
        self._pair_g = row(g, 2, 'Pair analysis at:',
                           _spin(0.2, 3.0, 1.5, ' g', dec=2, step=0.1),
                           'Cornering g for graph 4.  It sets the load '
                           'transfer, and therefore the two front wheel '
                           'loads, from the solved car.')
        self._aero = QCheckBox('Include aero downforce')
        self._aero.setToolTip(
            'Adds downforce at each row’s speed (speed follows from radius + '
            'lateral g) using the SAME aero package the Dynamics panel '
            'applies.  Downforce adds vertical load but NO lateral demand, so '
            'expect smaller slip angles and a higher grip limit.')
        side.addWidget(self._aero)

        hdr('LATERAL g SWEEP')
        g = grid()
        self._g_lo = row(g, 0, 'from:',
                         _spin(0.05, 3.0, 0.30, ' g', dec=2, step=0.05),
                         'Start low: at 0 g every slip angle is 0 and the '
                         'answer must come back at exactly 100% Ackermann — '
                         'a built-in sanity anchor.')
        self._g_hi = row(g, 1, 'to:',
                         _spin(0.3, 3.0, 2.00, ' g', dec=2, step=0.05),
                         'Run past the grip limit so the SATURATED rows show '
                         'up as gaps — a saturated wheel’s angle is a '
                         'peak-angle fallback, not a solution.')
        self._g_n = row(g, 2, 'points:',
                        _spin(4, 40, 12, '', dec=0, step=1),
                        'Rows per radius.  Each row is a full steady-state '
                        'solve plus four tyre-curve inversions.')

        hdr('RADII  (graph 1 + 2)')
        self._radii_txt = QLineEdit('3, 5, 8, 12')
        self._radii_txt.setToolTip(
            'One toe-demand curve per radius, in metres.  Four is the design '
            'maximum for readability (colour AND line style differ on each).')
        side.addWidget(self._radii_txt)

        hdr('ACKERMANN SWEEP  (graph 4 + 5)')
        self._sweep_txt = QLineEdit('-100, -60, -30, 0, 30, 45, 60, 100')
        self._sweep_txt.setToolTip(
            'Settings to sweep, in percent.  Negative = reverse (anti-) '
            'Ackermann.  100% is the KINEMATIC value, not a ceiling.')
        side.addWidget(self._sweep_txt)

        hdr('LAP TIME CHAIN  (graph 5)')
        self._do_lap = QCheckBox('Run the lap-time chain (~15 min)')
        self._do_lap.setChecked(True)
        self._do_lap.setToolTip(
            'The RAW lap simulator is blind to Ackermann — it is not in '
            'VehicleParams and the lap grip ceiling comes from a steady-state '
            'solve that carries no steer geometry.  This chain gives it '
            'sight, PER STATION: every station is capped by what the front '
            'axle can hold at THAT station’s radius and this setting, and the '
            'front tyres’ scrub is charged as drag there.  Cost: one '
            'capability table (~1 min) plus one lap sim per setting per '
            'track, plus two calibration laps per track.')
        side.addWidget(self._do_lap)
        g = grid()
        self._lap_grip = row(g, 0, 'Lap grip base ×:',
                             _spin(0.30, 1.20, 0.65, ' ×', dec=2, step=0.05),
                             'Baseline grip scale handed to the lap sim, so '
                             'the absolute lap times here match the Laptime '
                             'page at the same setting.')

        self._btn = QPushButton('►  ANALYSE')
        self._btn.setStyleSheet(
            'QPushButton { background:#FFD600; color:#0a0a0a; padding:10px;'
            ' font-weight:bold; font-size:13px; border-radius:4px; }'
            'QPushButton:hover { background:#FFEB3B; }'
            'QPushButton:disabled { background:#3a3a3a; color:#777; }')
        self._btn.clicked.connect(self._on_analyse)
        side.addWidget(self._btn)

        self._ymd_btn = QPushButton('YMD GRID  +100% → −70%')
        self._ymd_btn.setToolTip(
            'Yaw-moment diagrams at eight Ackermann settings on the chosen '
            'radius, plus trimmed-g / control / stability vs Ackermann with '
            'the 0.006 g noise band.  The balance-and-control view of the '
            'Ackermann decision — a slip-tangency demand curve cannot '
            'see what happens when the driver adds steer at the limit.')
        self._ymd_btn.clicked.connect(self._on_ymd_grid)
        side.addWidget(self._ymd_btn)

        mmd_row = QHBoxLayout()
        self._mmd_btn = QPushButton('FULL MMD SWEEP  (+100% ... -70%)')
        self._mmd_btn.setToolTip(
            'The Milliken Moment Diagram exactly as the Analysis panel '
            'draws it (same car, same speed/toe/ARB inputs), one tab per '
            'Ackermann setting: +100 / +70 / +50 / +30 / 0 / -30 / -70 %. '
            'PNGs saved to figs/.')
        self._mmd_btn.clicked.connect(
            lambda: self._main._on_plot_mmd_ack_sweep())
        mmd_row.addWidget(self._mmd_btn, 1)
        self._mmd_info = QPushButton('i')
        self._mmd_info.setFixedSize(22, 22)
        self._mmd_info.setToolTip('How to read the MMD')
        self._mmd_info.setStyleSheet(
            'QPushButton { border-radius: 11px; background: #FFD600; '
            'color: #0a0a0a; font-weight: bold; }')
        self._mmd_info.clicked.connect(self._show_mmd_help)
        mmd_row.addWidget(self._mmd_info, 0)
        side.addLayout(mmd_row)

        self._save_btn = QPushButton('Save all five figures to figs/')
        self._save_btn.clicked.connect(self._on_save)
        side.addWidget(self._save_btn)

        self._status = QLabel('')
        self._status.setStyleSheet('color:#FFB74D; font-size:11px;')
        self._status.setWordWrap(True)
        side.addWidget(self._status)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet('color:#2a2823;')
        side.addWidget(sep)

        self._tire_note = QLabel('')
        self._tire_note.setStyleSheet(
            "color:#d8d2c8; font-family:'Consolas','SF Mono',monospace;"
            'font-size:11px; background:#0e0d10; padding:6px;'
            'border:1px solid #2a2823; border-radius:3px;')
        self._tire_note.setWordWrap(True)
        self._tire_note.setToolTip(
            'The tyre and its PRESSURE are set on the Dynamics panel '
            '(Suspension page) — this page reads whatever is loaded there.  '
            'Pressure moves the answer: a blended model is not one tyre.')
        side.addWidget(self._tire_note)
        side.addStretch(1)

        inner = QWidget(); inner.setLayout(side)
        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(300)
        scroll.setStyleSheet('QScrollArea { border:0; background:#000; }')
        root.addWidget(scroll)

        # ── right: the grid of canvases ──────────────────────────────────
        self._slots = {
            'demand': _Slot('1 · PER-CORNER TOE DEMAND — how many degrees '
                            'more the inner wheel must be turned than the '
                            'outer, one curve per corner radius'),
            'pct': _Slot('2 · THE SAME THING AS A PERCENTAGE, and why we do '
                         'not spec it — the divider collapses'),
            'fzfy': _Slot('3 · Fz–Fy MAP — what each front wheel CAN make at '
                          'its own load vs what it DELIVERS'),
            'pair': _Slot('4 · PAIR ANALYSIS (RCVD ch.7) — axle force vs '
                          'steer angle, one curve per Ackermann setting'),
            'lap': _Slot('5 · LAP TIME vs ACKERMANN — mean of every track, '
                         'each station capped by what the front axle can hold '
                         'at ITS OWN radius, with the method’s measured noise '
                         'band drawn'),
        }
        gridw = QWidget()
        gl = QGridLayout(gridw)
        gl.setContentsMargins(0, 0, 0, 0); gl.setSpacing(8)
        order = ['demand', 'pct', 'fzfy', 'pair', 'lap']
        for i, key in enumerate(order):
            gl.addWidget(self._slots[key], i // 2, i % 2)
        gl.setRowStretch(0, 1); gl.setRowStretch(1, 1); gl.setRowStretch(2, 1)
        gl.setColumnStretch(0, 1); gl.setColumnStretch(1, 1)
        gridw.setMinimumHeight(1080)

        gscroll = QScrollArea()
        gscroll.setWidget(gridw)
        gscroll.setWidgetResizable(True)
        gscroll.setStyleSheet('QScrollArea { border:0; background:#000; }')
        root.addWidget(gscroll, 1)

        for s in self._slots.values():
            s.busy('press ANALYSE')

    # ── inputs ──────────────────────────────────────────────────────────
    def _refresh_tire_note(self):
        tm = getattr(self._main, '_tire_model', None)
        if tm is None:
            self._tire_note.setText(
                'NO TYRE DATA LOADED.\nEvery method on this page inverts or '
                'evaluates the measured tyre curve — load a file on the '
                'Dynamics panel first.')
            return
        psi = float(getattr(tm, 'pressure_psi', 0.0) or 0.0)
        blend = bool(getattr(tm, 'pressure_blended', False))
        avail = getattr(tm, 'available_pressures_psi', []) or []
        lines = [f'tyre     {getattr(tm, "tire_id", "?")}']
        if psi <= 0.0:
            lines.append('pressure BLENDED over every pressure in the file '
                         '— that is not one tyre.  Pick one on the Dynamics '
                         'panel.')
        elif blend and len(avail) > 1:
            lines.append(f'pressure {psi:.1f} psi  BLENDED — pick one')
        else:
            lines.append(f'pressure {psi:.1f} psi')
        if avail:
            lines.append('file has ' + ', '.join(f'{p:.0f}' for p in avail)
                         + ' psi')
        lines.append('set on the Dynamics panel — this page follows it')
        self._tire_note.setText('\n'.join(lines))

    def _cfg(self):
        lo = float(self._g_lo.value()); hi = float(self._g_hi.value())
        n = int(self._g_n.value())
        if hi <= lo:
            hi = lo + 0.1
        g_list = list(np.round(np.linspace(lo, hi, max(n, 4)), 4))
        radii = _parse_list(self._radii_txt.text(),
                            [r for r, _c, _s in _RADII])[:4]
        sweep = sorted(set(_parse_list(
            self._sweep_txt.text(),
            [-100, -60, -30, 0, 30, 45, 60, 100])))[:10]
        aero = None
        if self._aero.isChecked():
            # ONE aero path: the getter the dynamics view uses.  It gates on
            # the Apply Aero toggle; the checkbox HERE is the user asking for
            # aero, so lift the gate just for the read (same as
            # main_window._on_solve_ackermann).
            was = getattr(self._main, '_aero_active', False)
            try:
                self._main._aero_active = True
                aero = self._main._get_aero_Fz_per_g()
            except Exception:
                aero = None
            finally:
                self._main._aero_active = was
            if not aero:
                self._status.setText(
                    'Aero requested but no package data — run the aero target '
                    'solver or enter custom CFD on the Dynamics panel. '
                    'Analysing WITHOUT aero.')
        try:
            asb = float(self._main._probe_static_ackermann())
        except Exception:
            asb = float('nan')
        # A failed probe falls back to 0%, which is a REAL setting — so the
        # fallback must be announced, not silently treated as measured.
        asb_ok = bool(np.isfinite(asb))
        if not asb_ok:
            asb = 0.0
        return {
            'radius': float(self._radius.value()),
            'grip': float(self._grip.value()),
            'pair_g': float(self._pair_g.value()),
            'g_list': g_list, 'radii': radii, 'sweep': sweep,
            'aero': aero, 'as_built_pct': asb, 'as_built_probed': asb_ok,
            'do_lap': bool(self._do_lap.isChecked()),
            'lap_grip': float(self._lap_grip.value()),
        }

    # ── run ─────────────────────────────────────────────────────────────
    def _on_analyse(self):
        if self._worker is not None and self._worker.isRunning():
            # Two workers writing the same five slots interleave their stages
            # and the page ends up showing half of each run.
            self._status.setText('Already analysing — wait for this run.')
            return
        self._refresh_tire_note()
        tire = getattr(self._main, '_tire_model', None)
        if tire is None:
            self._status.setText('Load tyre data on the Dynamics panel first '
                                 '— every method here reads the measured '
                                 'curve.')
            return
        try:
            solver = self._main._build_dynamics_solver()
        except Exception as e:
            self._status.setText(f'Could not build the car: {e}')
            return
        cfg = self._cfg()
        # Make sure the as-built % lands in the swept family, so graphs 4 and
        # 5 always contain the car the user actually has.
        if all(abs(cfg['as_built_pct'] - p) > 0.6 for p in cfg['sweep']):
            cfg['sweep'] = sorted(cfg['sweep']
                                  + [round(cfg['as_built_pct'], 1)])
        self._cfg_used = cfg
        self._data = {}
        for s in self._slots.values():
            s.busy('computing…')
            s.summary('—')
        self._btn.setEnabled(False)
        self._status.setText('Analysing…')

        from gui.laptime_page import _TRACKS
        self._worker = _AckWorker(tire, solver, cfg, list(_TRACKS))
        self._worker.note.connect(self._status.setText)
        self._worker.stage.connect(self._on_stage)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_fail(self, msg):
        self._btn.setEnabled(True)
        self._status.setText(f'Analyse failed: {msg}')

    def _on_done(self):
        self._btn.setEnabled(True)
        asb = self._cfg_used['as_built_pct']
        if self._cfg_used.get('as_built_probed', True):
            self._status.setText(f'Done.  Car as built: {asb:+.1f}% Ackermann '
                                 f'(probed from the steering linkage).')
        else:
            self._status.setText(
                'Done — but the as-built Ackermann probe FAILED, so every '
                '"this car" reference on these graphs assumed 0%. Check the '
                'steering geometry on the Suspension page.')

    def _on_stage(self, key, payload):
        self._data[key] = payload
        try:
            if key == 'bucket':
                self._draw_demand(payload)
                self._draw_pct(payload)
            elif key == 'fzfy':
                self._draw_fzfy(payload)
            elif key == 'pair':
                self._draw_pair(payload)
            elif key == 'lap':
                self._draw_lap(payload)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._status.setText(f'Draw failed on {key}: {e}')

    def _on_ymd_grid(self):
        """Run vahan.analysis_plots.plot_ymd_ackermann_grid on the ONE
        solved model and show it in a dialog.  Heavy (~1-2 min)."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout
        from matplotlib.figure import Figure
        from vahan.analysis_plots import plot_ymd_ackermann_grid
        mw = self._main
        try:
            ss = mw._build_dynamics_solver()
            tm = mw._tire_model
            if tm is None:
                raise RuntimeError('no tyre model loaded')
            aero = (mw._get_aero_Fz_per_g()
                    if self._aero.isChecked() else None)
            self._ymd_btn.setEnabled(False)
            self._ymd_btn.setText('solving… (~1-2 min)')
            from PyQt6.QtWidgets import QApplication as _QA
            _QA.processEvents()
            fig = Figure(figsize=(15.5, 11.5))
            fig, trim = plot_ymd_ackermann_grid(
                tm, ss, radius_m=float(self._radius.value()),
                grip_multiplier=float(self._grip.value()),
                aero_Fz_per_g=aero, fig=fig)
            dlg = QDialog(self)
            dlg.setWindowTitle('YMD vs Ackermann — balance, control, '
                               'stability')
            dlg.resize(1400, 900)
            lay = QVBoxLayout(dlg)
            from PyQt6.QtWidgets import QPlainTextEdit
            rows = [' RCVD Fig 8.26 limit parameters (printed p334-335) '
                    'per Ackermann setting:',
                    ' ACK%   TRIM LIMIT   SI dCN/dAy      APEX Ay   '
                    'LIMIT CHARACTER          CONTROL @trim',
                    '        (g, N=0)     (neg=stable)    (g)       '
                    '(race target: NEUTRAL)   (N·m/deg)']
            ays = [r.get('Ay_trim_max', float('nan')) for r in trim]
            import numpy as _np
            best = _np.nanmax(ays) if _np.isfinite(ays).any() else float('nan')
            for r in trim:
                a = r.get('Ay_trim_max', float('nan'))
                tie = (' (all TIE: spread inside 0.006 g noise)'
                       if _np.isfinite(a) and best - a <= 0.006 else '')
                rows.append(
                    f" {r['pct']:+5.0f}   {a:+8.3f}{tie[:1] and '':s}   "
                    f"{r.get('SI_CN_per_g', float('nan')):+10.4f}      "
                    f"{r.get('Ay_apex_g', float('nan')):+7.3f}   "
                    f"{r.get('limit_character', '?'):24s} "
                    f"{r.get('N_delta', float('nan')):+8.0f}")
            rows.append('')
            rows.append(' How to judge (RCVD): trim limit bigger is better '
                        'ONLY beyond the 0.006 g noise band; SI must be '
                        'negative (zero = divergence boundary, p205); '
                        'apex character should be NEUTRAL or mild PLOW '
                        '(race cars are tuned neutral at the limit, p320); '
                        'control at trim is ~zero BY DEFINITION at max trim '
                        '(p321) — judge it before the limit, not at it.')
            _sp = (best - _np.nanmin([r.get('Ay_trim_max', _np.nan)
                                      for r in trim]))
            rows.append(f' Trim spread this sweep: {_sp:.3f} g vs 0.006 g '
                        'noise -> ' + ('settings SEPARATE'
                                       if _sp > 0.012 else 'TIE on grip'))
            txt = QPlainTextEdit(chr(10).join(rows))
            txt.setReadOnly(True)
            txt.setMaximumHeight(230)
            txt.setStyleSheet('font-family: Consolas, monospace; '
                              'font-size: 12px;')
            lay.addWidget(txt)
            lay.addWidget(FigureCanvas(fig))
            dlg.show()
        except Exception as e:
            import traceback; traceback.print_exc()
            mw.statusBar().showMessage(f'YMD grid: {e}', 8000)
        finally:
            self._ymd_btn.setEnabled(True)
            self._ymd_btn.setText('YMD GRID  +100% → −70%')

    def _show_mmd_help(self):
        self._main._show_text_popup(
            'How to read the MMD',
            'Each diagram is every steady cornering state the car can hold at this speed.\n'
            '\n'
            'AXES\n'
            '  Across: cornering force, in g.  Farther out = cornering harder.\n'
            '  Up/down: yaw moment.  Above zero the car is rotating INTO the turn; below zero it runs wide.\n'
            '\n'
            'LINES\n'
            '  Solid: body slip held, steering swept.\n'
            '  Dashed: steering held, body slip swept.\n'
            '\n'
            'THE LINE THAT MATTERS\n'
            '  Yaw moment = 0 (the horizontal axis) is a balanced corner - the car holds its arc.  Only these states are drivable steady-state.\n'
            '  The farthest point where any line crosses zero = the cornering limit, driven clean.\n'
            '\n'
            'COMPARING TABS (Ackermann settings)\n'
            '  1. Does the zero crossing move outward?  That setting corners harder.  (Here it barely moves - grip ties.)\n'
            '  2. Near the crossing, do the solid lines still slope?  Sloped = adding steering still turns the car at the limit.  Flat = steering has stopped working.\n'
            '\n'
            'For the numbers behind point 2, use the YMD GRID button: trimmed g, control and stability per setting.')

    def _on_save(self):
        out = os.path.join(_repo_root(), 'figs')
        os.makedirs(out, exist_ok=True)
        names = {'demand': 'ackpage_1_toe_demand',
                 'pct': 'ackpage_2_percent_health',
                 'fzfy': 'ackpage_3_fz_fy',
                 'pair': 'ackpage_4_pair_analysis',
                 'lap': 'ackpage_5_laptime'}
        for key, base in names.items():
            self._slots[key].save(os.path.join(out, base + '.png'))
        self._status.setText(f'Five figures written to {out}')

    def _style(self, key):
        """Colour + line style for a radius, matched to the fixed table so
        graph 1 and graph 2 always agree on which curve is which."""
        for R, c, ls in _RADII:
            if abs(R - key) < 1e-6:
                return c, ls
        i = int(abs(key)) % len(_SWEEP_STYLES)
        return _SWEEP_STYLES[i]

    # ═══════════════════════════════════════════════════════════════════
    #  graph 1 — per-corner toe demand, in DEGREES
    # ═══════════════════════════════════════════════════════════════════
    def _draw_demand(self, buckets):
        s = self._slots['demand']
        ax = s.axes()
        gm = self._cfg_used['grip']
        cross, drawn, sat_g = {}, 0, []
        g_all = np.array([r['lat_g'] for r in buckets[sorted(buckets)[0]]],
                         float)
        for i, R in enumerate(sorted(buckets)):
            rows = buckets[R]
            col, ls = self._style(R)
            g = np.array([r['lat_g'] for r in rows], float)
            # A saturated wheel's slip angle is the PEAK-ANGLE FALLBACK, so
            # any toe difference built from it is meaningless — and when BOTH
            # fronts saturate they return the SAME fallback and the difference
            # collapses to the bare tangent difference, i.e. exactly 100%
            # Ackermann wearing a result's clothes.  Those rows are BROKEN
            # (NaN), never drawn.  That is what used to make this look erratic.
            d = np.array([(r['point_spread_deg'] if r.get('valid', True)
                           else np.nan) for r in rows], float)
            if (~np.isfinite(d)).any() and np.isfinite(d).any():
                sat_g.append(float(g[~np.isfinite(d)][0]))
            xg = _zero_crossing(g, d)
            # The crossover g goes in the LEGEND LABEL, not in a floating text
            # box: four crossings within 0.3 g of each other guarantee the
            # boxes land on top of the neighbouring curves.
            lab = f'{R:.0f} m corner'
            if xg is not None:
                cross[R] = xg
                lab += f' — turns REVERSE at {xg:.2f} g'
            elif np.isfinite(d).any():
                lab += (' — stays PRO' if float(d[np.isfinite(d)][-1]) > 0
                        else ' — stays REVERSE')
            ax.plot(g, d, color=col, ls=ls, lw=1.9, label=lab)
            if xg is not None:
                ax.plot([xg], [0.0], 'o', color=col, ms=6, zorder=6)
            drawn += int(np.isfinite(d).any())
        ax.axhline(0.0, color=_WGY, lw=1.0)
        ax.set_xlim(float(g_all[0]) - 0.02, float(g_all[-1]) + 0.02)
        ax.set_xlabel('Lateral acceleration (g, on asphalt)', fontsize=9)
        # The PRO / REVERSE meaning lives on the axis label: any in-plot text
        # box here collides with the 3 m curve, which starts at +9 deg.
        ax.set_ylabel('Toe difference the corner needs (degrees)\n'
                      'plus = PRO (inner wheel turned MORE)  ·  '
                      'minus = REVERSE (turned LESS)', fontsize=8.5)
        if sat_g:
            gs = min(sat_g)
            ax.axvspan(gs, float(g_all[-1]) + 0.02, color=_RED, alpha=0.11,
                       zorder=0)
            ax.text(0.5 * (gs + float(g_all[-1])), 0.0,
                    'no answer\npast here —\na front wheel\nis out of grip',
                    fontsize=7, color=_RED, va='center', ha='center')
        ax.set_title(f'grip ×{gm:.2f} on asphalt · dot = where the demand '
                     f'crosses from pro to reverse', fontsize=8.5)
        s.legend(ax, ncol=2, fontsize=7, loc='lower left')
        s.finish()

        if not drawn:
            s.summary('No valid rows: every front wheel saturates over this '
                      'whole g range at every radius. Lower the g range or '
                      'raise the grip multiplier.')
            return
        bits = []
        for R in sorted(buckets):
            rows = buckets[R]
            d = [r['point_spread_deg'] for r in rows if r.get('valid', True)]
            if not d:
                bits.append(f'{R:.0f} m: no valid rows (wheel saturated)')
                continue
            if R in cross:
                bits.append(f'{R:.0f} m crosses from pro to reverse at '
                            f'{cross[R]:.2f} g')
            else:
                sign = 'pro' if d[-1] > 0 else 'reverse'
                bits.append(f'{R:.0f} m stays {sign} '
                            f'({d[0]:+.2f} to {d[-1]:+.2f} deg)')
        s.summary('WHAT IT MEANS: ' + '; '.join(bits)
                  + '.  Build the steering arm to DEGREES.  One linkage gives '
                    'a toe spread proportional to steer angle SQUARED with a '
                    'single fixed constant, so no one linkage can match every '
                    'radius at once — pick the radii this car actually drives.')

    # ═══════════════════════════════════════════════════════════════════
    #  graph 2 — the percentage, with its own health warning
    # ═══════════════════════════════════════════════════════════════════
    def _draw_pct(self, buckets):
        s = self._slots['pct']
        # TWO panels, not a twin axis.  A fifth curve sharing the left panel's
        # axis is a fifth visual identity this palette does not have (and
        # amber, the obvious spare, is a colour this user cannot separate from
        # yellow).  Left = the percentage.  Right = the DIVIDER that makes it
        # explode.  Same x, same shading, so the eye links them.
        ax, ax2 = s.axes(2)
        R_sel = self._cfg_used['radius']
        R_shade = min(sorted(buckets), key=lambda r: abs(r - R_sel))

        # HARD CLIP the percentage axis.  Unclipped, the 12 m curve reaches
        # about -1280% and squashes every other curve onto the zero line — the
        # explosion is the message, but a flat-lined plot is not a graph.  The
        # curves visibly plunge off the bottom of the frame instead, and the
        # off-scale minimum goes in the legend label.
        Y_LO, Y_HI = -250.0, 200.0
        g_all, bad_x = None, None
        for R in sorted(buckets):
            rows = buckets[R]
            col, ls = self._style(R)
            g = np.array([r['lat_g'] for r in rows], float)
            g_all = g
            pct = np.array([(r['ackermann_pct'] if r.get('valid', True)
                             else np.nan) for r in rows], float)
            den = np.array([r['travel_spread_deg'] for r in rows], float)
            lab = f'{R:.0f} m'
            if np.isfinite(pct).any() and float(np.nanmin(pct)) < Y_LO:
                lab += f'  (dives to {float(np.nanmin(pct)):.0f}%, off scale)'
            ax.plot(g, pct, color=col, ls=ls, lw=1.9, label=lab)
            ax2.plot(g, den, color=col, ls=ls, lw=1.9, label=f'{R:.0f} m')
            b = den < _DENOM_FLOOR_DEG
            if abs(R - R_shade) < 1e-9 and b.any():
                bad_x = float(g[b][0])

        for a in (ax, ax2):
            if bad_x is not None:
                a.axvspan(bad_x, float(g_all[-1]), color=_RED, alpha=0.13,
                          zorder=0)
            a.set_xlim(float(g_all[0]) - 0.02, float(g_all[-1]) + 0.02)
            a.set_xlabel('Lateral acceleration (g, on asphalt)', fontsize=8.5)
        ax.axhline(0.0, color=_WGY, lw=0.9)
        ax.axhline(100.0, color=_WGY, lw=0.8, ls='--', alpha=0.7)
        ax.set_ylim(Y_LO, Y_HI)
        ax.set_ylabel('Ackermann (%)', fontsize=8.5)
        ax.set_title('THE PERCENTAGE   (axis clipped at ±250%)', fontsize=8.5)
        ax2.axhline(_DENOM_FLOOR_DEG, color=_RED, lw=1.2, ls=':')
        # LOG y: the spread runs 0.1 to 9 degrees across these radii, and on a
        # linear axis the 3 m curve flattens the 1-degree floor line and the
        # 8/12 m collapse — the only part of this panel worth reading.
        ax2.set_yscale('log')
        # Plain decimal tick labels — a bare log axis here shows only 10^0 and
        # 10^1, and the reader needs to see 0.2 / 0.5 / 2 / 5 degrees.
        from matplotlib.ticker import FixedLocator, FuncFormatter
        ax2.yaxis.set_major_locator(
            FixedLocator([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]))
        ax2.yaxis.set_minor_locator(FixedLocator([]))
        ax2.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _p: (f'{v:g}')))
        ax2.set_ylabel('Kinematic travel spread (degrees, log)', fontsize=8.5)
        ax2.set_title(f'THE DIVIDER   (red dotted = the '
                      f'{_DENOM_FLOOR_DEG:.0f} deg floor)', fontsize=8.5)
        if bad_x is not None:
            # ONE warning box, on the percentage panel.  The x-axis and the
            # shading are identical on both, so a second copy only fights the
            # right panel's legend for space.
            ax.text(0.5 * (bad_x + float(g_all[-1])),
                    Y_HI - 0.04 * (Y_HI - Y_LO),
                    f'PERCENTAGE\nUNUSABLE\n(divider under '
                    f'{_DENOM_FLOOR_DEG:.0f} deg\nat {R_shade:.0f} m)',
                    ha='center', va='top', fontsize=8, color=_RED,
                    fontweight='bold')
        s.legend(ax, ncol=1, fontsize=6.5, loc='lower left')
        s.legend(ax2, ncol=2, fontsize=7, loc='lower left')
        s.finish()

        # summary: the measured collapse, in words
        rr = buckets[R_shade]
        val = [r for r in rr if r.get('valid', True)]
        if len(val) >= 2:
            a, b = val[0], val[-1]
            s.summary(
                f'WHAT IT MEANS: at {R_shade:.0f} m the real answer (the toe '
                f'difference) goes {a["point_spread_deg"]:+.3f} to '
                f'{b["point_spread_deg"]:+.3f} deg between '
                f'{a["lat_g"]:.2f} and {b["lat_g"]:.2f} g, while the divider '
                f'shrinks {a["travel_spread_deg"]:.3f} to '
                f'{b["travel_spread_deg"]:.3f} deg — so the percentage reads '
                f'{a["ackermann_pct"]:+.0f}% then {b["ackermann_pct"]:+.0f}%. '
                f'Nothing about the car ran away; the divider did. Spec '
                f'degrees.')
        else:
            s.summary('WHAT IT MEANS: too few valid rows at '
                      f'{R_shade:.0f} m to show the collapse — widen the g '
                      'range or raise the grip multiplier.')

    # ═══════════════════════════════════════════════════════════════════
    #  graph 3 — Fz–Fy capability vs delivered
    # ═══════════════════════════════════════════════════════════════════
    def _draw_fzfy(self, d):
        s = self._slots['fzfy']
        ax = s.axes()
        self._fzfy_proxy = None          # never carry last run's marker over
        tire = getattr(self._main, '_tire_model', None)
        gm = d['grip_multiplier']
        fz_o = d['fz_outer']; fz_i = d['fz_inner']
        n = d['n_delivered']

        # iso-slip background, in dim warm grey (context, never an accent).
        # Label dodge: near the top the iso-lines converge on a saturated
        # tyre and stacked labels overprint, so a label is skipped if it
        # would land within 4% of the previous one's height.
        if tire is not None and len(fz_o):
            fz_hi = max(1500.0, float(np.nanmax(fz_o)) * 1.10)
            fzr = np.linspace(50.0, fz_hi, 60)
            placed = []
            for sa in (2, 4, 6, 8, 10, 12):
                line = np.array([abs(float(tire.Fy(sa, f, 0.0)))
                                 for f in fzr]) * gm
                ax.plot(fzr, line, color=_DIM, lw=0.7, alpha=0.35)
                y_end = float(line[-1])
                span = max(float(np.nanmax(line)), 1.0)
                if all(abs(y_end - p) > 0.04 * span for p in placed):
                    ax.text(fzr[-1], y_end, f' {sa}° slip', fontsize=6.5,
                            color=_DIM, va='center')
                    placed.append(y_end)

        # COLOUR says which wheel, LINE STYLE says capability or delivered.
        # (Four different colours here put yellow next to amber, which this
        # user cannot separate.)
        ax.plot(fz_o, d['cap_outer'], color=_WHT, lw=2.4, ls='-',
                label='OUTER — most it can make')
        ax.plot(fz_i, d['cap_inner'], color=_YEL, lw=2.4, ls='-',
                label='INNER — most it can make')
        if n > 0:
            ax.plot(fz_o[:n], d['del_outer'], color=_WHT, lw=1.7, ls='--',
                    label=f'OUTER delivered at {d["ackermann_pct"]:+.0f}%')
            ax.plot(fz_i[:n], d['del_inner'], color=_YEL, lw=1.7, ls='--',
                    label=f'INNER delivered at {d["ackermann_pct"]:+.0f}%')
            if d['g_limit'] is not None:
                for x, y, c in ((fz_o[n - 1], d['del_outer'][-1], _RED),
                                (fz_i[n - 1], d['del_inner'][-1], _RED)):
                    ax.plot([x], [y], marker='x', color=c, ms=10, mew=2.2,
                            zorder=6)
                # The explanation goes in the LEGEND, not next to the marker:
                # at the limit the delivered path rejoins the capability curve,
                # so any text pinned there lands on top of it.
                from matplotlib.lines import Line2D
                self._fzfy_proxy = Line2D(
                    [], [], color=_RED, marker='x', ls='none', ms=9, mew=2.2,
                    label=f'✕ grip limit {d["g_limit"]:.2f} g — past here no '
                          f'steer angle reaches the demand')
        # g tags so the reader knows the curves are PATHS over lateral g
        g = d['lat_g']
        xr = max(float(np.diff(ax.get_xlim())[0]), 1.0)
        yr = max(float(np.diff(ax.get_ylim())[0]), 1.0)
        # Where the grip-limit crosses sit, so a g tag never lands on one.
        xmarks = []
        if n > 0 and d['g_limit'] is not None:
            xmarks = [(float(fz_o[n - 1]), float(d['del_outer'][-1])),
                      (float(fz_i[n - 1]), float(d['del_inner'][-1]))]
        for gm_tag in (0.5, 1.0, 1.5):
            if not len(g) or gm_tag > float(g[-1]):
                continue
            # Both capability curves rise left-to-right, so a tag placed
            # below-LEFT or above-RIGHT lands back on the line.  Outer tags go
            # below-RIGHT, inner tags above-LEFT — the empty side in each case.
            for xa, ya, col, dx, dy, va in (
                    (fz_o, d['cap_outer'], _WHT, 8, -7, 'top'),
                    (fz_i, d['cap_inner'], _YEL, -8, 7, 'bottom')):
                k = int(np.argmin(np.abs(g - gm_tag)))
                px, py = float(xa[k]), float(ya[k])
                ax.plot([px], [py], 'o', color=col, ms=4, zorder=5)
                if any(abs(px - mx) / xr < 0.03 and abs(py - my) / yr < 0.04
                       for mx, my in xmarks):
                    continue          # the ✕ already carries a g label there
                ax.annotate(f'{gm_tag:.1f} g', xy=(px, py),
                            xytext=(dx, dy), textcoords='offset points',
                            ha='left' if dx > 0 else 'right', va=va,
                            fontsize=6.5, color=col)
        ax.axhline(0.0, color=_WGY, lw=0.8)
        ax.set_xlabel('Vertical load on that wheel (N) — from the solved car',
                      fontsize=9)
        ax.set_ylabel('Side force toward the corner (N, signed)', fontsize=9)
        ax.set_title(f'R = {d["radius_m"]:.1f} m, grip ×{gm:.2f} · solid = '
                     f'CAPABILITY, dashed = DELIVERED · below zero fights '
                     f'the turn', fontsize=8.5)
        h, l = ax.get_legend_handles_labels()
        proxy = getattr(self, '_fzfy_proxy', None)
        if proxy is not None and d['g_limit'] is not None:
            h.append(proxy); l.append(proxy.get_label())
        ax.legend(h, l, facecolor='#0e0d10', labelcolor=_TEXT,
                  edgecolor='#2a2823', framealpha=0.9, fontsize=7,
                  loc='upper left')
        s.finish()

        if n > 0:
            hi = min(n, len(g)) - 1
            head_o = 100.0 * float(d['del_outer'][hi]) / max(
                float(d['cap_outer'][hi]), 1e-9)
            head_i = 100.0 * float(d['del_inner'][hi]) / max(
                float(d['cap_inner'][hi]), 1e-9)
            lim = (f'the axle runs out of grip at {d["g_limit"]:.2f} g'
                   if d['g_limit'] is not None else
                   f'the axle still holds at {float(g[-1]):.2f} g, the top of '
                   f'the sweep')
            s.summary(
                f'WHAT IT MEANS: at {float(g[hi]):.2f} g the outer wheel is '
                f'using {head_o:.0f}% of what it can make and the inner '
                f'{head_i:.0f}% — the outer is the one that runs out, and '
                f'{lim}. The gap between a solid line and its dashed partner '
                f'is grip left on the table at that load.')
        else:
            s.summary('WHAT IT MEANS: no steer angle reaches the axle demand '
                      'anywhere in this g range — the corner is impossible '
                      'at this radius on this surface.')

    # ═══════════════════════════════════════════════════════════════════
    #  graph 4 — pair analysis
    # ═══════════════════════════════════════════════════════════════════
    def _draw_pair(self, d):
        s = self._slots['pair']
        # TWO panels, because ONE was a lie of omission: on the absolute axis
        # all nine settings draw the same curve to within a couple of newtons
        # out of 3000, so the graph looked empty of information.  Left panel =
        # the RCVD Fig 7.2 format (shape, and where the best steer angle is);
        # right panel = the DIFFERENCE from the car as built, which is the
        # only place the family actually separates.
        ax, ax2 = s.axes(2)
        asb = self._cfg_used['as_built_pct']
        steer = d['steer_deg']
        curves = sorted(d['curves'].items())
        ref_key = min((k for k, _v in curves), key=lambda k: abs(k - asb))
        ref = np.asarray(d['curves'][ref_key]['useful_N'], float)
        peaks = {}
        for i, (ack, c) in enumerate(curves):
            col, ls = _SWEEP_STYLES[i % len(_SWEEP_STYLES)]
            mine = abs(ack - ref_key) < 1e-9
            lw = 2.8 if mine else 1.5
            lab = f'{ack:+.0f}%' + ('  ← this car' if mine else '')
            u = np.asarray(c['useful_N'], float)
            ax.plot(steer, u, color=col, ls=ls, lw=lw, label=lab)
            ax.plot([c['peak_steer_deg']], [c['peak_N']], 'o', color=col,
                    ms=5, zorder=6)
            ax2.plot(steer, u - ref, color=col, ls=ls, lw=lw, label=lab)
            peaks[float(ack)] = (c['peak_N'], c['peak_steer_deg'],
                                 c['scrub_at_peak_N'])
        ax.axhline(0.0, color=_WGY, lw=0.8)
        ax2.axhline(0.0, color=_WGY, lw=1.0)
        ax.set_xlabel('Steer angle (deg) = mean of both wheels', fontsize=8.5)
        ax.set_ylabel('Axle side force toward the corner (N)', fontsize=8.5)
        ax.set_title(f'at {d["lat_g"]:.2f} g · inner {d["fz_in"]:.0f} N,'
                     f' outer {d["fz_out"]:.0f} N\n'
                     f'{100 * d["flt"]:.0f}% of the axle load outboard',
                     fontsize=8)
        ax2.set_xlabel('Steer angle (deg)', fontsize=8.5)
        ax2.set_ylabel(f'Difference from this car ({ref_key:+.0f}%), in N',
                       fontsize=8.5)
        ax2.set_title('THE SAME CURVES, MINUS EACH OTHER\n'
                      'above 0 = more force than this car', fontsize=8)
        # ONE legend, on the left panel's empty lower-right corner.  Both
        # panels carry the identical nine series, and a second copy only
        # covered the -100% curve's dive on the right.
        s.legend(ax, ncol=3, fontsize=6.5, loc='lower right')
        s.finish()

        if peaks:
            best = max(peaks, key=lambda k: peaks[k][0])
            worst = min(peaks, key=lambda k: peaks[k][0])
            pk_b, st_b, _sc = peaks[best]
            pk_w = peaks[worst][0]
            spread = 100.0 * (pk_b - pk_w) / max(abs(pk_b), 1e-9)
            verdict = ('a spread that small is one curve — on this criterion '
                       'the choice decides nothing'
                       if spread < 1.0 else
                       'that is a real separation, worth acting on')
            s.summary(
                f'WHAT IT MEANS: the best setting here is {best:+.0f}% at '
                f'{pk_b:.0f} N of axle force (peaking at {st_b:.1f} deg of '
                f'steer); the worst is {worst:+.0f}% at {pk_w:.0f} N — '
                f'{pk_b - pk_w:.0f} N, or {spread:.2f}%, across the whole '
                f'family. This car ({ref_key:+.0f}%) makes '
                f'{peaks[ref_key][0]:.0f} N, '
                f'{peaks[ref_key][0] - pk_b:+.0f} N vs the best. {verdict}.')

    # ═══════════════════════════════════════════════════════════════════
    #  graph 5 — lap time vs Ackermann, with the noise band
    # ═══════════════════════════════════════════════════════════════════
    def _draw_lap(self, payload):
        s = self._slots['lap']
        if payload is None:
            s.busy('lap-time chain not run\n(tick the box in the sidebar)')
            s.summary('WHAT IT MEANS: nothing yet — the chain was skipped.')
            return
        tracks = payload.get('tracks') or {}
        if not tracks:
            s.busy('no lap-time result\n' + str(payload.get('why') or ''))
            s.summary('WHAT IT MEANS: the per-station capability table did '
                      'not solve, so no lap could be run against it.')
            return
        # TWO panels.  Left = THE ANSWER (mean lap time across every track,
        # with each track drawn faintly behind it, and the method's own noise
        # band).  Right = WHERE the effect comes from: the front-axle force
        # spread against corner radius, which is the whole reason a single
        # global grip scale could never carry this.
        ax, ax2 = s.axes(2)
        asb = self._cfg_used['as_built_pct']
        labels = sorted(tracks)
        pct = np.asarray(tracks[labels[0]]['pct'], float)

        # Per-track curves, as a DELTA against each track's own best setting
        # (absolute lap times sit seconds apart, and at that scale the whole
        # effect is an invisible line).  Faint, thin, LINE STYLE varied — they
        # are context for the mean, not four competing series.
        per = []
        faint = ['-', '--', '-.', ':']
        for i, label in enumerate(labels):
            t = tracks[label]
            lap = np.asarray(t['lap_s'], float)
            d = (lap - np.nanmin(lap)) * 1000.0
            per.append(d)
            ax.plot(pct, d, color=_DIM, ls=faint[i % len(faint)], lw=1.2,
                    alpha=0.85,
                    label=f'{label} alone ({t["control_lap_s"]:.2f} s '
                          f'with Ackermann off)')
        per = np.asarray(per, float)
        mean = np.nanmean(per, axis=0)
        band = float(np.mean([tracks[l]['band_s'] for l in labels])) * 1000.0
        ax.fill_between(pct, mean - band, mean + band, color=_YEL, alpha=0.16,
                        lw=0, label=f'method noise ±{band:.0f} ms')
        ax.plot(pct, mean, color=_YEL, ls='-', lw=2.6, marker='o', ms=5,
                label='MEAN of every track')
        k_best = int(np.argmin(mean))
        ax.plot([pct[k_best]], [mean[k_best]], 'o', color=_RED, ms=10,
                mfc='none', mew=2.0, zorder=7)
        ax.axvline(asb, color=_WGY, lw=0.9, ls=':')
        ax.set_xlabel('Ackermann setting (%)  —  negative = reverse',
                      fontsize=8.5)
        ax.set_ylabel('Lap time above the best setting (milliseconds)',
                      fontsize=8.5)
        ax.set_title('MEAN LAP TIME vs ACKERMANN\n'
                     'per-station: each corner gets the capability its OWN '
                     'radius earns', fontsize=8.5)

        # ── the verdict: is the mean spread outside the method's noise? ────
        spread_ms = float(np.nanmax(mean) - np.nanmin(mean))
        tie = spread_ms <= 2.0 * band
        if tie:
            msg = ('THE SETTINGS TIE\nthe whole mean spread '
                   f'({spread_ms:.0f} ms) is inside this method’s own\n'
                   f'noise (±{band:.0f} ms) — this plot picks NO winner')
        else:
            sep = [f'{pct[j]:+.0f}%' for j in range(len(pct))
                   if mean[j] - mean[k_best] > 2.0 * band]
            msg = (f'BEST: {pct[k_best]:+.0f}%   '
                   f'(mean spread {spread_ms:.0f} ms, noise ±{band:.0f} ms)\n'
                   f'{len(sep)} of {len(pct)} settings are a real loss '
                   f'against it')
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0, y1 + 0.34 * (y1 - y0))
        ax.text(0.5, 0.98, msg, transform=ax.transAxes, ha='center', va='top',
                fontsize=8, color=_RED if tie else _YEL, fontweight='bold',
                linespacing=1.4,
                bbox=dict(facecolor='#1a1210', edgecolor=_RED if tie else _YEL,
                          lw=1.0, boxstyle='round,pad=0.4'))
        # "this car" label pinned in blended coords at mid-height: the curves
        # all run high-left to low-right, so the middle of the right half is
        # the one region that is empty whatever the numbers do — and it is not
        # the lower-right corner, where the legend lives.
        from matplotlib.transforms import blended_transform_factory as _btf
        ax.text(asb, 0.55, f' this car {asb:+.0f}%',
                transform=_btf(ax.transData, ax.transAxes),
                ha='left', va='center', fontsize=7.5, color=_WGY)
        s.legend(ax, loc='lower right', fontsize=6.5)

        # ── right panel: WHY it is radius-dependent ───────────────────────
        sp = payload.get('spread') or []
        if sp:
            R = np.array([r['radius_m'] for r in sp], float)
            fc = np.array([r['ceil_spread_pct'] for r in sp], float)
            sc = np.array([r['scrub_spread_N'] for r in sp], float)
            ax2.plot(R, fc, color=_YEL, ls='-', lw=2.2, marker='o', ms=4,
                     label='front-axle force spread (% of the axle force)')
            axb = ax2.twinx()
            axb.plot(R, sc, color=_WHT, ls='--', lw=1.8, marker='s', ms=4,
                     label='scrub drag spread (N)')
            axb.set_ylabel('Scrub drag spread across the family (N)',
                           fontsize=8, color=_WHT)
            axb.tick_params(colors=_WHT, labelsize=7)
            ax2.set_xscale('log')
            from matplotlib.ticker import FixedLocator, FuncFormatter
            ax2.xaxis.set_major_locator(FixedLocator(list(R)))
            ax2.xaxis.set_minor_locator(FixedLocator([]))
            ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f'{v:g}'))
            ax2.set_xlabel('Corner radius (m, log)', fontsize=8.5)
            ax2.set_ylabel('Front-axle force spread, -100% to +100% (%)',
                           fontsize=8.5, color=_YEL)
            ax2.tick_params(axis='y', colors=_YEL)
            ax2.set_title('WHY ONE GLOBAL GRIP SCALE CANNOT CARRY THIS\n'
                          'the whole effect lives under about 8 m of radius',
                          fontsize=8.5)
            h1, l1 = ax2.get_legend_handles_labels()
            h2, l2 = axb.get_legend_handles_labels()
            ax2.legend(h1 + h2, l1 + l2, facecolor='#0e0d10',
                       labelcolor=_TEXT, edgecolor='#2a2823', framealpha=0.9,
                       fontsize=7, loc='upper right')
        s.finish()

        # ── the words underneath ──────────────────────────────────────────
        bits = []
        for label in labels:
            t = tracks[label]
            lap = np.asarray(t['lap_s'], float)
            bits.append(
                f'{label}: {np.nanmin(lap):.3f}–{np.nanmax(lap):.3f} s '
                f'(spread {1000 * (np.nanmax(lap) - np.nanmin(lap)):.0f} ms, '
                f'best {pct[int(np.argmin(lap))]:+.0f}%, '
                f'scrub costs it {np.mean(t["scrub_mean_N"]):.0f} N on average)')
        err_g = float(payload.get('err_g', 0.0))
        err_sc = float(payload.get('err_scrub_frac', 0.0))
        head = ('WHAT IT MEANS: THE SETTINGS TIE — '
                if tie else
                f'WHAT IT MEANS: {pct[k_best]:+.0f}% is the fastest on the '
                f'average of every track — ')
        s.summary(
            head
            + f'mean spread {spread_ms:.0f} ms against a measured noise band '
              f'of ±{band:.0f} ms.  The band is not asserted and not '
              f'converted: this table’s own binning error ({err_g:.4f} g of '
              f'front-axle cap and {100 * err_sc:.1f}% of scrub, from '
              f're-solving at radii deliberately between bins) was applied to '
              f'the table and the lap RE-RUN, so the band is seconds the sim '
              f'itself produced.  '
            + '  '.join(bits)
            + '.  Each station is capped by what the front axle can hold at '
              'ITS radius, so tight corners and fast sweepers are treated '
              'differently by the same setting — which is the trade you asked '
              'about.')
