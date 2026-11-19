"""
gui/lap_loads_page.py — LOADS AROUND THE LAP (user 2026-10-21: "I want loads
swept across lap sim.  So I run a lap sim and I want to see loads experienced
by car throughout track.")

ONE MODEL: every number here comes from gui.wheel_package.compute_case — THE
load path the Loads page, the Bearings page and the report use — called at each
lap station's lateral g, longitudinal g and speed.  Nothing is re-modelled.

  * Turn direction comes from the track's signed curvature: kappa > 0 is a LEFT
    turn (heading rises), so the right-hand corners are on the outside.  The
    steady-state solver's positive lateral g loads the LEFT side (a right turn),
    so the case is solved at  lat = -sign(kappa) * |lap lat g|.
  * The case speed is the lap's speed at that station, so the aero load is the
    project aero package (Dynamics panel, Apply Aero) at that speed.  The lap
    sim has its own Cl·A box; the page says so when the two disagree.
  * compute_case reads the panels and the shared corner solvers, so it runs on
    the GUI thread in small time-sliced batches (the window stays live, with a
    progress readout and a Stop button) rather than on a worker thread.
  * Every station can be solved, or every N metres plus the stations of peak
    lateral, braking and drive g; between solved stations the curves are
    straight-line interpolated and the page says how many were solved.
"""
from __future__ import annotations

import time

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QListWidget, QListWidgetItem, QSplitter, QTableWidget, QTableWidgetItem,
    QDialog, QHeaderView, QAbstractItemView, QSizePolicy,
)
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

from gui.plot_dialog import ReadableCanvas, _ExportCanvas

CORNERS = ('FL', 'FR', 'RL', 'RR')
CORNER_NAMES = {'FL': 'front left', 'FR': 'front right', 'RL': 'rear left', 'RR': 'rear right'}


def _vnorm(v):
    try:
        a = np.asarray(v, float)
        return float(np.linalg.norm(a)) if a.size == 3 and np.all(np.isfinite(a)) else float('nan')
    except Exception:
        return float('nan')


# (key, label, sign note, value fn(ComponentLoads)).  Labels are the words the
# Loads 3-D view uses (gui.wheel_package._load_items).  All in newtons.
MEMBERS = [
    ('uca_front', 'upper arm front', '+ pull / - push', lambda c: float(c.uca_front_N)),
    ('uca_rear', 'upper arm rear', '+ pull / - push', lambda c: float(c.uca_rear_N)),
    ('lca_front', 'lower arm front', '+ pull / - push', lambda c: float(c.lca_front_N)),
    ('lca_rear', 'lower arm rear', '+ pull / - push', lambda c: float(c.lca_rear_N)),
    ('tierod', 'tie / toe rod', '+ pull / - push', lambda c: float(c.tierod_N)),
    ('pushrod', 'pushrod', '+ pull / - push', lambda c: float(c.pushrod_N)),
    ('spring', 'spring', '+ squeezed', lambda c: float(c.spring_force_N)),
    ('arb_link', 'ARB drop-link', '+ pull / - push', lambda c: float(c.arb_link_N)),
    ('uca_bj', 'upper ball joint', 'size of the force', lambda c: _vnorm((c.joint_forces or {}).get('uca'))),
    ('lca_bj', 'lower ball joint', 'size of the force', lambda c: _vnorm((c.joint_forces or {}).get('lca'))),
    ('tie_bj', 'tie / toe ball joint', 'size of the force', lambda c: _vnorm((c.joint_forces or {}).get('tie'))),
    ('brg_out', 'outer bearing radial', 'size of the force', lambda c: float(np.hypot(c.bearing_outer_V, c.bearing_outer_H))),
    ('brg_in', 'inner bearing radial', 'size of the force', lambda c: float(np.hypot(c.bearing_inner_V, c.bearing_inner_H))),
    ('brg_ax', 'outer bearing axial', '+ inboard', lambda c: float(c.bearing_axial_N)),
    ('tyre_z', 'tyre vertical load', '+ up', lambda c: float(c.Fz_N)),
    ('tyre_y', 'tyre sideways force', '+ toward car left', lambda c: float(c.Fy_N)),
    ('tyre_x', 'tyre fore-aft force', '+ drive / - braking', lambda c: float(c.Fx_N)),
]
MEMBER_LABEL = {k: l for k, l, _s, _f in MEMBERS}
MEMBER_SIGN = {k: s for k, _l, s, _f in MEMBERS}
# members whose numbers depend on the member solve being valid
_SOLVE_DEPENDENT = {'uca_front', 'uca_rear', 'lca_front', 'lca_rear', 'tierod', 'pushrod',
                    'arb_link', 'uca_bj', 'lca_bj', 'tie_bj'}


def member_values(loads: dict) -> dict:
    """{(corner, member key): value N} from one compute_case loads dict."""
    out = {}
    for lbl in CORNERS:
        c = loads.get(lbl)
        for key, _l, _s, fn in MEMBERS:
            v = float('nan')
            if c is not None and (getattr(c, 'valid', True) or key not in _SOLVE_DEPENDENT):
                try:
                    v = float(fn(c))
                except Exception:
                    v = float('nan')
            out[(lbl, key)] = v
    return out


def lap_stations(res, track) -> dict:
    """Per-station case inputs of a LapResult: s (m), lap lat g signed
    (+ = left turn, for display), the SOLVER lateral g (+ loads the left side),
    longitudinal g, speed (km/h)."""
    s = np.asarray(res.s_m, float)
    lat = np.abs(np.asarray(res.lat_g, float))
    kap = np.asarray(track.kappa_1pm, float)
    if len(kap) != len(s):
        kap = np.interp(s, np.linspace(s[0], s[-1], len(kap)), kap)
    turn = np.sign(kap)                       # +1 left turn, -1 right turn
    x, y = np.asarray(track.x_m, float), np.asarray(track.y_m, float)
    if len(x) != len(s):
        ts = track.s_m
        x, y = np.interp(s, ts, x), np.interp(s, ts, y)
    return dict(s=s, lat_display=turn * lat, lat_solver=-turn * lat,
                # the TYRE fore-aft g when the lap has it (method 'ggv': drag
                # taken out — the solve's longitudinal_g is tyre force), else
                # the lap's net g (old method)
                lon=np.clip(np.asarray(res.lon_tyre_g if getattr(res, 'lon_tyre_g', None) is not None
                                       else res.lon_g, float), -3.0, 3.0),
                v_kph=np.asarray(res.v_ms, float) * 3.6, x=x, y=y)


def solve_indices(st: dict, every_m: float) -> np.ndarray:
    """Stations to solve: every station when every_m <= 0, else every
    `every_m` metres plus the first/last station and the stations of peak
    lateral, braking and drive g (so decimation never hides a g peak)."""
    s = st['s']; n = len(s)
    if every_m <= 0.0 or n < 3:
        return np.arange(n)
    grid = np.arange(s[0], s[-1] + 1e-9, float(every_m))
    idx = set(int(i) for i in np.searchsorted(s, grid).clip(0, n - 1))
    idx |= {0, n - 1, int(np.argmax(np.abs(st['lat_solver']))),
            int(np.argmin(st['lon'])), int(np.argmax(st['lon']))}
    return np.array(sorted(idx), int)


class LapLoadSweep:
    """The loads at the lap's stations, via compute_case (ONE MODEL).
    step(budget_s) solves until the time budget is used; done when True."""

    def __init__(self, win, res, track, every_m: float = 0.0, indices=None, aero: str = 'lap'):
        from gui import wheel_package as WP
        self._WP = WP
        self.win, self.res, self.track = win, res, track
        self.st = lap_stations(res, track)
        self.idx = np.asarray(indices, int) if indices is not None else solve_indices(self.st, every_m)
        self.every_m = float(every_m)
        self.solved = {}                    # station index -> {(corner,key): N}
        self._next = 0
        self.t_solve_s = 0.0
        self.steady = None
        self._series = None
        # AERO.  'lap' = the downforce the lap was run with: a lap run with no
        # aero (Cl·A 0 on the lap page) is solved with Apply Aero off; a lap
        # with aero uses the project package (the only one compute_case has)
        # and the note says whether it matches the lap's.  'project' = always
        # the project package (Dynamics panel) at the station speed.
        self.aero = aero
        lap_down = (np.asarray(res.aero_down_N, float) if getattr(res, 'aero_down_N', None) is not None
                    else np.zeros(len(self.st['s'])))
        self.lap_has_aero = bool(np.nanmax(np.abs(lap_down)) > 1.0) if lap_down.size else False
        self.aero_off = (aero == 'lap' and not self.lap_has_aero)
        self.aero_note = self._aero_check(lap_down)

    @property
    def n_total(self) -> int:
        return len(self.idx)

    @property
    def n_done(self) -> int:
        return self._next

    @property
    def done(self) -> bool:
        return self._next >= len(self.idx)

    def case_at(self, i: int):
        """(solver lat g, lon g, speed km/h) of station i."""
        st = self.st
        return float(st['lat_solver'][i]), float(st['lon'][i]), float(st['v_kph'][i])

    def solve_station(self, i: int) -> dict:
        lat, lon, v = self.case_at(i)
        if not self.aero_off:
            return member_values(self._WP.compute_case(self.win, lat, lon, speed_kph=v)[0])
        was = getattr(self.win, '_aero_active', False)
        self.win._aero_active = False           # restored before any event is processed
        try:
            return member_values(self._WP.compute_case(self.win, lat, lon, speed_kph=v)[0])
        finally:
            self.win._aero_active = was

    def step(self, budget_s: float = 0.08) -> bool:
        t0 = time.perf_counter()
        while not self.done:
            i = int(self.idx[self._next])
            self.solved[i] = self.solve_station(i)
            self._next += 1
            self._series = None
            if time.perf_counter() - t0 > budget_s:
                break
        self.t_solve_s += time.perf_counter() - t0
        if self.done and self.steady is None:
            self._finish()
        return self.done

    def run_all(self):
        while not self.step(1e9):
            pass
        return self

    # ── after the sweep ──────────────────────────────────────────────────
    def _finish(self):
        self.steady = steady_reference(self.win)

    def _aero_check(self, lap_down) -> str:
        """Say which downforce the loads use and compare it with the lap's at
        the fastest station."""
        try:
            i = int(np.argmax(self.st['v_kph']))
            lat, lon, v = self.case_at(i)
            lap_N = float(lap_down[i]) if len(lap_down) > i else 0.0
            if self.aero_off:
                return (f'Aero: none, the same as the lap (the lap sim ran with Cl·A 0), so the loads are solved '
                        f'with Apply Aero off.')
            ca = self._WP.case_aero(self.win, lat, lon, v)
            ld_N = float(ca['total_N'])
            src = (f"the project aero ({ca['package']})" if ca['aero_Fz'] else
                   'no aero (Apply Aero is off on the Dynamics panel)')
            txt = (f'Aero: loads use {src} at each station speed; at the top speed {v:.0f} km/h that is '
                   f'{ld_N:.0f} N, the lap sim used {lap_N:.0f} N.')
            if abs(ld_N - lap_N) > max(20.0, 0.05 * max(ld_N, lap_N)):
                txt = ('NOTE: the lap and the loads do NOT use the same downforce.  ' + txt +
                       '  Set the lap page Cl·A / CoP to the project aero (or the other way round) to match.')
            return txt
        except Exception as e:
            return f'aero check not available ({e})'

    def series(self) -> dict:
        """{(corner,key): full-length array} — solved stations, straight-line
        interpolated between them (exact when every station is solved)."""
        if self._series is not None:
            return self._series
        idx = np.array(sorted(self.solved), int)
        s = self.st['s']
        out = {}
        if len(idx) == 0:
            return out
        for ck in self.solved[int(idx[0])]:
            y = np.array([self.solved[int(i)][ck] for i in idx], float)
            ok = np.isfinite(y)
            out[ck] = (np.interp(s, s[idx[ok]], y[ok]) if ok.sum() >= 2 else np.full(len(s), np.nan))
        self._series = out
        return out

    def peaks(self) -> list:
        """One row per (corner, member): the largest-size value at a SOLVED
        station and where it happens, plus the worst steady reference case."""
        rows = []
        idx = np.array(sorted(self.solved), int)
        st = self.st
        for lbl in CORNERS:
            for key, label, sign, _f in MEMBERS:
                y = np.array([self.solved[int(i)][(lbl, key)] for i in idx], float)
                if not np.any(np.isfinite(y)):
                    continue
                j = int(np.nanargmax(np.abs(y)))
                i = int(idx[j])
                ref = (self.steady or {}).get((lbl, key))
                rows.append(dict(corner=lbl, key=key, label=label, sign=sign, value=float(y[j]), station=i,
                                 s=float(st['s'][i]), lat=float(st['lat_display'][i]), lon=float(st['lon'][i]),
                                 v_kph=float(st['v_kph'][i]),
                                 ref_value=(ref[0] if ref else float('nan')), ref_case=(ref[1] if ref else '')))
        return rows


def steady_reference(win) -> dict:
    """{(corner,key): (largest-size value, case name)} over the Loads-page
    steady cases (wheel_package.CASES at their own speed rule), each cornering
    case solved turning BOTH ways so every corner sees its outside case."""
    from gui import wheel_package as WP
    best = {}
    for name, lat, lon, _rule in WP.CASES:
        for sgn in ((1.0, -1.0) if lat else (1.0,)):
            try:
                L = WP.compute_case(win, sgn * lat, lon)[0]
            except Exception:
                continue
            for ck, v in member_values(L).items():
                if np.isfinite(v) and (ck not in best or abs(v) > abs(best[ck][0])):
                    best[ck] = (v, name)
    return best


# ── drawing (shared by the page and the pop-outs) ───────────────────────────
_BG, _AX = '#0b0b0d', '#080808'
_COLORS = ('#FFD600', '#E53935', '#FFFFFF', '#42A5F5')
_STYLES = ('-', '--', '-.', ':')


def _style(ax, xlabel, ylabel, title=''):
    ax.set_facecolor(_AX)
    ax.tick_params(colors='#9a9aa2', labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor('#2a2a2a')
    ax.set_xlabel(xlabel, color='#a0a0a8', fontsize=8)
    ax.set_ylabel(ylabel, color='#a0a0a8', fontsize=8)
    if title:
        ax.set_title(title, color='#dddddd', fontsize=9)
    ax.grid(True, color='#1e1e22', lw=0.6)


def draw_traces(fig, sweep: LapLoadSweep, corner: str, keys: list):
    """Lap g (context), speed, and the chosen members at one corner vs distance."""
    fig.clear()
    fig.set_facecolor(_BG)
    st = sweep.st
    s = st['s']
    gs = fig.add_gridspec(3, 1, height_ratios=(1.0, 0.8, 2.6), hspace=0.12)
    a0 = fig.add_subplot(gs[0]); a1 = fig.add_subplot(gs[1], sharex=a0); a2 = fig.add_subplot(gs[2], sharex=a0)
    a0.plot(s, st['lat_display'], color='#FFD600', lw=1.3, label='lateral g (+ left turn)', antialiased=True)
    a0.plot(s, st['lon'], color='#42A5F5', lw=1.3, label='fore-aft g (+ speeding up)', antialiased=True)
    a0.axhline(0.0, color='#444', lw=0.6)
    _style(a0, '', 'g'); a0.legend(fontsize=7, facecolor='#06060e', labelcolor='white', framealpha=0.7, ncol=1,
              loc='upper left', bbox_to_anchor=(1.005, 1.0), borderaxespad=0.0)
    a0.set_title(f'Loads around the lap — {CORNER_NAMES.get(corner, corner)} ({corner})', color='#dddddd', fontsize=9)
    a1.plot(s, st['v_kph'], color='#FFFFFF', lw=1.2, label='speed', antialiased=True)
    _style(a1, '', 'speed (km/h)')
    ser = sweep.series()
    idx = np.array(sorted(sweep.solved), int)
    for n, key in enumerate(keys):
        y = ser.get((corner, key))
        if y is None:
            continue
        col = _COLORS[n % 4]; ls = _STYLES[(n // 4) % 4]
        a2.plot(s, y, color=col, ls=ls, lw=1.6, antialiased=True,
                label=f'{MEMBER_LABEL[key]} ({MEMBER_SIGN[key]})')
        if 0 < len(idx) < len(s):                       # show where it was actually solved
            a2.plot(s[idx], y[idx], ls='None', marker='o', ms=2.2, color=col, alpha=0.8)
    a2.axhline(0.0, color='#444', lw=0.6)
    _style(a2, 'distance around the lap (m)', 'force (N)')
    if keys:
        a2.legend(fontsize=7, facecolor='#06060e', labelcolor='white', framealpha=0.7,
                  ncol=1, loc='upper left', bbox_to_anchor=(1.005, 1.0), borderaxespad=0.0)
    for a in (a0, a1):
        a.tick_params(labelbottom=False)
    a0.set_xlim(s[0], s[-1])
    fig.subplots_adjust(left=0.08, right=0.74, top=0.95, bottom=0.08)


def draw_map(fig, sweep: LapLoadSweep, corner: str, key: str):
    """Track map coloured by the size of one member's load at one corner."""
    fig.clear()
    fig.set_facecolor(_BG)
    ax = fig.add_subplot(111)
    st = sweep.st
    x, y = st['x'], st['y']
    ser = sweep.series()
    val = np.abs(ser.get((corner, key), np.full(len(x), np.nan)))
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    cv = 0.5 * (val[1:] + val[:-1])
    ok = np.isfinite(cv)
    lc = LineCollection(seg[ok], cmap='cividis', linewidths=4.0, antialiaseds=True, capstyle='round')
    lc.set_array(cv[ok])
    if ok.any():
        lc.set_clim(float(np.nanmin(cv)), float(np.nanmax(cv)))
    ax.add_collection(lc)
    ax.plot(x[0], y[0], marker='o', color='#E53935', ms=7, ls='None')
    ax.annotate('start', (x[0], y[0]), color='#E53935', fontsize=8, xytext=(5, 5), textcoords='offset points')
    if ok.any():
        j = int(np.nanargmax(val))
        ax.plot(x[j], y[j], marker='*', color='#FFFFFF', ms=13, ls='None')
        ax.annotate(f'peak {val[j]:,.0f} N at {st["s"][j]:.0f} m', (x[j], y[j]), color='#FFFFFF', fontsize=8,
                    xytext=(6, -12), textcoords='offset points')
    ax.set_aspect('equal', adjustable='box')
    ax.autoscale_view()
    _style(ax, 'x (m)', 'y (m)', f'{MEMBER_LABEL[key]} at {corner} — size of the force')
    cb = fig.colorbar(lc, ax=ax, orientation='horizontal', fraction=0.06, pad=0.14, aspect=40)
    cb.set_label(f'|{MEMBER_LABEL[key]}| (N)', color='#a0a0a8', fontsize=8)
    cb.ax.tick_params(colors='#9a9aa2', labelsize=7)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.10)
    return ax


class _MapHover:
    """Hover the track map: nearest station's distance, g, speed and load."""

    def __init__(self, canvas, page, corner=None, key=None):
        self.canvas, self.page, self.ann = canvas, page, None
        self._fixed = (corner, key)
        canvas.mpl_connect('motion_notify_event', self._on)

    def _on(self, ev):
        sw = self.page._sweep
        if sw is None or ev.inaxes is None or ev.xdata is None or not sw.solved:
            return
        ax = ev.inaxes
        if not ax.collections:
            return
        st = sw.st
        d = np.hypot(st['x'] - ev.xdata, st['y'] - ev.ydata)
        i = int(np.argmin(d))
        corner = self._fixed[0] or self.page._corner()
        key = self._fixed[1] or self.page._map_key()
        v = sw.series().get((corner, key), np.full(len(d), np.nan))[i]
        txt = (f'{st["s"][i]:.0f} m   {st["v_kph"][i]:.0f} km/h\n'
               f'lat {st["lat_display"][i]:+.2f} g   fore-aft {st["lon"][i]:+.2f} g\n'
               f'{MEMBER_LABEL[key]} {corner}: {v:,.0f} N')
        if self.ann is None or self.ann.axes is not ax:
            self.ann = ax.annotate('', (0, 0), xytext=(12, 12), textcoords='offset points', fontsize=8,
                                   color='white', bbox=dict(boxstyle='round', fc='#101018', ec='#555', alpha=0.9))
        self.ann.xy = (st['x'][i], st['y'][i])
        self.ann.set_text(txt)
        self.canvas.draw_idle()


class _PopOut(QDialog):
    """A big copy of one plot with zoom / pan (matplotlib toolbar) + hover."""

    def __init__(self, parent, title, drawer, map_hover_page=None, fixed=(None, None)):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1300, 850)
        lay = QVBoxLayout(self)
        fig = Figure(figsize=(12, 8), dpi=100)
        self.canvas = (_ExportCanvas(fig) if map_hover_page is not None else ReadableCanvas(fig))
        lay.addWidget(NavigationToolbar2QT(self.canvas, self))
        lay.addWidget(self.canvas)
        drawer(fig)
        if map_hover_page is not None:
            self._hover = _MapHover(self.canvas, map_hover_page, *fixed)
        self.canvas.draw_idle()


class _NumItem(QTableWidgetItem):
    """Table cell that sorts by its number (UserRole), not its text."""

    def __lt__(self, other):
        a = self.data(Qt.ItemDataRole.UserRole); b = other.data(Qt.ItemDataRole.UserRole)
        try:
            a = float(a); b = float(b)
            a = -np.inf if not np.isfinite(a) else a; b = -np.inf if not np.isfinite(b) else b
            return a < b
        except Exception:
            return super().__lt__(other)


class LapLoadsWindow(QWidget):
    """Loads around the lap.  Opened from the Lap Time page (button under the
    Sim button); takes the page's last LapResult and track."""

    _SPACING = (('every station (exact, slowest)', 0.0), ('every 2 m + g peaks', 2.0),
                ('every 5 m + g peaks', 5.0), ('every 10 m + g peaks (fast)', 10.0))

    def __init__(self, main, lap_page=None):
        super().__init__(None, Qt.WindowType.Window)
        self._main, self._lap_page = main, lap_page
        self._sweep: LapLoadSweep | None = None
        self._timer = QTimer(self); self._timer.setInterval(0); self._timer.timeout.connect(self._tick)
        self._t0 = 0.0
        self.setWindowTitle('Vahan — loads around the lap')
        self.resize(1500, 950)
        self.setStyleSheet('QWidget { background:#0e0e10; color:#e8e8ec; }'
                           'QPushButton { background:#26262c; padding:5px 10px; border:1px solid #3a3a40; border-radius:3px; }'
                           'QPushButton:hover { background:#34343c; }'
                           'QTableWidget { gridline-color:#2a2a2a; }'
                           'QHeaderView::section { background:#1a1a1e; color:#cfcfd4; padding:3px; border:0px; }')
        self._build()

    # ── UI ──
    def _build(self):
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        self._go = QPushButton('Compute loads for the last lap')
        self._go.setStyleSheet('QPushButton { background:#E23B48; color:white; font-weight:bold; padding:6px 12px; }')
        self._go.clicked.connect(self.start)
        self._stop = QPushButton('Stop'); self._stop.setEnabled(False); self._stop.clicked.connect(self.stop)
        self._spacing = QComboBox()
        for lbl, _v in self._SPACING:
            self._spacing.addItem(lbl)
        self._spacing.setToolTip('How many lap stations get a full load solve.  Between solved stations the curves '
                                 'are straight lines (dots on the plot show the solved stations).')
        self._prog = QLabel('Run a lap on the Lap Time page, then press Compute.')
        self._prog.setStyleSheet('color:#cfcfd4;')
        self._aero_cb = QComboBox()
        self._aero_cb.addItem('aero: same as the lap', 'lap')
        self._aero_cb.addItem('aero: project package (Dynamics panel)', 'project')
        self._aero_cb.setToolTip('Same as the lap: a lap run with Cl·A 0 is solved with no aero.  A lap with aero uses '
                                 'the project aero package at each station speed (the note says if it differs from '
                                 'the downforce of the lap).')
        for w in (self._go, self._stop, QLabel('Solve at:'), self._spacing, self._aero_cb, self._prog):
            top.addWidget(w)
        top.addStretch(1)
        root.addLayout(top)
        self._info = QLabel(''); self._info.setWordWrap(True); self._info.setStyleSheet('color:#9a9aa2; font-size:11px;')
        root.addWidget(self._info)

        body = QSplitter(Qt.Orientation.Horizontal)
        # left: choices
        left = QWidget(); ll = QVBoxLayout(left); ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel('Corner'))
        self._corner_cb = QComboBox()
        for c in CORNERS:
            self._corner_cb.addItem(f'{c} — {CORNER_NAMES[c]}', c)
        self._corner_cb.currentIndexChanged.connect(self._redraw)
        ll.addWidget(self._corner_cb)
        ll.addWidget(QLabel('Parts to plot (tick several)'))
        self._mem_list = QListWidget()
        for key, label, sign, _f in MEMBERS:
            it = QListWidgetItem(f'{label}   ({sign})')
            it.setData(Qt.ItemDataRole.UserRole, key)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if key in ('lca_front', 'lca_rear', 'pushrod', 'tierod')
                             else Qt.CheckState.Unchecked)
            self._mem_list.addItem(it)
        self._mem_list.itemChanged.connect(self._redraw)
        ll.addWidget(self._mem_list, 1)
        ll.addWidget(QLabel('Colour the track map by'))
        self._map_cb = QComboBox()
        for key, label, _s, _f in MEMBERS:
            self._map_cb.addItem(label, key)
        self._map_cb.setCurrentIndex([k for k, *_ in MEMBERS].index('pushrod'))
        self._map_cb.currentIndexChanged.connect(self._redraw)
        ll.addWidget(self._map_cb)
        ll.addWidget(QLabel('Table shows'))
        self._tbl_cb = QComboBox(); self._tbl_cb.addItems(['the chosen corner', 'all four corners'])
        self._tbl_cb.currentIndexChanged.connect(self._fill_table)
        ll.addWidget(self._tbl_cb)
        body.addWidget(left)

        # middle: traces + map (vertical splitter with the table)
        right = QSplitter(Qt.Orientation.Vertical)
        plots = QSplitter(Qt.Orientation.Horizontal)
        tw = QWidget(); tl = QVBoxLayout(tw); tl.setContentsMargins(0, 0, 0, 0)
        self._fig_tr = Figure(figsize=(9, 6), dpi=100)
        self._cv_tr = ReadableCanvas(self._fig_tr)
        b1 = QPushButton('Pop out (zoom / pan)'); b1.clicked.connect(self._pop_traces)
        tl.addWidget(self._cv_tr, 1); tl.addWidget(b1, 0, Qt.AlignmentFlag.AlignRight)
        mw = QWidget(); ml = QVBoxLayout(mw); ml.setContentsMargins(0, 0, 0, 0)
        self._fig_map = Figure(figsize=(5, 6), dpi=100)
        self._cv_map = _ExportCanvas(self._fig_map)
        self._map_hover = _MapHover(self._cv_map, self)
        b2 = QPushButton('Pop out (zoom / pan)'); b2.clicked.connect(self._pop_map)
        ml.addWidget(self._cv_map, 1); ml.addWidget(b2, 0, Qt.AlignmentFlag.AlignRight)
        # traces full width on top; map | peak table underneath (tracks are
        # usually long and flat, so the map wants width, not height)
        right.addWidget(tw)
        plots.addWidget(mw)
        self._tbl = QTableWidget(0, 11)
        self._tbl.setHorizontalHeaderLabels(
            ['Corner', 'Part', 'Peak on the lap (N)', 'Sign', 'At distance (m)', 'Lateral g', 'Fore-aft g',
             'Speed (km/h)', 'Worst steady case (N)', 'Steady case', 'Lap peak / steady'])
        self._tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._tbl.verticalHeader().setVisible(False)
        self._tbl.setSortingEnabled(True)
        plots.addWidget(self._tbl)
        plots.setStretchFactor(0, 2); plots.setStretchFactor(1, 3)
        right.addWidget(plots)
        right.setStretchFactor(0, 3); right.setStretchFactor(1, 2)
        body.addWidget(right)
        body.setStretchFactor(0, 0); body.setStretchFactor(1, 1)
        body.setSizes([260, 1240])
        root.addWidget(body, 1)
        self._placeholder()

    def _placeholder(self):
        for fig, txt in ((self._fig_tr, 'No loads yet — run a lap, then Compute.'), (self._fig_map, 'Track map')):
            fig.clear(); fig.set_facecolor(_BG)
            fig.text(0.5, 0.5, txt, color='#777', ha='center', va='center')
        self._cv_tr.draw_idle(); self._cv_map.draw_idle()

    # ── choices ──
    def _corner(self) -> str:
        return self._corner_cb.currentData() or 'FL'

    def _keys(self) -> list:
        out = []
        for r in range(self._mem_list.count()):
            it = self._mem_list.item(r)
            if it.checkState() == Qt.CheckState.Checked:
                out.append(it.data(Qt.ItemDataRole.UserRole))
        return out

    def _map_key(self) -> str:
        return self._map_cb.currentData() or 'pushrod'

    # ── compute ──
    def start(self, res=None, track=None):
        lp = self._lap_page or getattr(self._main, '_laptime_page', None)
        res = res if res is not None else getattr(lp, '_result', None)
        track = track if track is not None else getattr(lp, '_track', None)
        if res is None or track is None:
            self._prog.setText('No lap yet: press Sim on the Lap Time page first.')
            return
        every = self._SPACING[self._spacing.currentIndex()][1]
        try:
            self._sweep = LapLoadSweep(self._main, res, track, every_m=every,
                                       aero=self._aero_cb.currentData() or 'lap')
        except Exception as e:
            self._prog.setText(f'Could not set up the sweep: {e}')
            return
        self._go.setEnabled(False); self._stop.setEnabled(True)
        self._t0 = time.perf_counter()
        self._timer.start()

    def stop(self):
        self._timer.stop()
        self._go.setEnabled(True); self._stop.setEnabled(False)
        if self._sweep is not None and self._sweep.solved:
            self._prog.setText(f'Stopped after {self._sweep.n_done} of {self._sweep.n_total} stations.')
            self._redraw()

    def _tick(self):
        sw = self._sweep
        try:
            done = sw.step(0.08)
        except Exception as e:
            self._timer.stop(); self._go.setEnabled(True); self._stop.setEnabled(False)
            self._prog.setText(f'Load solve failed: {e}')
            return
        el = time.perf_counter() - self._t0
        if not done:
            rate = el / max(sw.n_done, 1)
            self._prog.setText(f'Solving loads: {sw.n_done} / {sw.n_total} stations '
                               f'({100 * sw.n_done / sw.n_total:.0f} %), about {rate * (sw.n_total - sw.n_done):.0f} s left')
            return
        self._timer.stop(); self._go.setEnabled(True); self._stop.setEnabled(False)
        self.finished(el)

    def finished(self, elapsed_s: float = 0.0):
        sw = self._sweep
        n = len(sw.st['s'])
        how = ('every station solved (no interpolation)' if sw.n_total >= n else
               f'{sw.n_total} of {n} stations solved (every {sw.every_m:g} m plus the peak-g stations); '
               f'straight lines between them, dots mark the solved stations')
        st = sw.st
        self._prog.setText(f'Done: {sw.n_total} load solves in {elapsed_s:.1f} s.')
        self._info.setText(
            f'Lap: {st["s"][-1]:.0f} m, peak lateral {np.max(np.abs(st["lat_display"])):.2f} g, peak braking '
            f'{-np.min(st["lon"]):.2f} g, peak drive {np.max(st["lon"]):.2f} g, top speed {np.max(st["v_kph"]):.0f} km/h.  '
            f'{how}.  Each station = the Loads-page solve (compute_case) at that station\'s g and speed; turn '
            f'direction from the track curvature.  {sw.aero_note}')
        self._redraw()

    # ── draw ──
    def _redraw(self, *_):
        if self._sweep is None or not self._sweep.solved:
            return
        draw_traces(self._fig_tr, self._sweep, self._corner(), self._keys())
        self._cv_tr.draw_idle()
        self._map_hover.ann = None
        draw_map(self._fig_map, self._sweep, self._corner(), self._map_key())
        self._cv_map.draw_idle()
        self._fill_table()

    def _fill_table(self, *_):
        sw = self._sweep
        if sw is None or not sw.solved:
            return
        rows = sw.peaks()
        if self._tbl_cb.currentIndex() == 0:
            rows = [r for r in rows if r['corner'] == self._corner()]
        self._tbl.setSortingEnabled(False)
        self._tbl.setRowCount(len(rows))
        for r, d in enumerate(rows):
            ratio = (abs(d['value']) / abs(d['ref_value']) if np.isfinite(d['ref_value']) and abs(d['ref_value']) > 1.0
                     else float('nan'))
            cells = [d['corner'], d['label'], d['value'], d['sign'], d['s'], d['lat'], d['lon'], d['v_kph'],
                     d['ref_value'], d['ref_case'], ratio]
            fmts = [None, None, '{:,.0f}', None, '{:.0f}', '{:+.2f}', '{:+.2f}', '{:.0f}', '{:,.0f}', None, '{:.0%}']
            for k, (v, f) in enumerate(zip(cells, fmts)):
                if f is None:
                    it = QTableWidgetItem(str(v))
                else:
                    it = _NumItem(f.format(v) if np.isfinite(v) else '—')
                    it.setData(Qt.ItemDataRole.UserRole, float(v))
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if k == 10 and np.isfinite(ratio) and ratio > 1.0:
                    it.setForeground(Qt.GlobalColor.red)     # lap goes past the steady design case
                self._tbl.setItem(r, k, it)
        self._tbl.setSortingEnabled(True)

    def _pop_traces(self):
        if self._sweep is None or not self._sweep.solved:
            return
        sw, c, k = self._sweep, self._corner(), self._keys()
        d = _PopOut(self, f'Loads around the lap — {c}', lambda fig: draw_traces(fig, sw, c, k))
        d.show()

    def _pop_map(self):
        if self._sweep is None or not self._sweep.solved:
            return
        sw, c, k = self._sweep, self._corner(), self._map_key()
        d = _PopOut(self, f'Track map — {MEMBER_LABEL[k]} at {c}', lambda fig: draw_map(fig, sw, c, k),
                    map_hover_page=self, fixed=(c, k))
        d.show()


def open_window(main, lap_page=None):
    """Show (and keep) the one lap-loads window of this MainWindow."""
    w = getattr(main, '_lap_loads_window', None)
    if w is None:
        w = LapLoadsWindow(main, lap_page)
        main._lap_loads_window = w
    w.show(); w.raise_(); w.activateWindow()
    return w
