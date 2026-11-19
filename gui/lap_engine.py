"""Engine curves the lap sim runs on: crank torque + power vs rpm (exactly
LapSimulator._engine_torque_Nm, i.e. the Engine-page curve or the dyno points
typed on the Laptime page), the wheel power after drivetrain losses, the
redline, and — if a lap has been run — how long the lap spent at each rpm.
Second plot: drive force at the wheels vs speed in every gear (primary x gear
x final from the Laptime page) against the tyre/drag needs."""
from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel
from matplotlib.figure import Figure

_YEL, _RED, _WHT, _BLU, _GRY = '#F2C14E', '#E53935', '#FFFFFF', '#42A5F5', '#8a8a8f'
_GEAR_COLS = ['#42A5F5', '#F2C14E', '#E53935', '#FFFFFF', '#90CAF9', '#FFE082']


def _style(ax, fig):
    ax.set_facecolor('#141417')
    ax.tick_params(colors=_WHT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color('#3a3a40')
    ax.grid(alpha=0.18, color=_WHT)


def draw_engine(fig, sim, res=None):
    fig.clear(); fig.set_facecolor('#0f0f12')
    red = float(getattr(sim, '_redline_rpm', 0.0) or 0.0)
    tq = getattr(sim, '_tq_rpm', None)
    # only the rpm range the curve really has (outside it the sim holds the
    # end value flat — drawing that would look like engine data)
    rpm_lo = float(np.min(tq)) if tq is not None else 1000.0
    rpm_hi = float(np.max(tq)) if tq is not None else (red if red > 0 else 14000.0)
    if red > 0:
        rpm_hi = max(rpm_hi, red) if tq is None else rpm_hi
    rpm = np.linspace(rpm_lo, rpm_hi, 300)
    T = np.array([sim._engine_torque_Nm(r) for r in rpm])
    w = rpm * 2 * np.pi / 60.0
    P_hp = T * w / 745.7
    eta = float(getattr(sim.veh, 'drivetrain_efficiency', 1.0) or 1.0)

    ax1 = fig.add_subplot(121); _style(ax1, fig)
    ax1b = ax1.twinx(); ax1b.tick_params(colors=_WHT, labelsize=8)
    if res is not None and getattr(res, 'engine_rpm', None) is not None:
        er = np.asarray(res.engine_rpm, float); t = np.asarray(res.t_s, float)
        dt = np.diff(t, prepend=t[0]); ok = np.isfinite(er) & (er > 0)
        if ok.any():
            bins = np.linspace(rpm_lo, max(rpm_hi, red), 40)
            h, e = np.histogram(er[ok], bins=bins, weights=dt[ok])
            ax3 = ax1.twinx(); ax3.set_ylim(0, max(h.max() * 100 / max(t[-1], 1e-6), 1) * 3)
            ax3.set_facecolor('none'); ax3.bar(0.5 * (e[1:] + e[:-1]), h * 100 / max(t[-1], 1e-6), width=np.diff(e), color=_GRY, alpha=0.30, zorder=0)
            ax3.set_yticks([]); ax3.set_zorder(0); ax1.set_zorder(1); ax1b.set_zorder(2)
            ax1.patch.set_visible(False)
            ax1.text(0.02, 0.02, 'grey bars: share of the lap at each rpm', transform=ax1.transAxes, color=_GRY, fontsize=7)
    l1, = ax1.plot(rpm, T, color=_YEL, lw=2, label='crank torque (N·m)')
    l2, = ax1b.plot(rpm, P_hp, color=_RED, lw=2, label='crank power (hp)')
    l3, = ax1b.plot(rpm, P_hp * eta, color=_RED, lw=1.2, ls='--', label=f'wheel power (x{eta:.2f}) (hp)')
    if red > 0:
        ax1b.axvline(red, color=_WHT, ls=':', lw=1.2)
        ax1b.text(red, ax1b.get_ylim()[1] * 0.97, f' redline {red:.0f}', color=_WHT, fontsize=7, va='top')
        ax1.set_xlim(rpm_lo, max(rpm_hi, red) * 1.02)
    i = int(np.argmax(T)); j = int(np.argmax(P_hp))
    ax1.set_title(f'Peak {T[i]:.1f} N·m @ {rpm[i]:.0f} rpm  ·  {P_hp[j]:.1f} hp @ {rpm[j]:.0f} rpm',
                  color=_WHT, fontsize=9)
    ax1.set_xlabel('engine rpm', color=_WHT, fontsize=8)
    ax1.set_ylabel('torque (N·m)', color=_YEL, fontsize=8)
    ax1b.set_ylabel('power (hp)', color=_RED, fontsize=8)
    ax1.legend(handles=[l1, l2, l3], fontsize=7, loc='upper left', bbox_to_anchor=(0.0, -0.14), ncol=3,
               borderaxespad=0.0, facecolor='#141417', labelcolor=_WHT)

    ax2 = fig.add_subplot(122); _style(ax2, fig)
    r_t = float(sim.veh.tire_radius_m)
    n = len(sim._gears) if sim._gears else 1
    vmax = 0.0
    for gi in range(1, n + 1):
        ratio = sim._total_ratio(gi)
        rr = np.linspace(rpm_lo, min(red, rpm_hi) if red > 0 else rpm_hi, 120)
        v = rr * 2 * np.pi / 60.0 / ratio * r_t * 3.6
        F = np.array([sim._engine_torque_Nm(x) for x in rr]) * ratio * eta / r_t
        ax2.plot(v, F, color=_GEAR_COLS[(gi - 1) % len(_GEAR_COLS)], lw=1.8, label=f'gear {gi} (x{ratio:.2f})')
        vmax = max(vmax, float(v[-1]))
    vv = np.linspace(1, max(vmax, 10), 100) / 3.6
    drag = 0.5 * sim.rho * sim.cda * vv ** 2
    ax2.plot(vv * 3.6, drag, color=_GRY, ls='--', lw=1.2, label=f'drag (CdA {sim.cda:.2f})')
    if res is not None:
        ax2.set_title(f'lap {res.lap_time_s:.2f} s, top {res.max_speed_kph:.1f} km/h', color=_WHT, fontsize=9)
        ax2.axvline(float(res.max_speed_kph), color=_WHT, ls=':', lw=1)
    else:
        ax2.set_title('drive force at the wheels, per gear', color=_WHT, fontsize=9)
    ax2.set_xlabel('speed (km/h)', color=_WHT, fontsize=8)
    ax2.set_ylabel('drive force at the wheels (N)', color=_WHT, fontsize=8)
    ax2.legend(fontsize=7, loc='upper left', bbox_to_anchor=(0.0, -0.14), ncol=3,
               borderaxespad=0.0, facecolor='#141417', labelcolor=_WHT)
    fig.tight_layout()


class LapEngineDialog(QDialog):
    def __init__(self, main, page, sim, res):
        super().__init__(page)
        self.setWindowTitle('Engine curves — what the lap sim runs on')
        self.resize(1250, 560)
        lay = QVBoxLayout(self)
        try:                      # the same curve the lap sim uses (one source)
            _c = page._torque_curve()
            src = _c[2] if _c is not None else 'the simple peak-power model (curve box off)'
        except Exception as e:
            src = f'unknown ({e})'
        lay.addWidget(QLabel(f'Source: {src}.  Gearing from this page.  '
                             + ('Grey bars = the last lap.' if res is not None else 'Run a lap to see where it spends its rpm.')))
        from gui.plot_dialog import ReadableCanvas
        fig = Figure(figsize=(12, 5))
        self.canvas = ReadableCanvas(fig)
        lay.addWidget(self.canvas)
        draw_engine(fig, sim, res)
        self.canvas.draw()


def open_window(main, page):
    res = getattr(page, '_result', None)
    sim = getattr(page, '_last_sim', None) if res is not None else None
    if sim is None:
        sim = page.build_sim()          # the car + boxes as they are now
        res = None
    dlg = LapEngineDialog(main, page, sim, res)
    dlg.show()
    page._engine_dlg = dlg               # keep a reference
    return dlg
