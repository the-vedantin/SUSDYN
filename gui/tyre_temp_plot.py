"""Grip vs tyre temperature: the curve the lap sim applies (vahan.tyre_temperature
GripTempCurve) with the CURRENT cold/hot loss boxes, the measured range shaded,
the maker's window marked, and (if the data file has it) the all-R20-sizes
cross-check.  The data is read from the local tire_data/ file at run time."""
from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel
from matplotlib.figure import Figure

_YEL, _RED, _WHT, _BLU, _GRY = '#F2C14E', '#E53935', '#FFFFFF', '#42A5F5', '#8a8a8f'


def draw_grip_temp(fig, data: dict, cold_pct: float, hot_pct: float, marks=None):
    from vahan.tyre_temperature import GripTempCurve
    gc = data['grip_curve']
    cur = GripTempCurve(gc, cold_pct, hot_pct)
    fig.clear(); fig.set_facecolor('#0f0f12')
    ax = fig.add_subplot(111); ax.set_facecolor('#141417')
    ax.tick_params(colors=_WHT, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color('#3a3a40')
    ax.grid(alpha=0.18, color=_WHT)
    T = np.linspace(0.0, 110.0, 441)
    f = np.array([cur.factor(t) for t in T])
    lo, hi = cur.lo, cur.hi
    m = (T >= lo) & (T <= hi)
    ax.axvspan(lo, hi, color=_BLU, alpha=0.12, lw=0)
    ax.plot(T[T <= lo], f[T <= lo], color=_YEL, lw=2, ls='--', label=f'assumed: -{cold_pct:g} % per 10 °C colder')
    ax.plot(T[T >= hi], f[T >= hi], color=_RED, lw=2, ls='--', label=f'assumed: -{hot_pct:g} % per 10 °C hotter')
    ax.plot(T[m], f[m], color=_WHT, lw=2.6, label=f'measured (TTC, {lo:.0f}-{hi:.0f} °C)')
    cc = gc.get('cross_check_all_R20_sizes')
    if isinstance(cc, dict) and cc.get('T_C') and cc.get('rel_mu'):
        ax.plot(cc['T_C'], cc['rel_mu'], color=_BLU, lw=1.4, ls=':', marker='o', ms=3, label='cross-check: all R20 sizes')
    elif isinstance(cc, dict) and cc:
        try:
            pts = sorted((float(k), float(v)) for k, v in cc.items())
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=_BLU, lw=1.4, ls=':', marker='o', ms=3,
                    label='cross-check: all R20 sizes (TTC)')
        except (TypeError, ValueError):
            pass
    win = data.get('maker_window_C')
    if win:
        for x in win:
            ax.axvline(float(x), color=_GRY, lw=1, ls=':')
        ax.text(float(win[0]), 0.62, ' maker window', color=_GRY, fontsize=8)
    ax.axhline(1.0, color=_GRY, lw=0.8)
    ax.text(cur.ref_T, 1.005, f' x1.00 = {cur.ref_T:.0f} °C (what the tyre model is)', color=_GRY, fontsize=8, va='bottom')
    for lab, t in (marks or []):
        ax.plot([t], [cur.factor(t)], 'o', color=_YEL, ms=6)
        ax.annotate(f'{lab} {t:.0f} °C: x{cur.factor(t):.2f}', (t, cur.factor(t)), textcoords='offset points',
                    xytext=(6, -14), color=_YEL, fontsize=8)
    ax.set_xlim(0, 110); ax.set_ylim(0.6, 1.08)
    ax.set_xlabel('tyre surface temperature (°C)', color=_WHT, fontsize=10)
    ax.set_ylabel('grip multiplier used by the lap sim', color=_WHT, fontsize=10)
    ax.set_title('Grip vs tyre temperature', color=_WHT, fontsize=11)
    ax.legend(fontsize=8, facecolor='#141417', labelcolor=_WHT, loc='lower center')
    fig.tight_layout()
    return cur


class GripTempDialog(QDialog):
    def __init__(self, page):
        super().__init__(page)
        self.setWindowTitle('Grip vs tyre temperature — what the lap sim uses')
        self.resize(900, 560)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel('Multiplies the tyre grip at both axles.  Solid = measured; dashed = '
                             'the cold/hot loss boxes on the Laptime page (assumed).'))
        from gui.plot_dialog import ReadableCanvas
        fig = Figure(figsize=(9, 5.2))
        self.canvas = ReadableCanvas(fig)
        lay.addWidget(self.canvas)
        marks = []
        res = getattr(page, '_result', None)
        for k, lab in (('tyre_temp_front_C', 'last lap front'), ('tyre_temp_rear_C', 'last lap rear')):
            v = getattr(res, k, None) if res is not None else None
            try:
                if v is not None and np.size(v):
                    marks.append((lab, float(np.nanmean(v))))
            except Exception:
                pass
        draw_grip_temp(fig, page._tt_data, float(page._tt_cold.value()), float(page._tt_hot.value()), marks)
        self.canvas.draw()


def open_window(page):
    if not getattr(page, '_tt_data', None):
        page._tt_note.setText('No tyre-temperature data loaded.')
        return None
    dlg = GripTempDialog(page)
    dlg.show()
    page._tt_dlg = dlg
    return dlg
