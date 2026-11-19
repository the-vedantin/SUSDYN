"""MMD from a lap: the app's Milliken Moment Diagram (MainWindow._mmd_kwargs
-> vahan.analysis_plots.plot_mmd, the ONE MMD path) drawn at the speed of a
corner of the lap just run, with that lap's grip scale and aero, and the
lap's own points in that corner marked on it.

The lap sim is quasi-steady: at every station the car is trimmed (yaw moment
zero), so its points sit on the N = 0 line.  The diagram shows what is left
around them: how much more sideways g, and how much yaw moment to turn in or
straighten up, the car has at that speed."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QComboBox,
                             QDoubleSpinBox, QPushButton, QLabel, QFileDialog,
                             QApplication)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

_YELLOW = '#F2C14E'
_WHITE = '#FFFFFF'


def lap_mmd_kwargs(main, sim, v_mps: float) -> dict:
    """plot_mmd kwargs for the car as the lap saw it at speed v_mps:
    the app's MMD inputs, with the LAP's grip scale and the LAP's aero
    (ClA and centre of pressure from the Laptime page) at this speed."""
    kw = main._mmd_kwargs()
    kw['velocity_mps'] = float(v_mps)
    kw['grip_multiplier'] = float(sim.grip_scale)
    aero = None
    if abs(float(getattr(sim, 'cla', 0.0))) > 1e-6:
        Fd = float(sim.downforce_N(float(v_mps)))
        fa = Fd * (1.0 - sim.cop_rear) / 2.0
        ra = Fd * sim.cop_rear / 2.0
        aero = {'FL': fa, 'FR': fa, 'RL': ra, 'RR': ra}
    kw['aero_Fz_N'] = aero
    return kw


def draw_lap_mmd(main, sim, res, corner: dict | None, v_kph: float):
    """Figure: the MMD at v_kph, plus the lap's points in `corner`."""
    from vahan.analysis_plots import plot_mmd
    v = max(float(v_kph), 1.0) / 3.6
    kw = lap_mmd_kwargs(main, sim, v)
    fig = plot_mmd(**kw)
    ax = fig.axes[0]
    if corner is not None and res is not None and res.lat_g is not None:
        a, b = int(corner['i_in']), int(corner['i_out'])
        ay = np.abs(np.asarray(res.lat_g)[a:b + 1])
        ax.plot(ay, np.zeros_like(ay), 'o', ms=4, color=_YELLOW, mec='k',
                mew=0.4, zorder=6, label=f'lap, corner {corner["index"]} (trimmed: N = 0)')
        ax.plot([ay.max()], [0.0], '*', ms=15, color=_YELLOW, mec='k',
                mew=0.8, zorder=7, label=f'lap peak {ay.max():.2f} g')
        h, l = ax.get_legend_handles_labels()
        keep = [k for k, s in enumerate(l) if not s.startswith(('β=', 'δ='))]
        ax.legend([h[k] for k in keep], [l[k] for k in keep],
                  loc='upper left', bbox_to_anchor=(1.02, 0.30), borderaxespad=0.0,
                  fontsize=8, framealpha=0.85, facecolor='#15151a', labelcolor='#e8e8ec')
    aero_txt = ('no aero' if not kw['aero_Fz_N'] else
                f'aero {sum(kw["aero_Fz_N"].values()):.0f} N')
    title = (f'MMD at {v_kph:.0f} km/h (from the lap)  |  grip x{kw["grip_multiplier"]:.2f}, {aero_txt}')
    if corner is not None:
        title += f'  |  corner {corner["index"]} ({corner["direction"]}), R {corner["R_harmonic_m"]:.1f} m'
    ax.set_title(title, fontsize=9)
    return fig, kw


class LapMMDDialog(QDialog):
    def __init__(self, main, page):
        super().__init__(page)
        self._main, self._page = main, page
        self._sim = getattr(page, '_last_sim', None)
        self._res = page._result
        self.setWindowTitle('MMD from this lap')
        self.resize(980, 820)
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel('Corner:'))
        self._corner = QComboBox()
        self._corners = list(getattr(self._res, 'corners', []) or [])
        for c in self._corners:
            self._corner.addItem(
                f'#{c["index"]} {c["direction"][0].upper()}  slowest {c["v_min_kph"]:.0f} / mean '
                f'{c["v_mean_kph"]:.0f} km/h, peak {abs(c["g_peak"]):.2f} g, R {c["R_harmonic_m"]:.1f} m')
        self._corner.addItem('No corner (just the speed below)')
        row.addWidget(self._corner, 1)
        row.addWidget(QLabel('Speed:'))
        self._v = QDoubleSpinBox(); self._v.setRange(5.0, 200.0); self._v.setDecimals(1)
        self._v.setSuffix(' km/h')
        self._v.setToolTip("Defaults to the corner's slowest (apex) speed on the lap.")
        row.addWidget(self._v)
        self._go = QPushButton('Draw'); self._go.clicked.connect(self._draw)
        row.addWidget(self._go)
        self._save = QPushButton('Save PNG…'); self._save.clicked.connect(self._save_png)
        row.addWidget(self._save)
        lay.addLayout(row)
        self._note = QLabel(
            'The lap is quasi-steady, so its points sit on N = 0 (the car is balanced at every '
            'station). The lines show what is left at that speed. The MMD uses its own simpler '
            'load-transfer model (the Dynamics-page MMD), so its limit can differ a little from '
            "the lap's G-G limit.")
        self._note.setWordWrap(True)
        self._note.setStyleSheet('color:#8a8a92; font-size:10px;')
        lay.addWidget(self._note)
        self._holder = QVBoxLayout(); lay.addLayout(self._holder, 1)
        self._canvas = None; self._fig = None
        self._corner.currentIndexChanged.connect(self._on_corner)
        if self._corners:     # open on the hardest corner of the lap
            self._corner.setCurrentIndex(int(np.argmax([abs(c['g_peak']) for c in self._corners])))
        self._on_corner()
        self._draw()

    def _cur_corner(self):
        i = self._corner.currentIndex()
        return self._corners[i] if 0 <= i < len(self._corners) else None

    def _on_corner(self, *_):
        c = self._cur_corner()
        if c is not None:
            self._v.setValue(float(c['v_min_kph']))

    def _draw(self):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            fig, _ = draw_lap_mmd(self._main, self._sim, self._res,
                                  self._cur_corner(), float(self._v.value()))
        except Exception as e:
            import traceback; traceback.print_exc()
            self._note.setText(f'MMD error: {e}')
            return
        finally:
            QApplication.restoreOverrideCursor()
        if self._canvas is not None:
            self._holder.removeWidget(self._canvas); self._canvas.deleteLater()
        self._fig = fig
        self._canvas = FigureCanvas(fig)
        self._holder.addWidget(self._canvas)
        self._canvas.draw()

    def _save_png(self):
        if self._fig is None:
            return
        c = self._cur_corner()
        name = f'lap_mmd_{"c%d_" % c["index"] if c else ""}{self._v.value():.0f}kph.png'
        p, _ = QFileDialog.getSaveFileName(self, 'Save MMD', name, 'PNG (*.png)')
        if p:
            self._fig.savefig(p, dpi=150)


def open_window(main, page):
    """Entry point for the Laptime page button."""
    if getattr(page, '_result', None) is None or getattr(page, '_last_sim', None) is None:
        page._status.setText('Run a lap first (► Sim Current Car), then draw its MMD.')
        return None
    if getattr(main, '_tire_model', None) is None:
        page._status.setText('The MMD needs TTC tyre data loaded.')
        return None
    dlg = LapMMDDialog(main, page)
    page._lap_mmd_dlg = dlg
    dlg.show()
    return dlg
