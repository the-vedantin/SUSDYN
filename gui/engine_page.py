"""Engine page — the curve the lap sim runs on, with every input visible.

Three methods, all from vahan.engine (ONE source):
  raw       — the SDM26 1-D solver output, known ~2x low (reference).
  corrected — per-rpm physics correction: breathing (VE) raised to a stated
              target curve, solver friction replaced by a stated realistic line.
  anchored  — raw curve SHAPE scaled to a stated published crank peak.

The chosen method + its inputs are saved into the project (car dict) and the
lap sim consumes exactly this curve.  Nothing hidden behind the screen.

Credits — powertrain model: 1dFVEngineSolver, Sun Devil Motorsports (MIT).
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QComboBox,
    QDoubleSpinBox, QPushButton, QGroupBox,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from vahan.engine import (engine_curve, DEFAULT_METHOD, VE_TARGET,
                          VE_FALL_START_RPM, VE_AT_13K, FMEP_A_BAR,
                          FMEP_B_BAR, ANCHOR_HP, CREDIT_POWERTRAIN)

_INK, _RED, _AMBER, _GREY = '#e8e8ea', '#e53935', '#e0a021', '#8a8a8f'


class EnginePage(QWidget):
    def __init__(self, main):
        super().__init__()
        self._main = main
        root = QVBoxLayout(self)

        # ── inputs (ALL of them, no hidden constants) ─────────────────────
        box = QGroupBox('Engine model inputs — what the lap sim runs on')
        g = QGridLayout(box)

        def spin(row, col, label, lo, hi, val, dec, step, tip):
            g.addWidget(QLabel(label), row, col * 2)
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi); sp.setDecimals(dec)
            sp.setSingleStep(step); sp.setValue(val)
            sp.setToolTip(tip)
            sp.valueChanged.connect(self._refresh)
            g.addWidget(sp, row, col * 2 + 1)
            return sp

        g.addWidget(QLabel('Method:'), 0, 0)
        self._method = QComboBox()
        self._method.addItems(['corrected', 'anchored', 'raw'])
        self._method.setToolTip(
            'corrected = physics fix (breathing + friction), the default.\n'
            'anchored = raw shape scaled to the published peak.\n'
            'raw = uncalibrated solver output, known ~2x low.')
        self._method.currentTextChanged.connect(self._refresh)
        g.addWidget(self._method, 0, 1)

        self._ve = spin(0, 1, 'VE target (peak):', 0.5, 1.1, VE_TARGET, 3, 0.01,
                        'Breathing target: fraction of a full cylinder the engine '
                        'inhales at its best. Real race 600s: 0.85-0.95. The raw '
                        'solver caps at ~0.66 (its known flaw).')
        self._vefall = spin(0, 2, 'VE fall starts (rpm):', 6000, 13000,
                            VE_FALL_START_RPM, 0, 250,
                            'Above this rpm the breathing target tapers off '
                            '(restrictor starvation).')
        self._ve13 = spin(1, 0, 'VE at 13,000:', 0.4, 1.0, VE_AT_13K, 3, 0.01,
                          'Breathing target at 13,000 rpm (end of the taper).')
        self._fa = spin(1, 1, 'Friction A (bar):', 0.0, 3.0, FMEP_A_BAR, 2, 0.05,
                        'Realistic friction line: FMEP = A + B x (rpm/1000) bar. '
                        'The raw solver reads 2.6-5.4 bar (about double reality).')
        self._fb = spin(1, 2, 'Friction B (bar/krpm):', 0.0, 1.0, FMEP_B_BAR,
                        2, 0.02, 'Slope of the friction line per 1000 rpm.')
        self._anchor = spin(2, 0, 'Anchor peak (hp crank):', 40, 130, ANCHOR_HP,
                            1, 1.0, 'Published restricted-CBR600RR crank peak '
                            '(75-85 typical) used by the anchored method.')
        apply_btn = QPushButton('Apply to lap sim')
        apply_btn.setToolTip('Save this method + inputs into the project; the '
                             'lap sim and every consumer use this curve.')
        apply_btn.clicked.connect(self._apply)
        g.addWidget(apply_btn, 2, 3, 1, 2)
        root.addWidget(box)

        # ── plot ──────────────────────────────────────────────────────────
        self._fig = Figure(figsize=(9, 5), facecolor='#1b1b1e')
        self._canvas = FigureCanvasQTAgg(self._fig)
        root.addWidget(self._canvas, 1)

        self._status = QLabel('')
        self._status.setWordWrap(True)
        root.addWidget(self._status)
        self._load_saved()
        self._refresh()

    # ── settings <-> project (car dict) ──────────────────────────────────
    def _load_saved(self):
        car = getattr(self._main, '_car', {})
        m = car.get('engine_method')
        if m in ('corrected', 'anchored', 'raw'):
            self._method.setCurrentText(m)
        for key, sp in (('engine_ve_target', self._ve),
                        ('engine_ve_fall_rpm', self._vefall),
                        ('engine_ve_13k', self._ve13),
                        ('engine_fmep_a', self._fa),
                        ('engine_fmep_b', self._fb),
                        ('engine_anchor_hp', self._anchor)):
            if key in car:
                sp.blockSignals(True); sp.setValue(float(car[key]))
                sp.blockSignals(False)

    def _apply(self):
        car = getattr(self._main, '_car', {})
        car['engine_method'] = self._method.currentText()
        car['engine_ve_target'] = float(self._ve.value())
        car['engine_ve_fall_rpm'] = float(self._vefall.value())
        car['engine_ve_13k'] = float(self._ve13.value())
        car['engine_fmep_a'] = float(self._fa.value())
        car['engine_fmep_b'] = float(self._fb.value())
        car['engine_anchor_hp'] = float(self._anchor.value())
        self._main.statusBar().showMessage(
            'Engine model saved to project — lap sim now uses: '
            + self._method.currentText(), 6000)
        self._refresh()

    def current_curve(self):
        """(rpm[], torque[], label) for the page's current settings."""
        import vahan.engine as VE
        # push the VE-taper knobs (module-level defaults feed ve_target_curve)
        VE.VE_FALL_START_RPM = float(self._vefall.value())
        VE.VE_AT_13K = float(self._ve13.value())
        return engine_curve(self._method.currentText(),
                            ve_target=float(self._ve.value()),
                            fmep_a=float(self._fa.value()),
                            fmep_b=float(self._fb.value()),
                            anchor_hp=float(self._anchor.value()))

    # ── plotting ──────────────────────────────────────────────────────────
    def _refresh(self):
        self._fig.clear()
        ax1 = self._fig.add_subplot(121, facecolor='#141417')
        ax2 = self._fig.add_subplot(122, facecolor='#141417')
        sel = self._method.currentText()
        styles = {'raw': (_GREY, 1.4, '--'), 'corrected': (_AMBER, 1.6, '-'),
                  'anchored': (_RED, 1.6, '-')}
        peak_note = ''
        for m in ('raw', 'corrected', 'anchored'):
            try:
                import vahan.engine as VE
                VE.VE_FALL_START_RPM = float(self._vefall.value())
                VE.VE_AT_13K = float(self._ve13.value())
                c = engine_curve(m, ve_target=float(self._ve.value()),
                                 fmep_a=float(self._fa.value()),
                                 fmep_b=float(self._fb.value()),
                                 anchor_hp=float(self._anchor.value()))
            except Exception:
                c = None
            if c is None:
                continue
            r = np.asarray(c[0]); t = np.asarray(c[1])
            p = t * r * 2 * np.pi / 60 / 745.7
            col, lw, ls = styles[m]
            lw2 = lw + (1.4 if m == sel else 0.0)
            ax1.plot(r, t, color=col, lw=lw2, ls=ls, label=m)
            ax2.plot(r, p, color=col, lw=lw2, ls=ls, label=m)
            if m == sel:
                i = int(p.argmax())
                peak_note = (f'SELECTED "{m}": peak {p[i]:.1f} hp @ '
                             f'{r[i]:.0f} rpm, torque peak {t.max():.1f} N·m '
                             f'@ {r[t.argmax()]:.0f} rpm')
        for ax, ttl, yl in ((ax1, 'Crank torque', 'N·m'),
                            (ax2, 'Crank power', 'hp')):
            ax.set_title(ttl, color=_INK, fontsize=10)
            ax.set_xlabel('rpm', color=_INK, fontsize=8)
            ax.set_ylabel(yl, color=_INK, fontsize=8)
            ax.tick_params(colors=_GREY, labelsize=7)
            ax.grid(color='#26262c', lw=0.6)
            for s in ax.spines.values():
                s.set_color('#33333a')
            ax.legend(fontsize=7, facecolor='#1b1b1e', edgecolor='#33333a',
                      labelcolor=_INK)
        self._fig.tight_layout()
        self._canvas.draw_idle()
        self._status.setText(
            peak_note + '\nProvenance: 1dFVEngineSolver (Sun Devil Motorsports, '
            'MIT), 15-point converged sweep. The raw solver is known ~2x low '
            '(breathing floor ~0.66 VE + ~2x friction) — verified by sweeping '
            'plenum, runners, cams, valve flow and even removing the '
            'restrictor. "corrected" and "anchored" are two independent '
            'calibrations; they agree within ~3%. Anchor to a real dyno pull '
            'the day one exists.  ' + CREDIT_POWERTRAIN)
