"""Packaging page — move the inboard actuation without changing the car.

MANUAL tab: mirror / rotate / translate / lever-scale the actuation slice of
either axle on the LIVE model (3-D view + graphs follow), with the full
validate() readout — every held parameter, the geometric laws, and the
full-travel clash sweep — refreshed after every move.  Wheel-locating points
are never touched, so wheel parameters are held exactly by construction.

GENERATOR tab: sample hundreds of transform compositions, re-tune MR + ARB
rate back to baseline, keep the validate() passers, rank by keep-out
clearance, save to configs/experiments/ (never configs/ root).

All physics comes from vahan.packaging -> the ONE model (corner solvers,
KinematicMetrics, dynamics build, vahan.interference).  Nothing here computes
suspension behaviour on its own.
"""
from __future__ import annotations

import os
import time

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QComboBox,
    QDoubleSpinBox, QSpinBox, QPushButton, QGroupBox, QTextEdit, QTabWidget,
    QTableWidget, QTableWidgetItem, QProgressBar, QApplication, QCheckBox,
)

from vahan import packaging as pkg

_INK, _RED, _GREY = '#e8e8ea', '#e53935', '#8a8a8f'


class PackagingPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self._main = main
        self._baseline = None          # validate() reference (capture_baseline)
        self._ref_bundles = None       # {'front':, 'rear':} bundles at capture
        self._tol = pkg.Tolerances()
        self._cancel = False

        root = QVBoxLayout(self)
        tabs = QTabWidget()
        root.addWidget(tabs)
        tabs.addTab(self._build_manual_tab(), 'Manual')
        tabs.addTab(self._build_generator_tab(), 'Generator')
        self._capture_baseline()

    # ══════════════════════════════════════════════════════════════════════
    #  Manual tab
    # ══════════════════════════════════════════════════════════════════════
    def _build_manual_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        top = QHBoxLayout()
        top.addWidget(QLabel('Axle:'))
        self._axle = QComboBox(); self._axle.addItems(['front', 'rear'])
        top.addWidget(self._axle)
        top.addWidget(QLabel('Step:'))
        self._step = QDoubleSpinBox()
        self._step.setRange(0.1, 50.0); self._step.setValue(5.0)
        self._step.setSuffix(' mm'); self._step.setDecimals(1)
        top.addWidget(self._step)
        top.addWidget(QLabel('Angle:'))
        self._angle = QDoubleSpinBox()
        self._angle.setRange(0.1, 45.0); self._angle.setValue(2.0)
        self._angle.setSuffix(' deg'); self._angle.setDecimals(1)
        top.addWidget(self._angle)
        cap_btn = QPushButton('Re-capture baseline')
        cap_btn.setToolTip('Snapshot the CURRENT model as the reference every '
                           'candidate is validated against (done automatically '
                           'when the page opens).')
        cap_btn.clicked.connect(self._capture_baseline)
        top.addWidget(cap_btn)
        top.addStretch(1)
        lay.addLayout(top)

        box = QGroupBox('Transforms — wheel-side points are NEVER moved; '
                        'the ARB re-hangs about its bar axis automatically')
        g = QGridLayout(box)

        def btn(row, col, text, fn, tip, span=1):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            g.addWidget(b, row, col, 1, span)
            return b

        btn(0, 0, 'Mirror about pushrod plane', self._do_mirror,
            'Mirror the actuation slice + ARB about the vertical plane through '
            'the pushrod line.  Exact isometry: all rates preserved.', 2)
        btn(0, 2, 'Rotate +', lambda: self._do_rotate(+1),
            'Rotate the slice about the pushrod line by +angle.  Isometry: '
            'MR preserved; ARB re-hung and re-tuned if needed.')
        btn(0, 3, 'Rotate -', lambda: self._do_rotate(-1),
            'Rotate the slice about the pushrod line by -angle.')
        for i, (lbl, ax) in enumerate((('X', 0), ('Y', 1), ('Z', 2))):
            btn(1, i + 1, f'{lbl} +', lambda _=0, a=ax: self._do_translate(a, +1),
                f'Translate the whole slice +step along {lbl}.  Changes the '
                'pushrod angle -> re-tune MR afterwards.')
            btn(2, i + 1, f'{lbl} -', lambda _=0, a=ax: self._do_translate(a, -1),
                f'Translate the whole slice -step along {lbl}.')
        btn(1, 0, 'Lever + 2%', lambda: self._do_lever(1.02),
            'Scale the rocker pushrod lever by 1.02 (raises/lowers MR).')
        btn(2, 0, 'Lever - 2%', lambda: self._do_lever(1 / 1.02),
            'Scale the rocker pushrod lever by 1/1.02.')
        btn(3, 0, 'Re-tune MR + ARB to baseline', self._do_retune,
            'Bisect the rocker lever until MR matches the baseline, then the '
            'drop-link radius until the ARB rate matches (the two physical '
            'knobs).', 2)
        btn(3, 2, 'Revert axle', self._do_revert,
            'Put this axle back to the captured baseline geometry.')
        btn(3, 3, 'Save experiment…', self._do_save,
            'Save the current model to configs/experiments/ (never configs/ '
            'root — the binder must not pick experiments up).')
        lay.addWidget(box)

        self._tol_box = self._build_tolerance_box()
        lay.addWidget(self._tol_box)

        self._readout = QTextEdit()
        self._readout.setReadOnly(True)
        self._readout.setStyleSheet(
            'QTextEdit { background: #141417; color: ' + _INK +
            '; font-family: Consolas; font-size: 9pt; }')
        lay.addWidget(self._readout, 1)
        return w

    def _build_tolerance_box(self) -> QGroupBox:
        box = QGroupBox('Tolerances (every held parameter)')
        box.setCheckable(True); box.setChecked(False)   # collapsed by default
        g = QGridLayout(box)
        self._tol_spins = {}
        fields = [(f, getattr(self._tol, f)) for f in
                  self._tol.__dataclass_fields__]
        for i, (name, val) in enumerate(fields):
            r, c = divmod(i, 4)
            g.addWidget(QLabel(name.replace('_', ' ')), r, c * 2)
            sp = QDoubleSpinBox()
            sp.setRange(0.0, 1e6); sp.setDecimals(3); sp.setValue(float(val))
            sp.valueChanged.connect(self._tol_changed)
            g.addWidget(sp, r, c * 2 + 1)
            self._tol_spins[name] = sp
        return box

    def _tol_changed(self):
        for name, sp in self._tol_spins.items():
            setattr(self._tol, name, float(sp.value()))
        self._refresh_readout()

    # ── helpers ───────────────────────────────────────────────────────────
    def _capture_baseline(self):
        try:
            self._baseline = pkg.capture_baseline(self._main)
            self._ref_bundles = {a: pkg.get_bundle(self._main, a)
                                 for a in ('front', 'rear')}
            self._refresh_readout()
            self._main.statusBar().showMessage(
                'Packaging baseline captured — transforms validate against '
                'THIS geometry now', 5000)
        except Exception as e:
            self._readout.setPlainText(f'Baseline capture failed: {e}')

    def _apply(self, bundle, refit=True):
        axle = self._axle.currentText()
        if refit:
            bundle = pkg.refit_arb(bundle, self._ref_bundles[axle])
        pkg.set_bundle(self._main, axle, bundle)
        self._main._update_3d()
        try:
            self._main._edit_sweep_timer.start()   # graphs follow (ONE MODEL)
        except Exception:
            pass
        self._refresh_readout()

    def _bundle(self):
        return pkg.get_bundle(self._main, self._axle.currentText())

    def _guard(self, fn, *a):
        try:
            return fn(*a)
        except Exception as e:
            self._main.statusBar().showMessage(f'Packaging: {e}', 6000)
            return None

    # ── transform slots ───────────────────────────────────────────────────
    def _do_mirror(self):
        b = self._guard(pkg.mirror_about_pushrod_plane, self._bundle())
        if b is not None:
            self._apply(b)

    def _do_rotate(self, sign):
        b = self._guard(pkg.rotate_about_pushrod_line, self._bundle(),
                        sign * float(self._angle.value()))
        if b is not None:
            self._apply(b)

    def _do_translate(self, axis, sign):
        d = np.zeros(3)
        d[axis] = sign * float(self._step.value()) / 1000.0
        b = self._guard(pkg.translate_actuation, self._bundle(), d)
        if b is not None:
            self._apply(b)

    def _do_lever(self, k):
        b = self._guard(pkg.scale_rocker_lever, self._bundle(), k)
        if b is not None:
            self._apply(b, refit=False)   # lever doesn't move the drop point

    def _do_retune(self):
        axle = self._axle.currentText()
        base = self._baseline
        if base is None:
            return
        def run():
            b = self._bundle()
            b, k, _ = pkg.retune_mr(self._main, b, axle,
                                    base['rates'][f'motion_ratio_{axle}'])
            pkg.set_bundle(self._main, axle, b)
            b, m, _ = pkg.retune_arb(self._main, b, axle,
                                     base['rates'][f'arb_rate_{axle}_Npm'],
                                     self._ref_bundles[axle])
            pkg.set_bundle(self._main, axle, b)
            return (k, m)
        out = self._guard(run)
        if out is not None:
            self._main.statusBar().showMessage(
                f'Re-tuned: lever x{out[0]:.4f}, drop radius x{out[1]:.4f}', 6000)
        self._main._update_3d()
        self._refresh_readout()

    def _do_revert(self):
        axle = self._axle.currentText()
        if self._ref_bundles:
            pkg.set_bundle(self._main, axle, self._ref_bundles[axle])
            self._main._update_3d()
            self._refresh_readout()

    def _do_save(self):
        out_dir = os.path.join('configs', 'experiments')
        os.makedirs(out_dir, exist_ok=True)
        fn = os.path.join(out_dir, f'manual_{time.strftime("%Y%m%d_%H%M%S")}.vahan')
        self._main._save_project_to_path(fn)
        self._main.statusBar().showMessage(f'Saved experiment: {fn}', 8000)

    # ── validity readout ──────────────────────────────────────────────────
    def _refresh_readout(self):
        if self._baseline is None:
            return
        try:
            res = pkg.validate(self._main, self._baseline, self._tol)
        except Exception as e:
            self._readout.setHtml(
                f'<span style="color:{_RED}">validate() failed: {e}</span>')
            return
        lines = []
        head_col = _INK if res.ok else _RED
        head = ('&#10003; ALL PARAMETERS HELD — valid packaging solution'
                if res.ok else
                f'&#10007; {len(res.failures())} CHECK(S) OUT OF TOLERANCE')
        lines.append(f'<b><span style="color:{head_col}">{head}</span></b>')
        for c in res.checks:
            mark = ('<span style="color:%s">&#10003;</span>' % _INK if c['ok']
                    else '<span style="color:%s">&#10007;</span>' % _RED)
            col = _GREY if c['ok'] else _RED
            if isinstance(c['value'], float) and isinstance(c['ref'], float):
                body = (f"{c['axle']:>5s}  {c['name']:<28s} "
                        f"{c['value']:+10.4f}  (baseline {c['ref']:+10.4f}, "
                        f"tol {c['tol']}{c['unit']})")
            else:
                body = f"{c['axle']:>5s}  {c['name']}"
            lines.append(f'{mark} <span style="color:{col}">{body}</span>')
        # clash detail
        for station, lst in (res.clashes or {}).items():
            for cl in lst:
                col = _RED if cl['gap_mm'] < 0 else _GREY
                lines.append(
                    f'&nbsp;&nbsp;<span style="color:{col}">{station}: '
                    f"{cl['corner']} {cl['a']} &harr; {cl['b']} "
                    f"gap {cl['gap_mm']:+.1f} mm</span>")
        self._readout.setHtml('<pre>' + '\n'.join(lines) + '</pre>')

    # ══════════════════════════════════════════════════════════════════════
    #  Generator tab
    # ══════════════════════════════════════════════════════════════════════
    def _build_generator_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        top = QHBoxLayout()
        top.addWidget(QLabel('Axle:'))
        self._gen_axle = QComboBox(); self._gen_axle.addItems(['front', 'rear'])
        top.addWidget(self._gen_axle)
        top.addWidget(QLabel('Solutions:'))
        self._gen_n = QSpinBox(); self._gen_n.setRange(1, 1000)
        self._gen_n.setValue(100)
        top.addWidget(self._gen_n)
        top.addWidget(QLabel('Seed:'))
        self._gen_seed = QSpinBox(); self._gen_seed.setRange(0, 99999)
        top.addWidget(self._gen_seed)
        self._gen_save = QCheckBox('Save configs')
        self._gen_save.setChecked(True)
        self._gen_save.setToolTip('Write each passing solution to '
                                  'configs/experiments/pkg_NNN.vahan + index JSON')
        top.addWidget(self._gen_save)
        top.addStretch(1)
        lay.addLayout(top)

        ko = QHBoxLayout()
        ko.addWidget(QLabel('Keep-out plane:  point Y ='))
        self._ko_y = QDoubleSpinBox()
        self._ko_y.setRange(-3000, 3000); self._ko_y.setValue(123.0)
        self._ko_y.setSuffix(' mm')
        self._ko_y.setToolTip('Default: the front-hoop plane at y = +123 mm. '
                              'Score = smallest clearance of any actuation/ARB '
                              'point to this plane (bigger = better packaged).')
        ko.addWidget(self._ko_y)
        ko.addWidget(QLabel('allowed side:'))
        self._ko_side = QComboBox()
        self._ko_side.addItems(['forward of it (-Y)', 'rearward of it (+Y)'])
        ko.addWidget(self._ko_side)
        run_btn = QPushButton('Run generator')
        run_btn.clicked.connect(self._run_generator)
        ko.addWidget(run_btn)
        cancel_btn = QPushButton('Cancel')
        cancel_btn.clicked.connect(self._cancel_generator)
        ko.addWidget(cancel_btn)
        ko.addStretch(1)
        lay.addLayout(ko)

        self._gen_bar = QProgressBar()
        self._gen_bar.setTextVisible(True)
        lay.addWidget(self._gen_bar)
        self._gen_status = QLabel('')
        lay.addWidget(self._gen_status)

        self._gen_table = QTableWidget(0, 8)
        self._gen_table.setHorizontalHeaderLabels(
            ['#', 'score mm', 'mirror', 'rotate deg', 'dx mm', 'dy mm',
             'dz mm', 'file'])
        self._gen_table.setSortingEnabled(True)
        self._gen_table.itemDoubleClicked.connect(self._load_solution)
        self._gen_table.setToolTip('Double-click a row to LOAD that solution '
                                   'into the 3-D view (whole project loads '
                                   'from its saved config).')
        lay.addWidget(self._gen_table, 1)
        return w

    def _cancel_generator(self):
        self._cancel = True

    def _run_generator(self):
        """Runs on the GUI thread on purpose: the generator drives the LIVE
        MainWindow solvers (apply candidate -> rebuild -> validate), and those
        objects are not thread-safe (the app's own background sweep worker
        deep-copies snapshots instead of sharing them).  processEvents() after
        every candidate keeps the UI responsive and the Cancel button live."""
        if self._baseline is None:
            self._capture_baseline()
        self._cancel = False
        n = int(self._gen_n.value())
        self._gen_bar.setRange(0, 0)     # busy until first progress call
        forward = self._ko_side.currentIndex() == 0
        ko = {'point_mm': [0.0, float(self._ko_y.value()), 0.0],
              'normal': [0.0, -1.0 if forward else 1.0, 0.0],
              'name': f'plane y={self._ko_y.value():+.0f} mm'}

        def prog(done, tried, found):
            self._gen_bar.setRange(0, n)
            self._gen_bar.setValue(min(found, n))
            self._gen_status.setText(f'tried {tried} candidates — '
                                     f'{found} valid so far')
            QApplication.processEvents()
            return not self._cancel

        try:
            result = pkg.generate_solutions(
                self._main, axle=self._gen_axle.currentText(), n_target=n,
                seed=int(self._gen_seed.value()), keep_out=ko, tol=self._tol,
                max_candidates=max(50, n * 25), progress=prog,
                save=self._gen_save.isChecked())
        except Exception as e:
            self._gen_status.setText(f'Generator failed: {e}')
            self._gen_bar.setRange(0, 1); self._gen_bar.setValue(0)
            return
        self._gen_bar.setRange(0, n)
        self._gen_bar.setValue(result['found'])
        fs = ', '.join(f'{k}: {v}' for k, v in result['fail_stage'].items())
        self._gen_status.setText(
            f"{result['found']} valid of {result['tried']} tried in "
            f"{result['elapsed_s']:.0f} s.  Failures — {fs or 'none'}")
        self._gen_table.setSortingEnabled(False)
        self._gen_table.setRowCount(len(result['solutions']))
        for i, s in enumerate(result['solutions']):
            r = s['recipe']
            vals = [i, round(s['score_mm'], 1), 'yes' if r['mirror'] else 'no',
                    round(r['rotate_deg'], 1),
                    round(r['translate_mm'][0], 1),
                    round(r['translate_mm'][1], 1),
                    round(r['translate_mm'][2], 1),
                    s.get('file', '(not saved)')]
            for j, v in enumerate(vals):
                it = QTableWidgetItem()
                if isinstance(v, (int, float)):
                    it.setData(Qt.ItemDataRole.DisplayRole, v)
                else:
                    it.setText(str(v))
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._gen_table.setItem(i, j, it)
        self._gen_table.setSortingEnabled(True)
        # the generator restores the pre-run geometry; refresh the picture
        self._main._update_3d()
        self._refresh_readout()

    def _load_solution(self, item):
        row = item.row()
        fit = self._gen_table.item(row, 7)
        path = fit.text() if fit else ''
        if not path or not os.path.exists(path):
            self._gen_status.setText('This solution was not saved to a file — '
                                     'enable "Save configs" and re-run.')
            return
        try:
            self._main._load_project_from_path(path)
            self._main._rebuild_solvers(0.)
            self._main._update_3d()
            self._gen_status.setText(f'Loaded {path}')
            self._refresh_readout()
        except Exception as e:
            self._gen_status.setText(f'Load failed: {e}')
