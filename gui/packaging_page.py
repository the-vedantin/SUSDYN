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
        tabs.addTab(self._build_relocate_tab(), 'Relocate (IK)')
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

    # ══════════════════════════════════════════════════════════════════════
    #  Relocate (IK) tab — curve-preserving single-point relocation
    # ══════════════════════════════════════════════════════════════════════
    def _build_relocate_tab(self):
        from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                     QComboBox, QDoubleSpinBox, QSpinBox,
                                     QPushButton, QTableWidget, QScrollArea,
                                     QFrame)
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel(
            'Pick ONE hardpoint - find WHERE ELSE it can live with every '
            'kinematic curve within tolerance of the original (the tolerances '
            'on the Manual tab apply). Solutions are shown at least the '
            'similarity setting apart (packaging distance only); select one '
            'and "Explore near selected" reveals the finer solutions inside '
            'its neighbourhood.'))
        top = QHBoxLayout()
        top.addWidget(QLabel('Axle:'))
        self._rl_axle = QComboBox(); self._rl_axle.addItems(['front', 'rear'])
        self._rl_axle.currentTextChanged.connect(self._rl_fill_points)
        top.addWidget(self._rl_axle)
        top.addWidget(QLabel('Point:'))
        self._rl_point = QComboBox(); top.addWidget(self._rl_point, 1)
        top.addWidget(QLabel('Radius:'))
        self._rl_radius = QDoubleSpinBox()
        self._rl_radius.setRange(5, 500); self._rl_radius.setValue(120)
        self._rl_radius.setSuffix(' mm'); top.addWidget(self._rl_radius)
        top.addWidget(QLabel('Similarity:'))
        self._rl_sim = QDoubleSpinBox()
        self._rl_sim.setRange(1, 90); self._rl_sim.setValue(20)
        self._rl_sim.setSuffix(' %')
        self._rl_sim.setToolTip('No two shown solutions closer than this '
                                'fraction of the largest feasible extent.')
        top.addWidget(self._rl_sim)
        top.addWidget(QLabel('Samples:'))
        self._rl_n = QSpinBox(); self._rl_n.setRange(20, 2000)
        self._rl_n.setValue(150); top.addWidget(self._rl_n)
        lay.addLayout(top)

        trow = QHBoxLayout()
        trow.addWidget(QLabel('Target:'))
        self._rl_tgt = []
        for ax in ('X', 'Y', 'Z'):
            trow.addWidget(QLabel(ax))
            sp = QDoubleSpinBox()
            sp.setRange(-3000.0, 3000.0); sp.setDecimals(2)
            sp.setSingleStep(1.0); sp.setSuffix(' mm')
            trow.addWidget(sp)
            self._rl_tgt.append(sp)
        trow.addStretch(1)
        lay.addLayout(trow)

        # Buttons on their own row — keeping them in the target row forced a
        # ~1900 px minimum width onto the whole MainWindow (scroll/clipping
        # bug): a QHBoxLayout's minimum is the SUM of its children's minimums.
        brow = QHBoxLayout()
        self._rl_test = QPushButton('Test this position')
        self._rl_test.setToolTip('Put the point EXACTLY at the target, run the '
                                 'full oracle (curves + laws + rates + clash), '
                                 'report PASS/FAIL, put it back.')
        self._rl_test.clicked.connect(self._rl_test_target)
        brow.addWidget(self._rl_test)
        self._rl_apply_test = QPushButton('Apply tested position')
        self._rl_apply_test.setEnabled(False)
        self._rl_apply_test.setToolTip('Install the last PASS-tested position '
                                       'WITH its resolved chain (rocker plane, '
                                       'axis, ARB, rates).')
        self._rl_apply_test.clicked.connect(self._rl_apply_test_bundle)
        brow.addWidget(self._rl_apply_test)
        self._rl_go_tgt = QPushButton('Search near target')
        self._rl_go_tgt.setToolTip('Find the valid positions CLOSEST to the '
                                   'target (works even if the exact target is '
                                   'infeasible - searches from the nearest '
                                   'valid approach).')
        self._rl_go_tgt.clicked.connect(self._rl_search_target)
        brow.addWidget(self._rl_go_tgt)
        brow.addStretch(1)
        lay.addLayout(brow)

        row2 = QHBoxLayout()
        self._rl_run = QPushButton('Search around current')
        self._rl_run.clicked.connect(lambda: self._rl_search(None))
        row2.addWidget(self._rl_run)
        self._rl_drill = QPushButton('Explore near selected')
        self._rl_drill.setToolTip('Re-search INSIDE the selected neighbourhood '
                                  '- the finer solutions the similarity filter '
                                  'hid.')
        self._rl_drill.clicked.connect(self._rl_drill_down)
        row2.addWidget(self._rl_drill)
        self._rl_apply = QPushButton('Apply selected')
        self._rl_apply.clicked.connect(self._rl_apply_sel)
        row2.addWidget(self._rl_apply)
        self._rl_revert = QPushButton('Revert point')
        self._rl_revert.clicked.connect(self._rl_revert_pt)
        row2.addWidget(self._rl_revert)
        row2.addStretch(1)
        lay.addLayout(row2)

        self._rl_status = QLabel('')
        self._rl_status.setWordWrap(True)
        lay.addWidget(self._rl_status)
        self._rl_table = QTableWidget(0, 7)
        self._rl_table.setHorizontalHeaderLabels(
            ['x (mm)', 'y (mm)', 'z (mm)', 'dist from OG (mm)',
             'dist from TARGET (mm)', 'worst tol used', 'clash'])
        lay.addWidget(self._rl_table, 1)
        self._rl_result = None
        self._rl_orig = None            # (axle, dict_name, key, orig pos mm)
        self._rl_fill_points()
        # Scroll container: keeps this tab's content from dictating the
        # MainWindow minimum size — anything that doesn't fit scrolls.
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QFrame.Shape.NoFrame)
        sa.setWidget(w)
        return sa

    def _rl_fill_points(self):
        import vahan.packaging as pkg
        axle = self._rl_axle.currentText()
        b = pkg.get_bundle(self._main, axle)
        self._rl_point.clear()
        for k in b['hp']:
            self._rl_point.addItem('hp.' + k)
        for k in b['arb']:
            self._rl_point.addItem('arb.' + k)
        try:
            self._rl_point.currentTextChanged.disconnect(self._rl_seed_target)
        except Exception:
            pass
        self._rl_point.currentTextChanged.connect(self._rl_seed_target)
        self._rl_seed_target()

    def _rl_seed_target(self, *_a):
        """Fill the target boxes with the point CURRENT position, so the
        user edits from where it is, not from zero."""
        import vahan.packaging as pkg
        txt = self._rl_point.currentText()
        if '.' not in txt:
            return
        dict_name, key = txt.split('.', 1)
        b = pkg.get_bundle(self._main, self._rl_axle.currentText())
        p = b[dict_name].get(key)
        if p is None:
            return
        for sp, v in zip(self._rl_tgt, p):
            sp.blockSignals(True)
            sp.setValue(float(v) * 1000.0)
            sp.blockSignals(False)

    def _rl_target_mm(self):
        return [sp.value() for sp in self._rl_tgt]

    def _rl_test_target(self):
        from vahan.relocate import test_position
        axle = self._rl_axle.currentText()
        dict_name, key = self._rl_point.currentText().split('.', 1)
        self._rl_status.setText('resolving chain + testing position...')
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()
        if self._baseline is None:
            self._capture_baseline()
        r = test_position(self._main, axle, dict_name, key,
                          self._rl_target_mm(),
                          baseline=self._baseline, tol=self._tol)
        t = tuple(self._rl_target_mm())
        self._rl_test_bundle = r.get('bundle')
        self._rl_test_meta = (axle,)
        if r['ok']:
            extra = ''
            info = r.get('info', {})
            if 'chain_rotation_deg' in info:
                extra = (' Chain re-solved: rotated %.1f deg%s%s.'
                         % (info['chain_rotation_deg'],
                            ', MR re-tuned' if info.get('mr_retuned') else '',
                            ', ARB re-tuned' if info.get('arb_retuned') else ''))
            self._rl_status.setText(
                'PASS - (%.1f, %.1f, %.1f) mm WORKS for %s %s.%s: coplanarity/'
                'axis/triad/rates all re-solved, curves in tolerance, no clash '
                'across travel.%s  Click "Apply tested position" to take it.'
                % (t[0], t[1], t[2], axle, dict_name, key, extra))
            self._rl_apply_test.setEnabled(True)
        else:
            why = '; '.join(str(f) for f in r['fails'][:4])
            note = (' (chain resolution attempted)' if r.get('resolvable')
                    else '')
            self._rl_status.setText(
                'FAIL - (%.1f, %.1f, %.1f) mm cannot work%s: %s. '
                'Use Search near target for the closest position that does.'
                % (t[0], t[1], t[2], note, why))
            self._rl_apply_test.setEnabled(False)

    def _rl_apply_test_bundle(self):
        import numpy as np
        import vahan.packaging as pkg
        b = getattr(self, '_rl_test_bundle', None)
        if not b:
            self._rl_status.setText('no tested PASS position to apply - '
                                    'run Test this position first')
            return
        axle = self._rl_test_meta[0]
        bundle = {'hp': {k: np.array(v, float) for k, v in b['hp'].items()},
                  'arb': {k: np.array(v, float) for k, v in b['arb'].items()}}
        pkg.set_bundle(self._main, axle, bundle)
        self._main._update_3d()
        self._rl_status.setText('APPLIED the tested position with its resolved '
                                'chain (%s axle). Revert axle is on the Manual '
                                'tab; or re-load the config.' % axle)

    def _rl_search_target(self):
        from PyQt6.QtWidgets import QApplication, QTableWidgetItem
        from vahan.relocate import generative_search
        axle = self._rl_axle.currentText()
        dict_name, key = self._rl_point.currentText().split('.', 1)
        self._rl_go_tgt.setEnabled(False)
        self._rl_status.setText('growing toward target (generative)...')

        def prog(msg):
            self._rl_status.setText(msg)
            QApplication.processEvents()

        try:
            if self._baseline is None:
                self._capture_baseline()
            if dict_name == 'hp' and key == 'spring_chassis_pt':
                # packaging MAP-Elites: fills a DIVERSE archive of rocker/ARB
                # assemblies with the mount PINNED at target (curves exact,
                # dynamics <= tol).  Far better than the tree search here.
                from vahan.packaging_qd import qd_relocate_search
                r = qd_relocate_search(
                    self._main, axle, dict_name, key, self._rl_target_mm(),
                    baseline=self._baseline, tol=self._tol,
                    n_init=int(self._rl_n.value()),
                    n_iter=int(self._rl_n.value()) * 2,
                    min_sep_frac=float(self._rl_sim.value()) / 100.0,
                    progress=prog)
            else:
                r = generative_search(
                    self._main, axle, dict_name, key, self._rl_target_mm(),
                    baseline=self._baseline, tol=self._tol,
                    n_iter=int(self._rl_n.value()) * 2,
                    step_mm=8.0, n_show=10,
                    min_sep_frac=float(self._rl_sim.value()) / 100.0,
                    progress=prog)
        except Exception as e:
            self._rl_status.setText('ERROR: %r' % (e,))
            self._rl_go_tgt.setEnabled(True)
            return
        finally:
            self._main._update_3d()
        self._rl_go_tgt.setEnabled(True)
        self._rl_result = r
        if 'error' in r:
            self._rl_status.setText(r['error'])
            self._rl_table.setRowCount(0)
            return
        self._rl_orig = (axle, dict_name, key, r['original_mm'])
        n_sol = len(r['solutions'])
        # label the searcher that actually ran (QD packaging vs generative tree)
        tag = ('PACKAGING QD' if 'MAP-Elites' in r.get('method', '')
               else 'GENERATIVE')
        reach = ('%d solutions' % n_sol if n_sol else
                 ('TARGET REACHED but 0 survive the full oracle'
                  if r.get('target_reached') else
                  'closest approach %.1f mm from target'
                  % r['best_approach_mm']))
        # When solutions are zero, name the DOMINANT binding constraint so the
        # result teaches instead of a silent 0.  reject_reasons maps a failed
        # check label -> count across the rejected candidates.
        binding = ''
        rr = r.get('reject_reasons') or {}
        if n_sol == 0 and rr:
            top = max(rr, key=rr.get)
            _law = {'front coplanar_mm': ' = the NO-BENDING law (chain leaves '
                    'its plane across travel; this target forces the pushrod '
                    'to bend)',
                    'rear coplanar_mm': ' = the NO-BENDING law',
                    'both clash sweep (static/bump/droop)': ' = interference '
                    'across travel'}.get(top, '')
            binding = ('  |  BINDING: %s%s  — the target itself is infeasible '
                       'here, not the search. Relax the tolerance, move the '
                       'target, or use "Search around current" to see where it '
                       'CAN go.' % (top, _law))
        self._rl_status.setText(
            '%s: %s - %d evals over %.0f mm span, %d final rejects%s'
            % (tag, reach, r['tree_size'], r['span_mm'],
               r['n_final_rejects'], binding))
        t = self._rl_table
        t.setRowCount(len(r['solutions']))
        for i, s2 in enumerate(r['solutions']):
            vals = ['%.2f' % s2['pos_mm'][0], '%.2f' % s2['pos_mm'][1],
                    '%.2f' % s2['pos_mm'][2], '%.1f' % s2['dist_from_og_mm'],
                    '%.1f' % s2['dist_from_target_mm'],
                    '%.0f %%' % (s2['worst_tol_frac'] * 100),
                    'clean' if s2['clash_checked'] else '?']
            for j, v in enumerate(vals):
                t.setItem(i, j, QTableWidgetItem(v))

    def _rl_search(self, center_mm, radius_override=None, target_mm=None):
        from PyQt6.QtWidgets import QApplication, QTableWidgetItem
        from vahan.relocate import relocate_search
        axle = self._rl_axle.currentText()
        dict_name, key = self._rl_point.currentText().split('.', 1)
        self._rl_run.setEnabled(False)
        self._rl_status.setText('searching...')

        def prog(msg):
            self._rl_status.setText(msg)
            QApplication.processEvents()

        try:
            r = relocate_search(
                self._main, axle, dict_name, key,
                baseline=self._baseline, tol=self._tol,
                radius_mm=float(radius_override or self._rl_radius.value()),
                n_samples=int(self._rl_n.value()),
                min_sep_frac=float(self._rl_sim.value()) / 100.0,
                center_mm=center_mm, target_mm=target_mm, progress=prog)
        except Exception as e:
            self._rl_status.setText('ERROR: %r' % (e,))
            self._rl_run.setEnabled(True)
            return
        finally:
            self._main._update_3d()
        self._rl_run.setEnabled(True)
        self._rl_result = r
        if 'error' in r:
            self._rl_status.setText(r['error'])
            self._rl_table.setRowCount(0)
            return
        self._rl_orig = (axle, dict_name, key, r['original_mm'])
        bind = ''
        if r.get('clash_binding'):
            worst = max(r['clash_binding'], key=r['clash_binding'].get)
            bind = '  |  binding clash: ' + worst
        if r.get('target_note'):
            bind += '  |  ' + r['target_note']
        self._rl_status.setText(
            '%s: %d solutions (>=%.1f mm apart) from %d feasible of %d tried'
            ' - max extent %.1f mm, %d clash-rejected%s'
            % (r['point'], len(r['solutions']), r['min_sep_mm'],
               r['n_feasible'], r['n_tried'], r['max_extent_mm'],
               r['n_clash_rejected'], bind))
        t = self._rl_table
        t.setRowCount(len(r['solutions']))
        for i, s in enumerate(r['solutions']):
            dt = ('%.1f' % s['dist_from_target_mm']
                  if 'dist_from_target_mm' in s else '-')
            vals = ['%.2f' % s['pos_mm'][0], '%.2f' % s['pos_mm'][1],
                    '%.2f' % s['pos_mm'][2], '%.1f' % s['dist_from_og_mm'],
                    dt, '%.0f %%' % (s['worst_tol_frac'] * 100),
                    'clean' if s['clash_checked'] else '?']
            for j, v in enumerate(vals):
                t.setItem(i, j, QTableWidgetItem(v))

    def _rl_selected_pos(self):
        r = self._rl_result
        row = self._rl_table.currentRow()
        if not r or 'solutions' not in r or row < 0 \
                or row >= len(r['solutions']):
            return None
        return r['solutions'][row]['pos_mm']

    def _rl_point_dict(self, axle, dict_name):
        if dict_name == 'hp':
            return (self._main._front_hp if axle == 'front'
                    else self._main._rear_hp)
        return (self._main._front_arb if axle == 'front'
                else self._main._rear_arb)

    def _rl_drill_down(self):
        pos = self._rl_selected_pos()
        if pos is None:
            self._rl_status.setText('select a solution row first')
            return
        self._rl_search(pos, radius_override=self._rl_result['min_sep_mm'])

    def _rl_apply_sel(self):
        import numpy as np
        import vahan.packaging as pkg
        r = self._rl_result
        row = self._rl_table.currentRow()
        if not r or 'solutions' not in r or row < 0 \
                or row >= len(r['solutions']):
            self._rl_status.setText('select a solution row first')
            return
        sol = r['solutions'][row]
        axle, dict_name, key, _ = self._rl_orig
        if sol.get('bundle'):
            bundle = {'hp': {k: np.array(v, float)
                             for k, v in sol['bundle']['hp'].items()},
                      'arb': {k: np.array(v, float)
                              for k, v in sol['bundle']['arb'].items()}}
            pkg.set_bundle(self._main, axle, bundle)
            self._main._update_3d()
            self._rl_status.setText(
                'applied solution %d WITH its resolved chain (%s %s.%s at '
                '%.1f, %.1f, %.1f mm). Manual tab Revert axle undoes it.'
                % (row + 1, axle, dict_name, key, *sol['pos_mm']))
            return
        pos = sol['pos_mm']
        d = self._rl_point_dict(axle, dict_name)
        d[key] = np.array(pos, float) / 1000.0
        self._main._rebuild_solvers()
        self._main._update_3d()
        self._rl_status.setText(
            'applied %s %s.%s -> (%.1f, %.1f, %.1f) mm - Revert point undoes '
            'this.' % (axle, dict_name, key, pos[0], pos[1], pos[2]))

    def _rl_revert_pt(self):
        import numpy as np
        if self._rl_orig is None:
            return
        axle, dict_name, key, orig_mm = self._rl_orig
        d = self._rl_point_dict(axle, dict_name)
        d[key] = np.array(orig_mm, float) / 1000.0
        self._main._rebuild_solvers()
        self._main._update_3d()
        self._rl_status.setText('reverted %s %s.%s to original'
                                % (axle, dict_name, key))
