"""Loads — its own full-window page (like the Laptime / Design City pages).

Left:  inputs (load case, corner, resultant/components).
Right: a LIVE, hoverable 3-D Load view (a second View3D driven from the one
       solved model) on top, and the full every-point load table below.

Selecting a corner + case isolates that corner in the embedded 3-D view, draws
the force vectors, and lets you hover any arrow to read its load — exactly like
the HTML loads viewer, but in-page (no redirect to the suspension GUI).  The
'All corners' entry exits isolation in one click.  Inputs -> table -> picture
all read the SAME solved model (ONE MODEL).
"""
import numpy as np
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLabel, QComboBox,
                             QRadioButton, QButtonGroup, QPushButton, QTableWidget,
                             QTableWidgetItem, QHeaderView, QAbstractItemView, QSplitter,
                             QDoubleSpinBox, QGridLayout, QCheckBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from gui.wheel_package import (CASES, _load_items, load_arrows, case_speed_kph, case_aero,
                               case_aero_text)

# Colourblind-safe category text (docs/DESIGN.md): NO yellow/amber, NO blue.
# Distinguished by luminance + the Impeccable red on the tyre row; the category
# NAME already carries the identity, so colour is only a light grouping cue.
_CAT_COLOR = {'CHASSIS': QColor(236, 236, 238), 'UPRIGHT': QColor(205, 200, 196),
              'ROCKER': QColor(200, 200, 204), 'ARB': QColor(200, 200, 204),
              'TYRE': QColor(226, 59, 72)}

# corner selector entries ('All' -> no isolation)
_CORNERS = ['All corners', 'FL', 'FR', 'RL', 'RR']


class LoadsPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self._main = main
        self._v3d = None            # lazily built on first show (needs GL context)
        self.setStyleSheet('QWidget{background:#0b0b0d;color:#e6e6e6;}'
                           'QComboBox,QPushButton{background:#1e1e24;border:1px solid #444;'
                           'border-radius:4px;padding:4px 8px;}'
                           'QPushButton:hover{background:#2a2a32;}')
        root = QHBoxLayout(self); root.setContentsMargins(12, 12, 12, 12); root.setSpacing(14)

        # ── left: inputs ──
        left = QVBoxLayout(); left.setSpacing(8)
        def hdr(t):
            l = QLabel(t); l.setStyleSheet('font-weight:bold;color:#e0a83a;font-size:13px'); return l
        left.addWidget(hdr('LOAD CASE (g)'))
        # The user sets the exact g's to analyse — NOT a fixed dropdown.
        def _gspin(val):
            s = QDoubleSpinBox(); s.setRange(-4.0, 4.0); s.setDecimals(2)
            s.setSingleStep(0.1); s.setValue(val); s.setSuffix(' g')
            s.setStyleSheet('QDoubleSpinBox{background:#1e1e24;border:1px solid #444;'
                            'border-radius:3px;padding:2px 4px;}')
            return s
        self._lat_g = _gspin(1.40)     # +lateral (cornering)
        self._lon_g = _gspin(-1.00)    # -longitudinal (braking) / + = accel
        _gg = QGridLayout(); _gg.setContentsMargins(0, 0, 0, 0); _gg.setSpacing(4)
        _gg.addWidget(QLabel('Lateral'), 0, 0); _gg.addWidget(self._lat_g, 0, 1)
        _gg.addWidget(QLabel('Longitudinal'), 1, 0); _gg.addWidget(self._lon_g, 1, 1)
        # SPEED of the case (user 2026-09-29): the aero downforce on the corners
        # is Cl·A·½ρv², so every case is solved at a speed.  Auto = the case
        # rule (cornering: the speed for this lateral g on the Dynamics-panel
        # turn radius; straight line: the aero package's reference speed);
        # untick to type any speed.
        self._speed = QDoubleSpinBox(); self._speed.setRange(0.0, 400.0); self._speed.setDecimals(1)
        self._speed.setSingleStep(5.0); self._speed.setSuffix(' km/h')
        self._speed.setStyleSheet('QDoubleSpinBox{background:#1e1e24;border:1px solid #444;'
                                  'border-radius:3px;padding:2px 4px;}')
        _gg.addWidget(QLabel('Speed'), 2, 0); _gg.addWidget(self._speed, 2, 1)
        self._speed_auto = QCheckBox('Speed from the case rule'); self._speed_auto.setChecked(True)
        self._speed_auto.setToolTip('Cornering: the speed that gives this lateral g on the Dynamics-panel '
                                    'turn radius.  Straight line: the aero package reference speed.')
        _gg.addWidget(self._speed_auto, 3, 0, 1, 2)
        left.addLayout(_gg)
        self._aero_lab = QLabel('')
        self._aero_lab.setWordWrap(True); self._aero_lab.setStyleSheet('color:#9A9AA2;font-size:11px')
        left.addWidget(self._aero_lab)
        # quick-fill presets (they just POPULATE the editable g fields above)
        _pg = QGridLayout(); _pg.setContentsMargins(0, 2, 0, 0); _pg.setSpacing(3)
        for _i, (_nm, _la, _lo) in enumerate(
                (('2.0g corner', 2.0, 0.0), ('1.6g brake', 0.0, -1.6),
                 ('1.0g accel', 0.0, 1.0), ('corner+brake', 1.4, -1.0))):
            _b = QPushButton(_nm); _b.setStyleSheet('font-size:10px;padding:2px 3px;'
                'background:#1e1e24;border:1px solid #444;border-radius:3px;')
            _b.clicked.connect(lambda _c=False, la=_la, lo=_lo:
                               (self._lat_g.setValue(la), self._lon_g.setValue(lo)))
            _pg.addWidget(_b, _i // 2, _i % 2)
        left.addLayout(_pg)
        left.addWidget(QLabel('Corner'))
        self._corner = QComboBox(); self._corner.addItems(_CORNERS); left.addWidget(self._corner)
        left.addWidget(QLabel('3D vectors'))
        self._res = QRadioButton('Resultant'); self._comp = QRadioButton('Components (X/Y/Z)')
        self._res.setChecked(True)
        bg = QButtonGroup(self); bg.addButton(self._res); bg.addButton(self._comp)
        left.addWidget(self._res); left.addWidget(self._comp)

        # ── UPRIGHT / BRAKE geometry — inputs live HERE on the Loads page (not
        #    the dynamics panel).  They write back to the same stores the model
        #    already reads (LoadsPanel UprightParams + car rotor dia) so there is
        #    ONE model.  These drive the wheel-bearing and brake-caliper loads
        #    (Seward Ch.6): bearing spacing/offset = the (l1+l2)/l1 lever;
        #    caliper mount angle = where the caliper sits around the disc.
        left.addWidget(hdr('UPRIGHT / BRAKE GEOMETRY'))
        _lp = getattr(self._main, '_loads_panel', None)
        def _mkspin(lo, hi, val, suffix, dec, step):
            s = QDoubleSpinBox(); s.setRange(lo, hi); s.setDecimals(dec)
            s.setSingleStep(step); s.setSuffix(suffix); s.setValue(float(val))
            s.setStyleSheet('QDoubleSpinBox{background:#1e1e24;border:1px solid #444;'
                            'border-radius:3px;padding:2px 4px;}')
            return s
        _bs = getattr(_lp, '_brg_spacing', None); _co = getattr(_lp, '_brg_inboard', None)
        _ca = getattr(_lp, '_cal_angle', None)
        self._brg_spacing = _mkspin(1, 500, _bs.value() if _bs else 50.8, ' mm', 1, 2)
        self._brg_inboard = _mkspin(0, 500, _co.value() if _co else 39.4, ' mm', 1, 2)
        self._cal_angle   = _mkspin(0, 360, _ca.value() if _ca else 45.0, ' °', 0, 5)
        self._rotor_dia   = _mkspin(80, 400, float(self._main._car.get('rotor_dia_mm', 240.)), ' mm', 0, 5)
        # Seward caliper geometry (Fig 6.15): R_pad = pad centre radius,
        # l4 = pad-centre offset from the bolt line, l5 = bolt (lug) spacing.
        # These set F_brake = T/R_pad, V_brake = F_brake/2, H_brake = F_brake*l4/l5.
        _bd = self._brk_dict_read()
        self._pad_radius = _mkspin(10, 400, _bd.get('pad_radius', 94.4), ' mm', 1, 2)
        self._mount_h    = _mkspin(0, 300, _bd.get('mount_height', 27.9), ' mm', 1, 1)
        self._bolt_l5    = _mkspin(1, 400, _bd.get('bolt_spacing', 60.0), ' mm', 1, 2)
        _grid = QGridLayout(); _grid.setContentsMargins(0, 0, 0, 0); _grid.setSpacing(4)
        for r, (lab, wdg) in enumerate((('Bearing spacing  l1', self._brg_spacing),
                                        ('Bearing inboard offset', self._brg_inboard),
                                        ('Caliper mount angle', self._cal_angle),
                                        ('Rotor diameter', self._rotor_dia),
                                        ('Pad radius  R_pad', self._pad_radius),
                                        ('Caliper mount height', self._mount_h),
                                        ('Bolt spacing  l5', self._bolt_l5))):
            _grid.addWidget(QLabel(lab), r, 0); _grid.addWidget(wdg, r, 1)
        left.addLayout(_grid)
        for _w in (self._brg_spacing, self._brg_inboard, self._cal_angle, self._rotor_dia,
                   self._pad_radius, self._mount_h, self._bolt_l5):
            _w.valueChanged.connect(self._on_geom)

        self._note = QLabel('Hover any force arrow in the 3-D view to read its load.\n\n'
                            'CHASSIS = reaction into the frame pickups (tension pulls it '
                            'outboard).\nUPRIGHT = ball joints, wheel bearings (radial + axial), '
                            'caliper, tyre patch.\nROCKER/ARB = pushrod / spring / drop-link '
                            '(axial) + the rocker pivot (the only moment).')
        self._note.setWordWrap(True); self._note.setStyleSheet('color:#999;font-size:11px')
        left.addWidget(self._note); left.addStretch(1)
        lw = QWidget(); lw.setLayout(left); lw.setFixedWidth(270); root.addWidget(lw)

        # ── right: LIVE 3-D view (top) + outputs table (bottom) ──
        self._split = QSplitter(Qt.Orientation.Vertical)
        self._view_host = QWidget()
        self._view_host_lay = QVBoxLayout(self._view_host)
        self._view_host_lay.setContentsMargins(0, 0, 0, 0)
        self._view_ph = QLabel('3-D Load view — select a corner and load case.')
        self._view_ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._view_ph.setStyleSheet('color:#666;background:#000;')
        self._view_host_lay.addWidget(self._view_ph)
        self._split.addWidget(self._view_host)

        self._tbl = QTableWidget(0, 5)
        # Magnitude column mixes forces (N) and moments (N·m); the Type column
        # carries the unit per row, so the header names BOTH rather than lying (N).
        self._tbl.setHorizontalHeaderLabels(['Category', 'Point / member', 'Magnitude (N / N·m)',
                                             'Direction (lat / fore-aft / vert)', 'Type'])
        self._tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._tbl.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._tbl.setStyleSheet('QTableWidget{background:#111;gridline-color:#333;}'
                                'QHeaderView::section{background:#1e1e24;color:#e0a83a;'
                                'padding:5px;border:0;}')
        self._tbl.setSortingEnabled(True)
        self._split.addWidget(self._tbl)
        self._split.setStretchFactor(0, 3)
        self._split.setStretchFactor(1, 2)
        root.addWidget(self._split, stretch=1)

        self._corner.currentTextChanged.connect(self._on_input)
        for _s in (self._lat_g, self._lon_g):
            _s.valueChanged.connect(self._on_input)
        self._speed.valueChanged.connect(self._on_speed_edit)
        self._speed_auto.toggled.connect(self._on_input)
        self._res.toggled.connect(self._on_input)
        self.refresh()

    # ── corner selection -> isolation arg (None for 'All corners') ──
    def _corner_arg(self):
        c = self._corner.currentText()
        return None if c.startswith('All') else c

    def _case_g(self):
        return float(self._lat_g.value()), float(self._lon_g.value())

    def _sync_speed(self):
        """Auto mode: fill the speed box from the case rule (no signal storm)."""
        if not self._speed_auto.isChecked():
            return
        try:
            lat, lon = self._case_g()
            v = case_speed_kph(self._main, lat, lon)
        except Exception:
            return
        self._speed.blockSignals(True); self._speed.setValue(float(v)); self._speed.blockSignals(False)

    def _case_speed(self):
        """The speed (km/h) this page's loads are solved at: the rule speed in
        auto mode, else the typed one.  None is never passed on, so the table,
        the 3-D arrows and the readout all use the SAME speed."""
        self._sync_speed()
        return float(self._speed.value())

    def _speed_override(self):
        """The typed speed when it differs from the case rule, else None."""
        if self._speed_auto.isChecked():
            return None
        return float(self._speed.value())

    def _on_speed_edit(self, *_):
        if self._speed_auto.isChecked():
            self._speed_auto.blockSignals(True); self._speed_auto.setChecked(False)
            self._speed_auto.blockSignals(False)
        self.refresh()

    def _refresh_aero_label(self):
        try:
            lat, lon = self._case_g()
            ca = case_aero(self._main, lat, lon, self._case_speed())
            self._aero_lab.setText('Solved at ' + case_aero_text(ca))
        except Exception as e:
            self._aero_lab.setText(f'Speed / aero: {e}')

    def _ensure_view(self):
        """Build the embedded View3D on first use (its GL canvas needs a running
        QApplication + a shown window, so we defer it out of __init__)."""
        if self._v3d is not None:
            return
        try:
            from gui.view3d import View3D
            self._v3d = View3D()
            # wire the embedded view's navcube controls (perspective / floor /
            # thickness) — without this they did nothing on the Loads page.
            self._v3d.set_on_view_controls(self._on_v3d_controls)
            self._view_host_lay.removeWidget(self._view_ph)
            self._view_ph.hide()
            self._view_host_lay.addWidget(self._v3d.native)
        except Exception:
            self._v3d = None

    def _on_v3d_controls(self, d):
        """Navcube controls on the EMBEDDED loads view (perspective/floor/thickness)."""
        v = self._v3d
        if v is None:
            return
        try:
            if 'perspective' in d:
                v.set_perspective(bool(d['perspective']))
            if 'floor' in d and hasattr(v, '_ground'):
                v._ground.visible = bool(d['floor'])
            if 'thickness' in d:
                # route through _car so build_load_view keeps it on the next rebuild
                self._main._car['show_shock_thickness'] = bool(d['thickness'])
                self._refresh_3d()
            v._canvas.update()
        except Exception:
            pass

    def _brk_dicts(self):
        """LoadsPanel brake widget dict(s) for the selected corner's axle
        (both axles when 'All corners' is selected)."""
        lp = getattr(self._main, '_loads_panel', None)
        if lp is None:
            return []
        c = self._corner.currentText() if hasattr(self, '_corner') else 'All'
        f = getattr(lp, '_brk_f', None); r = getattr(lp, '_brk_r', None)
        if c.startswith('F'):
            return [d for d in (f,) if d]
        if c.startswith('R'):
            return [d for d in (r,) if d]
        return [d for d in (f, r) if d]

    def _brk_dict_read(self):
        ds = self._brk_dicts()
        if not ds:
            return {}
        d = ds[0]
        return {k: d[k].value() for k in ('pad_radius', 'mount_height', 'bolt_spacing', 'rotor_dia')
                if k in d}

    def _sync_brk_inputs(self):
        """Reload the caliper-geometry spinboxes from the selected axle."""
        d = self._brk_dict_read()
        for key, w in (('pad_radius', getattr(self, '_pad_radius', None)),
                       ('mount_height', getattr(self, '_mount_h', None)),
                       ('bolt_spacing', getattr(self, '_bolt_l5', None))):
            if key in d and w is not None:
                w.blockSignals(True); w.setValue(d[key]); w.blockSignals(False)

    def _on_input(self, *_):
        self._sync_brk_inputs()
        self.refresh()

    def _on_geom(self, *_):
        """Bearing/caliper/rotor geometry changed on the Loads page → write back
        into the SAME stores the load model reads (LoadsPanel UprightParams +
        car rotor dia) so there is no second model, then recompute."""
        lp = getattr(self._main, '_loads_panel', None)
        if lp is not None:
            for src, name in ((self._brg_spacing, '_brg_spacing'),
                              (self._brg_inboard, '_brg_inboard'),
                              (self._cal_angle, '_cal_angle')):
                dst = getattr(lp, name, None)
                if dst is not None:
                    dst.blockSignals(True); dst.setValue(src.value()); dst.blockSignals(False)
        # caliper geometry -> the LoadsPanel brake widgets for the selected axle
        for d in self._brk_dicts():
            for key, src in (('pad_radius', self._pad_radius),
                             ('mount_height', self._mount_h),
                             ('bolt_spacing', self._bolt_l5)):
                w = d.get(key)
                if w is not None:
                    w.blockSignals(True); w.setValue(src.value()); w.blockSignals(False)
        try:
            self._main._car['rotor_dia_mm'] = float(self._rotor_dia.value())
            cp = getattr(getattr(self._main, '_car_panel', None), '_rotor_dia', None)
            if cp is not None:
                cp.blockSignals(True); cp.setValue(float(self._rotor_dia.value())); cp.blockSignals(False)
        except Exception:
            pass
        self.refresh()

    def refresh(self, *_):
        self._refresh_aero_label()
        self._refresh_table()
        self._refresh_3d()

    def _refresh_table(self):
        try:
            lat, lon = self._case_g()
            items = _load_items(self._main, lat, lon, only_corner=self._corner_arg(),
                                speed_kph=self._case_speed())
        except Exception as e:
            self._tbl.setRowCount(1)
            self._tbl.setItem(0, 0, QTableWidgetItem(f'Error: {e}'))
            return
        self._tbl.setSortingEnabled(False)
        self._tbl.setRowCount(len(items))
        for r, (p, v, col, lab) in enumerate(items):
            parts = lab.split(' · ')
            cat = parts[0].split(' ', 1)[-1] if ' ' in parts[0] else parts[0]
            point = parts[1] if len(parts) > 1 else ''
            mag = float(np.linalg.norm(v))
            is_moment = 'N·m' in lab
            typ = ''
            if is_moment:
                typ = 'MOMENT (N·m)'
            elif 'tension' in lab:
                typ = 'tension'
            elif 'compression' in lab:
                typ = 'compression'
            elif 'AXIAL' in lab or 'axial' in lab:
                typ = 'axial'
            elif 'RADIAL' in lab:
                typ = 'radial'
            elif 'PIVOT' in lab or 'pivot' in lab:
                typ = 'moment reaction'
            cells = [cat, point, f'{mag:,.0f}',
                     f'{v[0]:+,.0f} / {v[1]:+,.0f} / {v[2]:+,.0f}', typ]
            magitem = QTableWidgetItem()
            magitem.setData(Qt.ItemDataRole.DisplayRole, round(mag))   # numeric sort
            for cndx, txt in enumerate(cells):
                it = magitem if cndx == 2 else QTableWidgetItem(str(txt))
                if cndx == 0:
                    it.setForeground(_CAT_COLOR.get(cat, QColor(220, 220, 220)))
                self._tbl.setItem(r, cndx, it)
        self._tbl.setSortingEnabled(True)
        self._tbl.sortItems(2, Qt.SortOrder.DescendingOrder)

    def _refresh_3d(self):
        """Drive the embedded 3-D view into Load mode for this corner + case."""
        self._ensure_view()
        if self._v3d is None:
            return
        try:
            lat, lon = self._case_g()
            vec = 'components' if self._comp.isChecked() else 'resultant'
            self._main.build_load_view(self._v3d, self._corner_arg(), lat, lon, vec)
            # build_load_view draws the arrows at the case-rule speed; a typed
            # speed re-draws them at that speed so picture == table.
            v = self._speed_override()
            if v is not None:
                self._v3d.set_load_vectors(load_arrows(self._main, lat, lon, mode=vec,
                                                       only_corner=self._corner_arg(), speed_kph=v))
        except Exception:
            pass
