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
                             QDoubleSpinBox, QGridLayout, QCheckBox, QSlider, QDialog,
                             QApplication, QScrollArea)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QPointF
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush, QPolygonF

from gui.wheel_package import (CASES, _load_items, load_arrows, case_speed_kph, case_aero,
                               case_aero_text, angle_text, gg_envelope_at, gg_border_point,
                               gg_worst_loads, GG_JOINTS)

# Colourblind-safe category text (docs/DESIGN.md): NO yellow/amber, NO blue.
# Distinguished by luminance + the Impeccable red on the tyre row; the category
# NAME already carries the identity, so colour is only a light grouping cue.
_CAT_COLOR = {'CHASSIS': QColor(236, 236, 238), 'UPRIGHT': QColor(205, 200, 196),
              'ROCKER': QColor(200, 200, 204), 'ARB': QColor(200, 200, 204),
              'TYRE': QColor(226, 59, 72)}

# corner selector entries ('All' -> no isolation)
_CORNERS = ['All corners', 'FL', 'FR', 'RL', 'RR']

# plain names for the G-G worst-loads table
_GG_NAMES = {'uca_front_N': 'upper arm front leg', 'uca_rear_N': 'upper arm rear leg',
             'lca_front_N': 'lower arm front leg', 'lca_rear_N': 'lower arm rear leg',
             'tierod_N': 'tie rod / toe link', 'pushrod_N': 'pushrod',
             'spring_damper_force_N': 'spring + damper', 'arb_link_N': 'anti-roll-bar drop link',
             'bearing_axial_N': 'wheel bearing, sideways (axial)'}
for _j, _nm in GG_JOINTS:
    _GG_NAMES[f'{_j}_total'] = f'{_nm}, total'
    for _ax in ('lateral', 'fore-aft', 'vertical'):
        _GG_NAMES[f'{_j}_{_ax}'] = f'{_nm}, {_ax} part'


def _gg_dir(corner, item, v):
    """Which way a G-G worst-load row points, in words.  Members: tension / compression
    along the member; joint parts: the force ON the upright, outboard / inboard,
    forward / rearward, up / down (world +x = car left, +y = rearward, +z = up)."""
    if not np.isfinite(v) or abs(v) < 0.5:
        return ''
    if item == 'spring_damper_force_N':
        return 'compression' if v > 0 else 'tension'
    if item == 'bearing_axial_N' or item.endswith('_total'):
        return ''
    if item.endswith('_lateral'):
        return 'outboard' if (v > 0) == (str(corner)[1:2] == 'L') else 'inboard'
    if item.endswith('_fore-aft'):
        return 'rearward' if v > 0 else 'forward'
    if item.endswith('_vertical'):
        return 'up' if v > 0 else 'down'
    return 'tension' if v > 0 else 'compression'


class _GGWorker(QThread):
    """Builds the G-G envelope off the UI thread (~250 full solves)."""
    done = pyqtSignal(object)

    def __init__(self, main, speed_kph, engine_limit=False):
        super().__init__()
        self._main, self._v, self._eng = main, float(speed_kph), bool(engine_limit)

    def run(self):
        try:
            self.done.emit(gg_envelope_at(self._main, self._v, engine_limit=self._eng))
        except Exception as e:                      # reported in the readout
            self.done.emit(e)


class _GGPlot(QWidget):
    """Small G-G picture: the envelope border (white) and the chosen point
    (yellow).  Lateral across, forward g up, braking down."""
    def __init__(self):
        super().__init__()
        self.setFixedHeight(190)
        self._border, self._pt, self._lim = None, None, 2.5

    def set_border(self, border):
        self._border = None if border is None else np.asarray(border, float)
        if self._border is not None and np.all(np.isfinite(self._border)):
            self._lim = max(1.0, float(np.nanmax(np.abs(self._border))) * 1.12)
        self.update()

    def set_point(self, lat, lon):
        self._pt = (float(lat), float(lon)); self.update()

    def paintEvent(self, _e):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, QColor('#111114'))
        s = (min(W, H) / 2.0 - 14.0) / self._lim
        cx, cy = W / 2.0, H / 2.0
        def xy(la, lo):
            return QPointF(cx + la * s, cy - lo * s)
        p.setPen(QPen(QColor('#3a3a42'), 1))
        p.drawLine(QPointF(cx - self._lim * s, cy), QPointF(cx + self._lim * s, cy))
        p.drawLine(QPointF(cx, cy - self._lim * s), QPointF(cx, cy + self._lim * s))
        p.setPen(QColor('#9A9AA2'))
        p.drawText(QPointF(cx + 4, 12), 'accelerate'); p.drawText(QPointF(cx + 4, H - 4), 'brake')
        p.drawText(QPointF(W - 48, cy - 4), 'corner')
        if self._border is not None and len(self._border) > 2:
            poly = QPolygonF([xy(a, b) for a, b in self._border if np.isfinite(a) and np.isfinite(b)])
            p.setPen(QPen(QColor('#f2f2f2'), 2)); p.setBrush(Qt.BrushStyle.NoBrush); p.drawPolygon(poly)
        if self._pt is not None and all(np.isfinite(self._pt)):
            p.setPen(QPen(QColor('#f2c94c'), 1.5))
            p.drawLine(QPointF(cx, cy), xy(*self._pt))
            p.setBrush(QBrush(QColor('#f2c94c'))); p.drawEllipse(xy(*self._pt), 5, 5)
        p.end()


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
                 ('1.6g accel', 0.0, 1.6), ('corner+brake', 1.4, -1.0))):
            _b = QPushButton(_nm); _b.setStyleSheet('font-size:10px;padding:2px 3px;'
                'background:#1e1e24;border:1px solid #444;border-radius:3px;')
            _b.clicked.connect(lambda _c=False, la=_la, lo=_lo:
                               (self._lat_g.setValue(la), self._lon_g.setValue(lo)))
            _pg.addWidget(_b, _i // 2, _i % 2)
        left.addLayout(_pg)

        # ── G-G ENVELOPE (user 2026-10-21): slide the case around the border of
        #    the car's own predicted G-G envelope (with aero at the speed above)
        #    instead of typing g's — every point is a combination the tyres can
        #    actually carry at that speed (engine cap optional).
        left.addWidget(hdr('G-G ENVELOPE (with aero)'))
        self._gg_on = QCheckBox('Pick the case on the G-G border')
        self._gg_on.setToolTip("Solves the car's G-G envelope at the Speed above (aero at that speed,\n"
                               'tyre-limited all round unless the engine box below is ticked) and puts\n'
                               'the load case on its border.\n'
                               '0° = straight braking, 90° = pure cornering, 180° = straight acceleration;\n'
                               'negative angles = the turn the other way.')
        left.addWidget(self._gg_on)
        # Tyre-limited by default (user 2026-10-21): the load envelope is what the
        # tyres can carry; tick to cap the forward side with the engine instead.
        self._gg_engine = QCheckBox('Engine limit on the forward side')
        self._gg_engine.setToolTip('Off (default): forward g is what the driven tyres can put down.\n'
                                   'On: capped by the engine drive force at this speed.')
        self._gg_engine.setChecked(False)
        left.addWidget(self._gg_engine)
        self._gg_slider = QSlider(Qt.Orientation.Horizontal)
        self._gg_slider.setRange(-180, 180); self._gg_slider.setValue(90); self._gg_slider.setEnabled(False)
        left.addWidget(self._gg_slider)
        self._gg_plot = _GGPlot(); left.addWidget(self._gg_plot)
        self._gg_lab = QLabel('Tick the box to solve the envelope.')
        self._gg_lab.setWordWrap(True); self._gg_lab.setStyleSheet('color:#9A9AA2;font-size:11px')
        left.addWidget(self._gg_lab)
        self._gg_worst_btn = QPushButton('Worst loads around the border')
        self._gg_worst_btn.setEnabled(False)
        self._gg_worst_btn.setToolTip('Solves the loads at 36 points around the border (both turn directions)\n'
                                      'and lists the largest value of every member and ball joint, and where.')
        left.addWidget(self._gg_worst_btn)
        self._gg_env, self._gg_worker, self._gg_pending = None, None, False
        self._gg_timer = QTimer(self); self._gg_timer.setSingleShot(True); self._gg_timer.setInterval(120)
        self._gg_timer.timeout.connect(self._gg_apply_point)
        self._gg_on.toggled.connect(self._on_gg_toggle)
        self._gg_engine.toggled.connect(lambda *_: self._gg_build() if self._gg_on.isChecked() else None)
        self._gg_slider.valueChanged.connect(lambda *_: self._gg_timer.start())
        self._gg_worst_btn.clicked.connect(self._on_gg_worst)
        left.addWidget(QLabel('Corner'))
        self._corner = QComboBox(); self._corner.addItems(_CORNERS); left.addWidget(self._corner)
        left.addWidget(QLabel('3D vectors'))
        self._res = QRadioButton('Resultant'); self._comp = QRadioButton('Components (X/Y/Z)')
        self._ang = QRadioButton('Angles (hover / click a force)')
        self._ang.setToolTip('Resultant arrows plus, for the force under the cursor, how to apply it with two\n'
                             'angles: the cardinal plane it most nearly lies in (translucent), θ in that plane\n'
                             'from its first axis, and the tilt φ off the plane.  Click pins it; click again releases.')
        self._res.setChecked(True)
        bg = QButtonGroup(self); bg.addButton(self._res); bg.addButton(self._comp); bg.addButton(self._ang)
        left.addWidget(self._res); left.addWidget(self._comp); left.addWidget(self._ang)
        self._wheels = QCheckBox('Show wheels'); self._wheels.setChecked(True)
        self._wheels.setToolTip('Hide the tyres and rims to see the upright, bearings and arrows (also in the 3-D view box)')
        left.addWidget(self._wheels)

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
        # was a FIXED 270 px column in a fixed 292 px box with the sideways
        # bar off: wider rows were simply cut.  Resizable, never clipped now.
        from gui.side_panel import SideScroll, side_split
        lw = QWidget(); lw.setLayout(left)
        _sc = SideScroll(lw, 292)
        _sc.setStyleSheet('QScrollArea{border:0;}')

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

        self._tbl = QTableWidget(0, 6)
        # Magnitude column mixes forces (N) and moments (N·m); the Type column
        # carries the unit per row, so the header names BOTH rather than lying (N).
        # Angle column (user 2026-10-08): angle of the force — or of a moment's
        # axis — to each +axis of the chassis frame, 0..180° (0 = along +axis,
        # 90 = perpendicular, 180 = along the negative axis).  Same numbers as
        # the hover label on the arrows above.
        self._tbl.setHorizontalHeaderLabels(['Category', 'Point / member', 'Magnitude (N / N·m)',
                                             'Direction (lat / fore-aft / vert)',
                                             'Angle: plane θ / tilt φ · to +lat / +f-a / +vert (°)', 'Type'])
        self._tbl.horizontalHeaderItem(4).setToolTip(
            'How to apply the load with two angles: the cardinal plane it most nearly lies in,\n'
            'θ in that plane from its first axis, and the tilt φ off the plane toward the normal\n'
            '(the Angles option in the 3-D view draws this construction).  Then:\n'
            'Angle between the vector and each positive chassis axis, 0–180°.\n'
            '+lat = car left, +fore-aft = rearward, +vert = up.\n'
            '0° = along the axis, 90° = perpendicular, 180° = along the negative axis.\n'
            'For a moment row this is the orientation of the moment axis (right-hand rule).')
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
        self._side_split = side_split(_sc, self._split, 292, key='loads')
        root.addWidget(self._side_split, stretch=1)

        self._corner.currentTextChanged.connect(self._on_input)
        for _s in (self._lat_g, self._lon_g):
            _s.valueChanged.connect(self._on_input)
        self._speed.valueChanged.connect(self._on_speed_edit)
        self._speed_auto.toggled.connect(self._on_input)
        self._res.toggled.connect(self._on_input)
        self._ang.toggled.connect(self._on_input)
        self._wheels.toggled.connect(lambda on: self._v3d.set_wheels(bool(on)) if self._v3d is not None else None)
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
        if self._gg_on.isChecked():
            self._gg_build()            # the envelope belongs to ONE speed
            return
        self.refresh()

    # ── G-G envelope mode ──
    def _on_gg_toggle(self, on):
        on = bool(on)
        for wdg in (self._lat_g, self._lon_g, self._speed_auto):
            wdg.setEnabled(not on)
        self._gg_slider.setEnabled(on and self._gg_env is not None)
        self._gg_worst_btn.setEnabled(on and self._gg_env is not None)
        if not on:
            self._gg_lab.setText('Tick the box to solve the envelope.')
            return
        if self._speed_auto.isChecked():
            # default speed = the aero package reference speed
            self._speed_auto.blockSignals(True); self._speed_auto.setChecked(False)
            self._speed_auto.blockSignals(False)
            try:
                v = float(self._main._dynamics_panel.get_custom_aero_params()['V_ref_kph'])
                self._speed.blockSignals(True); self._speed.setValue(v); self._speed.blockSignals(False)
            except Exception:
                pass
        self._gg_build()

    def _gg_build(self):
        if self._gg_worker is not None and self._gg_worker.isRunning():
            self._gg_pending = True
            return
        self._gg_pending = False
        self._gg_slider.setEnabled(False); self._gg_worst_btn.setEnabled(False)
        self._gg_lab.setText(f'Solving the G-G envelope at {self._speed.value():.1f} km/h … '
                             '(about 15 s; the slider unlocks when it is done)')
        self._gg_worker = _GGWorker(self._main, self._speed.value(), self._gg_engine.isChecked())
        self._gg_worker.done.connect(self._on_gg_built)
        self._gg_worker.start()

    def _on_gg_built(self, env):
        if self._gg_pending:
            self._gg_build(); return
        if isinstance(env, Exception) or not np.isfinite(env.get('ay_limit_g', float('nan'))):
            self._gg_env = None
            self._gg_lab.setText(f'G-G envelope failed: {env if isinstance(env, Exception) else "no solved state"}')
            self._gg_plot.set_border(None)
            return
        self._gg_env = env
        self._gg_plot.set_border(env['border'])
        on = self._gg_on.isChecked()
        self._gg_slider.setEnabled(on); self._gg_worst_btn.setEnabled(on)
        self._gg_apply_point()

    def _gg_apply_point(self):
        env = self._gg_env
        if env is None or not self._gg_on.isChecked():
            return
        th = float(self._gg_slider.value())
        lat, lon = gg_border_point(env, th)
        if not (np.isfinite(lat) and np.isfinite(lon)):
            return
        self._gg_plot.set_point(lat, lon)
        for wdg, val in ((self._lat_g, lat), (self._lon_g, lon)):
            wdg.blockSignals(True); wdg.setValue(round(val, 2)); wdg.blockSignals(False)
        what = ('straight braking' if abs(th) < 1 else 'straight acceleration' if abs(th) > 179
                else 'pure cornering' if abs(abs(th) - 90) < 1 else
                'cornering + braking' if abs(th) < 90 else 'cornering + acceleration')
        self._gg_lab.setText(
            f'{th:+.0f}°  {what}: {lat:+.2f} g lateral, {abs(lon):.2f} g {"braking" if lon < 0 else "forward"}\n'
            f'Envelope at {env["speed_kph"]:.1f} km/h, {env["aero_total_N"]:.0f} N downforce: '
            f'cornering {env["ay_limit_g"]:.2f} g, braking {abs(env["brake_tip_g"]):.2f} g, '
            f'forward {env["accel_tip_g"]:.2f} g'
            + (' (engine-limited)' if bool(np.asarray(env['power_limited'])[len(env['power_limited']) // 2])
               else '' if env.get('engine_limit') else ' (tyre-limited)'))
        self.refresh()

    def _on_gg_worst(self):
        env = self._gg_env
        if env is None:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            res = gg_worst_loads(self._main, env, n=36)
        finally:
            QApplication.restoreOverrideCursor()
        self._gg_last_worst = res
        corner = self._corner_arg()
        rows = [r for r in res['rows'] if corner is None or r['corner'] == corner]
        dlg = QDialog(self); dlg.setWindowTitle(
            f'Worst loads around the G-G border — {res["speed_kph"]:.1f} km/h, 36 points, both turn directions')
        dlg.resize(980, 640)
        lay = QVBoxLayout(dlg)
        note = QLabel('Largest value of each member / joint force over the whole border at this speed. '
                      'Direction: members are pulled (tension) or pushed (compression) along their length; '
                      'joint parts are the force ON the upright — outboard / inboard, forward / rearward, up / down. '
                      'Signed column: + tension; joint parts +lateral = car left, +fore-aft = rearward, +vertical = up. '
                      'Double-click a row to put the slider on that point.')
        note.setWordWrap(True); lay.addWidget(note)
        t = QTableWidget(len(rows), 7)
        t.setHorizontalHeaderLabels(['Corner', 'Member / joint', 'Largest (N)', 'Direction', 'Signed (N)',
                                     'Where: angle', 'Where: lateral / longitudinal g'])
        for i, r in enumerate(rows):
            vals = [r['corner'], _GG_NAMES.get(r['item'], r['item']), None,
                    _gg_dir(r['corner'], r['item'], float(r['signed'])), f"{r['signed']:+,.0f}",
                    f"{r['theta']:+.0f}°", f"{r['lat_g']:+.2f} / {r['lon_g']:+.2f}"]
            for j, v in enumerate(vals):
                it = QTableWidgetItem()
                if j == 2:
                    it.setData(Qt.ItemDataRole.DisplayRole, round(r['max_abs']))
                else:
                    it.setText(v)
                if j == 5:
                    it.setData(Qt.ItemDataRole.UserRole, r['theta'])
                t.setItem(i, j, it)
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setSortingEnabled(True); t.sortItems(2, Qt.SortOrder.DescendingOrder)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        t.setStyleSheet('QTableWidget{background:#111;gridline-color:#333;}'
                        'QHeaderView::section{background:#1e1e24;color:#e0a83a;padding:5px;border:0;}')
        def _go(row, _col):
            it = t.item(row, 5)
            if it is not None:
                self._gg_slider.setValue(int(round(float(it.data(Qt.ItemDataRole.UserRole)))))
        t.cellDoubleClicked.connect(_go)
        lay.addWidget(t)
        dlg.show()

    def _refresh_aero_label(self):
        try:
            lat, lon = self._case_g()
            ca = case_aero(self._main, lat, lon, self._case_speed())
            if not self._speed_auto.isChecked():
                ca['rule'] = 'typed'
            txt = 'Solved at ' + case_aero_text(ca)
            env = self._gg_env
            if (env is not None and not self._gg_on.isChecked()
                    and abs(float(env['speed_kph']) - float(ca['speed_kph'])) < 0.05
                    and (abs(lat) + abs(lon)) > 1e-6):
                # where this typed case sits against the G-G border at the SAME speed
                th = float(np.degrees(np.arctan2(lat, -lon)))
                blat, blon = gg_border_point(env, th)
                r_b = float(np.hypot(blat, blon))
                if np.isfinite(r_b) and r_b > 1e-6:
                    pct = 100.0 * float(np.hypot(lat, lon)) / r_b
                    txt += (f'\nG-G envelope at this speed: this case is {pct:.0f} % of the way to the border'
                            + (' — OUTSIDE what the car can reach' if pct > 100.5 else ''))
            self._aero_lab.setText(txt)
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
            self._v3d.set_wheels(self._wheels.isChecked())
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
            if 'wheels' in d:
                self._wheels.blockSignals(True); self._wheels.setChecked(bool(d['wheels'])); self._wheels.blockSignals(False)
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
                     f'{v[0]:+,.0f} / {v[1]:+,.0f} / {v[2]:+,.0f}', angle_text(v), typ]
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
            self._v3d.set_angle_mode(self._ang.isChecked())
            self._main.build_load_view(self._v3d, self._corner_arg(), lat, lon, vec,
                                       speed_kph=self._speed_override())
            self._v3d.set_wheels(self._wheels.isChecked())
            # build_load_view draws the arrows at the case-rule speed; a typed
            # speed re-draws them at that speed so picture == table.
            v = self._speed_override()
            if v is not None:
                self._v3d.set_load_vectors(load_arrows(self._main, lat, lon, mode=vec,
                                                       only_corner=self._corner_arg(), speed_kph=v))
        except Exception:
            pass
