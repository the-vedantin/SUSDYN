"""
gui/panels.py — All Qt sidebar panels for Vahan

Axis convention: X=lateral(outboard), Y=longitudinal(fwd), Z=up
"""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QGroupBox, QVBoxLayout, QHBoxLayout, QGridLayout,
    QRadioButton, QButtonGroup, QSlider, QDoubleSpinBox, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QListWidget,
    QListWidgetItem, QComboBox, QPushButton, QCheckBox,
    QSizePolicy, QScrollArea, QFrame, QToolButton, QAbstractItemView,
    QDialog, QDialogButtonBox, QFileDialog, QApplication, QLineEdit,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont

from vahan.metrics_catalog import CATALOG, CATALOG_MAP, DEFAULT_Y_KEYS
from gui import section_info

HP_NAMES = [
    'uca_front', 'uca_rear',  'uca_outer',
    'lca_front', 'lca_rear',  'lca_outer',
    'tie_rod_inner', 'tie_rod_outer', 'wheel_center',
    'pushrod_outer', 'pushrod_inner',
    'rocker_pivot',  'rocker_spring_pt', 'spring_chassis_pt',
]

CHASSIS_PTS = frozenset({
    'uca_front', 'uca_rear', 'lca_front', 'lca_rear',
    'tie_rod_inner', 'rocker_pivot', 'spring_chassis_pt', 'rocker_axis_pt',
})

ARB_HP_NAMES = ['arb_drop_top', 'arb_arm_end', 'arb_pivot']

C_BLUE = '#cccccc'
C_RED  = '#EF5350'
C_TEXT = '#e0e0e0'
C_SUB  = '#888888'


# ══════════════════════════════════════════════════════════════════════════════
#  COLLAPSIBLE SECTION WIDGET
# ══════════════════════════════════════════════════════════════════════════════

class CollapsibleSection(QWidget):
    """A titled section that can be toggled open/closed.

    Every section also carries an info button (ⓘ) on the right of its
    header.  Call ``set_info(html)`` to attach an "immense detail" write-up
    that pops up in a scrollable dialog when the ⓘ is clicked.  Sections
    that never call ``set_info`` simply hide the button.
    """

    def __init__(self, title: str, parent=None, header_color: str = '#cccccc'):
        super().__init__(parent)
        self._title = title
        self._info_html: str | None = None
        self._info_title: str = title
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 2)
        layout.setSpacing(0)

        # ── header: toggle button (expanding) + info button (fixed) ──────────
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(3)

        self._btn = QToolButton()
        self._btn.setText(f'  v  {title}')
        self._btn.setCheckable(True)
        self._btn.setChecked(True)
        self._btn.setStyleSheet(f"""
            QToolButton {{
                background: #111111;
                color: {header_color};
                border: 1px solid #2a2a2a;
                border-radius: 3px;
                text-align: left;
                font-weight: bold;
                font-size: 12px;
                padding: 5px 8px;
                width: 100%;
            }}
            QToolButton:hover {{ background: #1a1a1a; }}
        """)
        self._btn.clicked.connect(self._toggle)
        self._btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        header.addWidget(self._btn, 1)

        # Info (ⓘ) button — colour-blind-safe blue circle, hidden until set_info.
        self._info_btn = QPushButton('ⓘ')          # ⓘ
        self._info_btn.setFixedSize(26, 26)
        self._info_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._info_btn.setStyleSheet(
            'QPushButton { background: #111111; color: #4FC3F7; '
            'border: 1px solid #2a2a2a; border-radius: 13px; '
            'font-weight: bold; font-size: 15px; }'
            'QPushButton:hover { background: #14344a; border-color: #4FC3F7; }')
        self._info_btn.setToolTip('What is this section? Click for full detail.')
        self._info_btn.clicked.connect(self._show_info)
        self._info_btn.setVisible(False)
        header.addWidget(self._info_btn, 0)
        layout.addLayout(header)

        # content
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(2, 2, 2, 2)
        self._content_layout.setSpacing(4)
        layout.addWidget(self._content)

    def add_widget(self, w: QWidget):
        self._content_layout.addWidget(w)

    def add_layout(self, lay):
        self._content_layout.addLayout(lay)

    def set_info(self, html: str, title: str | None = None):
        """Attach a detailed HTML description; reveals the ⓘ header button."""
        self._info_html = html
        if title:
            self._info_title = title
        self._info_btn.setVisible(bool(html))

    def _show_info(self):
        if not self._info_html:
            return
        from PyQt6.QtWidgets import QTextBrowser
        dlg = QDialog(self)
        dlg.setWindowTitle(f'ⓘ  {self._info_title}')
        dlg.resize(640, 720)
        dlg.setStyleSheet(
            'QDialog { background: #0a0a0a; }'
            'QTextBrowser { background: #0a0a0a; color: #e0e0e0; border: none; '
            'font-size: 12px; }')
        lay = QVBoxLayout(dlg)
        tb = QTextBrowser()
        tb.setOpenExternalLinks(False)
        tb.setHtml(self._info_html)
        lay.addWidget(tb)
        close_btn = QPushButton('Close')
        close_btn.clicked.connect(dlg.accept)
        close_btn.setStyleSheet(
            'QPushButton { background: #1a5276; color: white; padding: 6px 20px; '
            'border-radius: 3px; font-weight: bold; }'
            'QPushButton:hover { background: #1f6da0; }')
        lay.addWidget(close_btn)
        dlg.exec()

    def _toggle(self, checked: bool):
        self._content.setVisible(checked)
        arrow = 'v' if checked else '>'
        self._btn.setText(f'  {arrow}  {self._title}')


# ══════════════════════════════════════════════════════════════════════════════
#  MOTION PANEL
# ══════════════════════════════════════════════════════════════════════════════

class MotionPanel(CollapsibleSection):
    """
    Signals:
        motion_changed(str)           — 'heave'/'roll'/'pitch'
        range_changed(float, float)   — (min, max) in mm or deg
        position_changed(float)       — current position in mm or deg
        damper_params_changed(dict)   — {'stroke_mm',
                                         'preload_front_mm',
                                         'preload_rear_mm',
                                         'fully_extended_mm'}
        apply_sag_requested()         — user clicked "Apply to HPs":
                                        rewrite moving hardpoints so the
                                        physics-consistent static sag is
                                        baked into the CAD geometry.

    Sag is COMPUTED from stroke + preload + vehicle params (spring,
    mass, MR), not entered as an input.  Call update_sag_display()
    with a dict from VehicleParams.static_sag() to refresh the
    read-only sag labels on this panel.

    The `fully_extended_mm` field is the damper's eye-to-eye length
    at zero compression (from the damper spec).  The main window uses
    it to decide how much the kinematic geometry *would* shift from
    the CAD position to reach physics-consistent static ride height:
        comp_cad   = L_full − damper_length_at_CAD
        comp_need  = sag_shock (from spring physics)
        shift      = comp_need − comp_cad  (shock mm, in WHEEL via /MR)
    Changing damper params NEVER auto-shifts the geometry.  The shift
    is only applied to hardpoints when the user clicks "Apply to HPs",
    which emits apply_sag_requested.  If `fully_extended_mm` is 0
    (default) no shift is computed (CAD is treated as static).
    """
    motion_changed        = pyqtSignal(str)
    range_changed         = pyqtSignal(float, float)
    position_changed      = pyqtSignal(float)
    damper_params_changed = pyqtSignal(dict)
    apply_sag_requested   = pyqtSignal()

    def __init__(self):
        super().__init__('Motion')
        self._motion  = 'heave'
        self._min_val = -50.0
        self._max_val =  50.0
        self._pos     =   0.0
        self._building = False
        self._build()
        self.set_info(section_info.MOTION)

    @property
    def motion(self) -> str:    return self._motion
    @property
    def min_val(self) -> float: return self._min_val
    @property
    def max_val(self) -> float: return self._max_val
    @property
    def position(self) -> float: return self._pos

    def _build(self):
        # Radio
        self._btn_grp = QButtonGroup(self)
        row = QHBoxLayout()
        for label, key in [('Heave', 'heave'), ('Roll', 'roll'),
                            ('Pitch', 'pitch'), ('Steer', 'steer')]:
            rb = QRadioButton(label)
            rb.setChecked(key == 'heave')
            rb.toggled.connect(lambda chk, k=key: self._on_motion(chk, k))
            self._btn_grp.addButton(rb)
            row.addWidget(rb)
        self.add_layout(row)

        # Asymmetric range
        grid = QGridLayout(); grid.setSpacing(4)
        grid.addWidget(QLabel('Min:'), 0, 0)
        self._min_spin = _spin(-1e6, 0, -50, ' mm'); self._min_spin.valueChanged.connect(self._on_range)
        grid.addWidget(self._min_spin, 0, 1)
        grid.addWidget(QLabel('Max:'), 0, 2)
        self._max_spin = _spin(0, 1e6, 50, ' mm');   self._max_spin.valueChanged.connect(self._on_range)
        grid.addWidget(self._max_spin, 0, 3)
        self.add_layout(grid)

        # Damper limits — stroke + per-axle preload + fully-extended length.
        # Sag is COMPUTED (not entered) from spring/mass/MR + preload.
        dlim = QGridLayout(); dlim.setSpacing(4)
        dlim.addWidget(QLabel('Stroke:'), 0, 0)
        self._stroke = _spin(0.1, 1e6, 55, ' mm'); self._stroke.valueChanged.connect(self._on_damper)
        self._stroke.setToolTip('Total damper stroke (shock-frame mm)')
        dlim.addWidget(self._stroke, 0, 1)
        dlim.addWidget(QLabel('Preload F:'), 0, 2)
        self._preload_f = _spin(0, 1e6, 0, ' mm', dec=1, step=0.5)
        self._preload_f.valueChanged.connect(self._on_damper)
        dlim.addWidget(self._preload_f, 0, 3)
        dlim.addWidget(QLabel('Preload R:'), 1, 2)
        self._preload_r = _spin(0, 1e6, 0, ' mm', dec=1, step=0.5)
        self._preload_r.valueChanged.connect(self._on_damper)
        dlim.addWidget(self._preload_r, 1, 3)

        # Fully-extended eye-to-eye length (0 = disable shift diagnostic).
        # When nonzero, the main window computes how much the CAD damper
        # is already compressed relative to L_full; the "Apply to HPs"
        # button below lets the user commit the physics-static shift
        # to the actual hardpoints (the pushrod / rocker / wheel points
        # are NEVER moved just because damper params changed).
        dlim.addWidget(QLabel('Fully ext.:'), 1, 0)
        self._fully_extended = _spin(0, 1e6, 210.0, ' mm', dec=1, step=0.5)
        self._fully_extended.valueChanged.connect(self._on_damper)
        self._fully_extended.setToolTip(
            'Damper length at ZERO compression (from damper spec). '
            'Leave 0 to skip the shift diagnostic. When set, the model '
            'computes how much of the stroke is already used at CAD. '
            'Changing this value does NOT move any hardpoints — click '
            '"Apply to HPs" to commit the sag shift.')
        dlim.addWidget(self._fully_extended, 1, 1)
        self.add_layout(dlim)

        # "Apply" button — commits the computed sag shift into the
        # actual moving hardpoints (wheel, UCA/LCA/tie-rod outers,
        # pushrod endpoints, rocker spring point).  Everything in
        # the 3D view stays put until the user presses this.
        from PyQt6.QtWidgets import QPushButton
        btn_row = QHBoxLayout()
        self._apply_sag_btn = QPushButton('Apply Sag to Hardpoints')
        self._apply_sag_btn.setToolTip(
            'Rewrite the moving hardpoints so that travel=0 IS the physics-'
            'consistent static ride height.  Chassis-side points stay put; '
            'wheel, upright and rocker outboard points move to the sagged '
            'positions.  Use this when you want the 3D view to reflect '
            'where the car actually sits at rest.')
        self._apply_sag_btn.clicked.connect(self.apply_sag_requested.emit)
        btn_row.addWidget(self._apply_sag_btn)
        btn_row.addStretch(1)
        self.add_layout(btn_row)

        # Computed-sag display (read-only)
        self._sag_lbl = QLabel('Static sag: F — / R —')
        self._sag_lbl.setStyleSheet(
            'color: #b0b0b0; font-family: Consolas, monospace; '
            'font-size: 11px; padding: 2px 0;')
        self.add_widget(self._sag_lbl)

        # Slider
        self.add_widget(QLabel('Position (live 3D):'))
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 400)
        self._slider.setValue(200)
        self._slider.valueChanged.connect(self._on_slider)
        self.add_widget(self._slider)

        self._pos_label = QLabel(' 0.0 mm')
        self._pos_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.add_widget(self._pos_label)

    def _on_motion(self, checked, key):
        if not checked: return
        self._motion = key
        self._building = True
        defs = {'heave': (-50, 50, ' mm'), 'roll': (-5, 5, ' °'),
                'pitch': (-30, 30, ' mm'), 'steer': (-360, 360, ' °')}
        lo, hi, suf = defs[key]
        self._min_spin.setSuffix(suf); self._min_spin.setValue(lo)
        self._max_spin.setSuffix(suf); self._max_spin.setValue(hi)
        self._min_val, self._max_val = lo, hi
        self._building = False
        self._sync()
        self.motion_changed.emit(key)
        self.range_changed.emit(self._min_val, self._max_val)

    def _on_range(self):
        if self._building: return
        self._min_val = self._min_spin.value()
        self._max_val = self._max_spin.value()
        if self._min_val >= self._max_val: return
        self._sync()
        self.range_changed.emit(self._min_val, self._max_val)

    def _on_slider(self, _):
        self._sync()
        self.position_changed.emit(self._pos)

    def _on_damper(self):
        self.damper_params_changed.emit({
            'stroke_mm':         self._stroke.value(),
            'preload_front_mm':  self._preload_f.value(),
            'preload_rear_mm':   self._preload_r.value(),
            'fully_extended_mm': self._fully_extended.value(),
        })

    # Public getters (used by save/load + main window queries)
    @property
    def stroke_mm(self) -> float:
        return self._stroke.value()

    @property
    def preload_front_mm(self) -> float:
        return self._preload_f.value()

    @property
    def preload_rear_mm(self) -> float:
        return self._preload_r.value()

    @property
    def fully_extended_mm(self) -> float:
        return self._fully_extended.value()

    def set_fully_extended_mm(self, val: float):
        """Silently set the fully-extended length (used by project load)."""
        blocked = self._fully_extended.blockSignals(True)
        self._fully_extended.setValue(float(val))
        self._fully_extended.blockSignals(blocked)

    def update_sag_display(self, sag_info: dict):
        """Refresh the read-only sag readout from a VehicleParams.static_sag() dict.

        The dict may include extra diagnostic keys set by the main window:
            cad_damper_len_front_mm / cad_damper_len_rear_mm
            cad_compression_front_mm / cad_compression_rear_mm
            shift_shock_front_mm     / shift_shock_rear_mm
        to show how much the CAD damper is already compressed and how far
        the geometry is shifting to reach physics-consistent static sag.
        """
        if not sag_info:
            self._sag_lbl.setText('Static sag: F — / R —')
            return
        f_mm  = sag_info.get('sag_shock_front_mm', 0.0)
        r_mm  = sag_info.get('sag_shock_rear_mm',  0.0)
        f_pct = sag_info.get('sag_front_pct', 0.0)
        r_pct = sag_info.get('sag_rear_pct',  0.0)
        warn_list = []
        if sag_info.get('topped_out_front') or sag_info.get('topped_out_rear'):
            warn_list.append('TOP-OUT')
        if sag_info.get('bottomed_out_front') or sag_info.get('bottomed_out_rear'):
            warn_list.append('BOTTOM-OUT')
        if sag_info.get('cad_over_extended_front') or sag_info.get('cad_over_extended_rear'):
            warn_list.append('CAD > L_full')
        if sag_info.get('cad_over_compressed_front') or sag_info.get('cad_over_compressed_rear'):
            warn_list.append('CAD > stroke')
        warn = ('  ' + ' / '.join(warn_list)) if warn_list else ''

        lines = [
            f'Static sag:  F {f_mm:5.2f} mm ({f_pct:4.1f}%)   '
            f'R {r_mm:5.2f} mm ({r_pct:4.1f}%){warn}'
        ]

        # If the main window supplied CAD-compression diagnostics, show them
        # so the user can see how their CAD maps onto the physical damper.
        shift_f = sag_info.get('shift_shock_front_mm')
        shift_r = sag_info.get('shift_shock_rear_mm')
        cad_f   = sag_info.get('cad_compression_front_mm')
        cad_r   = sag_info.get('cad_compression_rear_mm')
        if shift_f is not None and shift_r is not None:
            cad_f_str = f'{cad_f:+5.2f}' if cad_f is not None else '  —  '
            cad_r_str = f'{cad_r:+5.2f}' if cad_r is not None else '  —  '
            # Label as "would-shift" so it's obvious the geometry hasn't
            # moved yet — clicking "Apply Sag to Hardpoints" is what
            # actually commits these numbers into the moving hardpoints.
            lines.append(
                f'CAD compr.:  F {cad_f_str} mm   R {cad_r_str} mm   '
                f'would-shift F {shift_f:+5.2f} / R {shift_r:+5.2f} mm'
            )
        self._sag_lbl.setText('\n'.join(lines))
        if warn_list:
            self._sag_lbl.setStyleSheet(
                'color: #ffaa33; font-family: Consolas, monospace; '
                'font-size: 11px; padding: 2px 0; font-weight: bold;')
        else:
            self._sag_lbl.setStyleSheet(
                'color: #b0b0b0; font-family: Consolas, monospace; '
                'font-size: 11px; padding: 2px 0;')

    def _sync(self):
        pct = self._slider.value() / 400.0
        self._pos = self._min_val + pct * (self._max_val - self._min_val)
        unit = '°' if self._motion == 'roll' else ' mm'
        self._pos_label.setText(f'{self._pos:+.1f}{unit}')


# ══════════════════════════════════════════════════════════════════════════════
#  STEERING PANEL (front only)
# ══════════════════════════════════════════════════════════════════════════════

class SteeringPanel(CollapsibleSection):
    """
    Steering geometry parameters.
    Steer angle is driven by the Motion panel (Steer mode),
    not by a slider here.

    Signals:
        steering_changed(dict) — {'rack_travel_per_rev_mm', 'total_rack_travel_mm'}
    """
    steering_changed = pyqtSignal(dict)

    def __init__(self):
        super().__init__('Steering Parameters (Front)')
        self._build()

    def get_params(self) -> dict:
        return {
            'rack_travel_per_rev_mm': self._rack_ratio.value(),
            'total_rack_travel_mm':   self._rack_total.value(),
        }

    def _build(self):
        grid = QGridLayout(); grid.setSpacing(4)

        grid.addWidget(QLabel('Rack travel/rev:'), 0, 0)
        self._rack_ratio = _spin(0.1, 100000, 120.0, ' mm/rev')
        self._rack_ratio.valueChanged.connect(self._on_rack_changed)
        grid.addWidget(self._rack_ratio, 0, 1)

        grid.addWidget(QLabel('Total rack travel:'), 1, 0)
        self._rack_total = _spin(0.1, 100000, 64.0, ' mm')
        self._rack_total.valueChanged.connect(self._on_rack_changed)
        grid.addWidget(self._rack_total, 1, 1)
        self.add_layout(grid)

        # ── Readout: max steering-wheel angle achievable ────────────────────
        # max_hw_deg (one-way) = (total_rack_travel / 2) / rack_per_rev × 360
        # lock-to-lock = 2 × max_hw_deg
        self._max_hw_lbl = QLabel()
        self._max_hw_lbl.setStyleSheet(
            'color: #4FC3F7; font-size: 11px; font-weight: bold;'
            ' padding: 2px 4px;')
        self._max_hw_lbl.setWordWrap(True)
        self.add_widget(self._max_hw_lbl)
        self._refresh_max_hw()

        info = QLabel('Use Motion > Steer mode to simulate steering.')
        info.setWordWrap(True)
        info.setStyleSheet('color: #888888; font-size: 11px;')
        self.add_widget(info)

        self.set_info(section_info.STEERING)

    def _on_rack_changed(self, *_):
        self._refresh_max_hw()
        self.steering_changed.emit(self.get_params())

    def _refresh_max_hw(self):
        """Recompute and display max steering-wheel angle from rack params."""
        rack_per_rev = float(self._rack_ratio.value())
        total_mm     = float(self._rack_total.value())
        if rack_per_rev <= 0 or total_mm <= 0:
            self._max_hw_lbl.setText('Max steering-wheel angle: —')
            return
        max_hw = (total_mm / 2.0) / rack_per_rev * 360.0
        turns  = max_hw / 360.0
        self._max_hw_lbl.setText(
            f'Max steering-wheel angle: ±{max_hw:.0f}°  '
            f'({turns:.2f} turns from centre)\n'
            f'Lock-to-lock: {2*max_hw:.0f}° ({2*turns:.2f} turns)')

    def max_handwheel_deg(self) -> float:
        """One-way max hand-wheel angle (deg) from the two rack params."""
        rack_per_rev = float(self._rack_ratio.value())
        total_mm     = float(self._rack_total.value())
        if rack_per_rev <= 0 or total_mm <= 0:
            return 0.0
        return (total_mm / 2.0) / rack_per_rev * 360.0


# ══════════════════════════════════════════════════════════════════════════════
#  CAR PARAMETERS PANEL
# ══════════════════════════════════════════════════════════════════════════════

class CarParamsPanel(CollapsibleSection):
    """Geometry (axle spacing, wheelbase, track, wheel offset), tire, CG."""
    params_changed = pyqtSignal(dict)
    perspective_changed = pyqtSignal(bool)   # True = perspective, False = orthographic

    def __init__(self):
        super().__init__('Car Parameters')
        self.set_info(section_info.CAR_PARAMS)
        self._build()

    def get_params(self) -> dict:
        return {
            'axle_spacing_mm':       self._axle_sp.value(),
            'wheelbase_mm':          self._wb.value(),
            'track_f_mm':            self._track_f.value(),
            'track_r_mm':            self._track_r.value(),
            'wheel_offset_f_mm':     self._woff_f.value(),
            'wheel_offset_r_mm':     self._woff_r.value(),
            'tire_outer_dia_mm':     self._t_outer.value(),
            'tire_rim_dia_mm':       self._t_rim.value(),
            'tire_width_mm':         self._t_width.value(),
            'show_ground':           self._show_ground.isChecked(),
            'cg_x_mm':              self._cg_x.value(),
            'cg_y_mm':              self._cg_y.value(),
            'cg_z_mm':              self._cg_z.value(),
            'front_brake_bias_pct':  self._brake_bias.value(),
            'rack_length_mm':        self._rack_len.value(),
            'spring_od_mm':          self._spring_od.value(),
            'damper_od_mm':          self._damper_od.value(),
            'track_pushes_inboard':  self._chk_track_inboard.isChecked(),
            # Rear driveshaft / differential packaging (view + interference only,
            # NOT a dynamics quantity — kept out of VehicleParams).  Placeholder
            # tripod/shaft dims; user overwrites with real numbers.
            'diff_long_mm':          self._diff_long.value(),
            'diff_vert_mm':          self._diff_vert.value(),
            'diff_lateral_offset_mm': self._diff_offset.value(),
            'diff_housing_width_mm': self._diff_hw.value(),
            'tripod_od_mm':          self._tripod_od.value(),
            'driveshaft_dia_mm':     self._ds_dia.value(),
            'rotor_dia_mm':          self._rotor_dia.value(),
            'show_driveshaft':       self._show_driveshaft.isChecked(),
            'show_brakes':           self._show_brakes.isChecked(),
            'show_shock_thickness':  self._show_shock_thick.isChecked(),
            'view_mode':             self._view_mode_combo.currentText().lower(),
        }

    def _build(self):
        g = QGridLayout(); g.setSpacing(4)
        def row(label, lo, hi, val, suf, r, dec=0, step=10):
            g.addWidget(QLabel(label), r, 0)
            sb = _spin(lo, hi, val, suf, dec=dec, step=step)
            sb.valueChanged.connect(lambda _: self.params_changed.emit(self.get_params()))
            g.addWidget(sb, r, 1)
            return sb
        r = 0
        # Geometry: axle spacing vs wheelbase (separated)
        self._axle_sp    = row('Axle spacing:',          0.1, 1e6, 1537, ' mm', r); r += 1
        self._wb         = row('Wheelbase:',             0.1, 1e6, 1537, ' mm', r); r += 1
        self._track_f    = row('Track width F:',         0.1, 1e6, 1222, ' mm', r); r += 1
        self._track_r    = row('Track width R:',         0.1, 1e6, 1200, ' mm', r); r += 1
        self._woff_f     = row('Wheel offset F:',       -1e6, 1e6,   25, ' mm', r, dec=1, step=1); r += 1
        self._woff_r     = row('Wheel offset R:',       -1e6, 1e6,   25, ' mm', r, dec=1, step=1); r += 1
        # Tire dimensions
        self._t_outer    = row('Tyre OD:',               0.1, 1e6,  406, ' mm', r); r += 1
        self._t_rim      = row('Rim dia:',               0.1, 1e6,  330, ' mm', r); r += 1
        self._t_width    = row('Tyre width:',            0.1, 1e6,  200, ' mm', r); r += 1
        # CG position
        self._cg_x       = row('CG X (lateral):',       -1e6, 1e6,    0, ' mm', r, dec=1, step=1); r += 1
        self._cg_y       = row('CG Y (longitudinal):',   0,   1e6, 1100, ' mm', r, dec=1, step=5); r += 1
        self._cg_z       = row('CG Z (height):',         0,   1e6,  280, ' mm', r, dec=1, step=1); r += 1
        self._brake_bias = row('Front Brake Bias:',      0,   100,   65, ' %',  r,
                               dec=0, step=1); r += 1
        # Rack + spring/damper hardware widths (set from wizard but live-editable)
        self._rack_len   = row('Rack length:',           0.1, 2000, 438.16, ' mm', r,
                               dec=2, step=5); r += 1
        self._spring_od  = row('Spring OD:',             1,   300,   63, ' mm', r,
                               dec=1, step=1); r += 1
        self._damper_od  = row('Damper OD:',             1,   300,   50, ' mm', r,
                               dec=1, step=1); r += 1
        # Rear driveshaft / differential packaging (rear-only, RWD).  Diff
        # default sits on the rear axle line, mid-height, on the centreline.
        # "Effective spacing" = diff HOUSING WIDTH (tripod centres at the
        # housing faces + a small stub).  Tripod OD / shaft dia are PLACEHOLDERS.
        self._diff_long  = row('Diff long (Y, fore-aft):', 0,  1e6, 1537, ' mm', r,
                               dec=1, step=5); r += 1
        self._diff_vert  = row('Diff vert (Z, height):',   0,  1e6,  150, ' mm', r,
                               dec=1, step=5); r += 1
        self._diff_offset = row('Diff lateral offset (+ = left):', -1e6, 1e6, 57.15,
                               ' mm', r, dec=2, step=1); r += 1
        self._diff_hw    = row('Inboard pivot spacing:',   1, 1000, 292.1, ' mm', r,
                               dec=1, step=5); r += 1
        self._tripod_od  = row('Tripod OD (placeholder):', 1, 1000,   90, ' mm', r,
                               dec=1, step=1); r += 1
        self._ds_dia     = row('Driveshaft OD:',           1, 300, 25.4, ' mm', r,
                               dec=2, step=1); r += 1
        # Brake rotor diameter (Wilwood GP200 max rotor dia = 11 in = 279 mm).
        self._rotor_dia  = row('Rotor dia:',               1, 279, 240, ' mm', r,
                               dec=1, step=5); r += 1
        self.add_layout(g)

        # Info: axle spacing vs wheelbase
        info = QLabel(
            'Axle spacing shifts rear hardpoints.\n'
            'Wheelbase sets contact-patch distance (dynamics).\n'
            'Wheel offset = how far wheel sits past outboard pickups.')
        info.setWordWrap(True)
        info.setStyleSheet('color: #555555; font-size: 10px;')
        self.add_widget(info)

        self._show_ground = QCheckBox('Show ground grid')
        self._show_ground.setChecked(True)
        self._show_ground.stateChanged.connect(
            lambda _: self.params_changed.emit(self.get_params()))
        self.add_widget(self._show_ground)

        # Perspective vs orthographic (CAD-style parallel) projection.
        self._chk_perspective = QCheckBox('Perspective view (uncheck = orthographic / CAD)')
        self._chk_perspective.setChecked(True)
        self._chk_perspective.setToolTip(
            'Perspective: far objects look smaller (natural eye view).\n'
            'Orthographic (unchecked): parallel projection like CAD — no '
            'foreshortening, so you can judge true alignment and check that '
            'parts line up / clear each other without depth distortion.')
        self._chk_perspective.toggled.connect(self.perspective_changed.emit)
        self.add_widget(self._chk_perspective)

        # View mode dropdown (grouped with the perspective / ground toggles).
        vm_row = QHBoxLayout()
        vm_row.addWidget(QLabel('View mode:'))
        self._view_mode_combo = QComboBox()
        self._view_mode_combo.addItems(['Normal', 'Load', 'Interference'])
        self._view_mode_combo.setToolTip(
            'Normal: standard view.\n'
            'Load: desaturate the links so force vectors read clearly.\n'
            'Interference: show part thicknesses and HIGHLIGHT any members '
            'that clash (e.g. driveshaft vs pushrod) in red.')
        self._view_mode_combo.currentTextChanged.connect(
            lambda _: self.params_changed.emit(self.get_params()))
        vm_row.addWidget(self._view_mode_combo)
        vm_row.addStretch(1)
        self.add_layout(vm_row)

        # Rear driveshaft package + shock-thickness view toggles (render only)
        self._show_driveshaft = QCheckBox('Show rear driveshaft / diff package')
        self._show_driveshaft.setChecked(True)
        self._show_driveshaft.setToolTip(
            'Draw the rear differential body, tripods and driveshafts (with '
            'thickness) so you can check packaging and interference and move '
            'the diff. Rear-only (RWD).')
        self._show_driveshaft.stateChanged.connect(
            lambda _: self.params_changed.emit(self.get_params()))
        self.add_widget(self._show_driveshaft)

        self._show_brakes = QCheckBox('Show brake rotors + calipers')
        self._show_brakes.setChecked(True)
        self._show_brakes.stateChanged.connect(
            lambda _: self.params_changed.emit(self.get_params()))
        self.add_widget(self._show_brakes)

        self._show_shock_thick = QCheckBox('Show shock thickness (spring/damper OD)')
        self._show_shock_thick.setChecked(True)
        self._show_shock_thick.setToolTip(
            'ON: springs/dampers drawn as solid cylinders at their OD.\n'
            'OFF: drawn as thin lines (declutter / see behind them).')
        self._show_shock_thick.stateChanged.connect(
            lambda _: self.params_changed.emit(self.get_params()))
        self.add_widget(self._show_shock_thick)

        # Track-width behaviour: by default only the OUTBOARD pickups
        # (uca_outer, lca_outer, tie_rod_outer, wheel_center, pushrod_outer)
        # shift when the user changes track — chassis inboard mounts stay
        # bolted to the frame so the control arms get longer/steeper as
        # track changes.  Tick this to ALSO shift the inboard chassis
        # pickups by the same Δx, keeping the control-arm geometry
        # constant (entire suspension subassembly translates out/in).
        self._chk_track_inboard = QCheckBox(
            'Track change also pushes INBOARD pickups (keep arm length)')
        self._chk_track_inboard.setToolTip(
            'OFF (default): only outboard pickups + wheel move. Control '
            "arms get longer/shorter as track changes.\n"
            'ON: inboard chassis pickups (uca_front/rear, lca_front/rear, '
            'tie_rod_inner) shift by the same Δx too. Arms keep length; '
            'whole suspension translates laterally.')
        self._chk_track_inboard.setChecked(False)
        self._chk_track_inboard.stateChanged.connect(
            lambda _: self.params_changed.emit(self.get_params()))
        self.add_widget(self._chk_track_inboard)

    def set_params(self, d: dict):
        """Populate widgets from a dict (e.g. loaded project)."""
        _map = {
            'axle_spacing_mm': self._axle_sp,
            'wheelbase_mm': self._wb,
            'track_f_mm': self._track_f, 'track_r_mm': self._track_r,
            'wheel_offset_f_mm': self._woff_f, 'wheel_offset_r_mm': self._woff_r,
            'tire_outer_dia_mm': self._t_outer, 'tire_rim_dia_mm': self._t_rim,
            'tire_width_mm': self._t_width,
            'cg_x_mm': self._cg_x, 'cg_y_mm': self._cg_y, 'cg_z_mm': self._cg_z,
            'front_brake_bias_pct': self._brake_bias,
            'rack_length_mm': self._rack_len,
            'spring_od_mm':   self._spring_od,
            'damper_od_mm':   self._damper_od,
            'diff_long_mm':   self._diff_long,
            'diff_vert_mm':   self._diff_vert,
            'diff_lateral_offset_mm': self._diff_offset,
            'diff_housing_width_mm':  self._diff_hw,
            'tripod_od_mm':   self._tripod_od,
            'driveshaft_dia_mm': self._ds_dia,
            'rotor_dia_mm':   self._rotor_dia,
        }
        # back-compat: older files have no diff/driveshaft block → defaults
        d.setdefault('diff_long_mm', 1537.)
        d.setdefault('diff_vert_mm', 150.)
        d.setdefault('diff_lateral_offset_mm', 0.)
        d.setdefault('diff_housing_width_mm', 292.1)   # inboard pivot spacing
        d.setdefault('tripod_od_mm', 90.)
        d.setdefault('driveshaft_dia_mm', 25.4)
        d.setdefault('rotor_dia_mm', 240.)
        # backward compat: old files have cg_height_mm → map to cg_z_mm
        if 'cg_height_mm' in d and 'cg_z_mm' not in d:
            d['cg_z_mm'] = d.pop('cg_height_mm')
        # backward compat: old files have single track_mm → use for both F and R
        if 'track_mm' in d and 'track_f_mm' not in d:
            d['track_f_mm'] = d['track_mm']
            d['track_r_mm'] = d['track_mm']
        # backward compat: old files without axle_spacing → default to wheelbase
        if 'axle_spacing_mm' not in d and 'wheelbase_mm' in d:
            d['axle_spacing_mm'] = d['wheelbase_mm']
        # backward compat: old files without wheel_offset
        d.setdefault('wheel_offset_f_mm', 25.)
        d.setdefault('wheel_offset_r_mm', 25.)
        for key, sb in _map.items():
            if key in d:
                sb.blockSignals(True)
                sb.setValue(d[key])
                sb.blockSignals(False)
        if 'show_ground' in d:
            self._show_ground.blockSignals(True)
            self._show_ground.setChecked(d['show_ground'])
            self._show_ground.blockSignals(False)
        for key, chk in (('show_driveshaft', self._show_driveshaft),
                         ('show_brakes', self._show_brakes),
                         ('show_shock_thickness', self._show_shock_thick)):
            if key in d:
                chk.blockSignals(True)
                chk.setChecked(bool(d[key]))
                chk.blockSignals(False)
        if 'view_mode' in d:
            self._view_mode_combo.blockSignals(True)
            self._view_mode_combo.setCurrentText(str(d['view_mode']).capitalize())
            self._view_mode_combo.blockSignals(False)
        if 'track_pushes_inboard' in d:
            self._chk_track_inboard.blockSignals(True)
            self._chk_track_inboard.setChecked(bool(d['track_pushes_inboard']))
            self._chk_track_inboard.blockSignals(False)


# ══════════════════════════════════════════════════════════════════════════════
#  HARDPOINT TABLE (one per axle)
# ══════════════════════════════════════════════════════════════════════════════

class HardpointPanel(CollapsibleSection):
    """
    Editable hardpoint table. Columns: Name | X | Y | Z (all in mm).

    Now hosts FOUR categories of hardpoints in a single table:
      * corner    — wishbones / tie rod / pushrod / rocker (always present)
      * arb       — anti-roll device hardware (bellcrank / T-bar / control-arm)
      * heave     — 3rd-element heave bracket (HEAVE_TBAR topology only)
      * decoupled — twin-bellcrank + cross-car damper attaches
                    (DECOUPLED topology only)

    Rows are colour-coded by category so the user can see at a glance which
    block they're editing.  Categories with empty dicts are omitted.

    Signals:
        hp_changed(dict, str)   — (new_values_dict, category) where
                                  category is one of:
                                    'corner', 'arb', 'heave', 'decoupled'
                                  The dict contains ONLY the keys for
                                  that category — MainWindow's _on_hp
                                  routes it to the right stored dict.
        row_selected(str)       — hp name when row selected
    """
    hp_changed   = pyqtSignal(dict, str)
    row_selected = pyqtSignal(str)

    def __init__(self, title: str, hp_dict: dict,
                 arb_dict: dict | None = None,
                 heave_dict: dict | None = None,
                 decoupled_dict: dict | None = None):
        super().__init__(title)
        self._hp        = {k: v.copy() for k, v in hp_dict.items()}
        self._arb       = {k: v.copy() for k, v in (arb_dict or {}).items()}
        self._heave     = {k: v.copy() for k, v in (heave_dict or {}).items()}
        self._decoupled = {k: v.copy() for k, v in (decoupled_dict or {}).items()}
        self._rebuild_name_lists()
        self._updating = False
        self.set_info(section_info.HARDPOINTS)
        self._build()

    def _rebuild_name_lists(self):
        """Re-derive self._all_names and the per-category index ranges."""
        self._names     = list(self._hp.keys())
        self._arb_names = list(self._arb.keys())
        self._heave_names     = list(self._heave.keys())
        self._decoupled_names = list(self._decoupled.keys())
        self._all_names = (self._names + self._arb_names +
                            self._heave_names + self._decoupled_names)

    def _category_for(self, name: str) -> str:
        if name in self._hp:        return 'corner'
        if name in self._arb:       return 'arb'
        if name in self._heave:     return 'heave'
        if name in self._decoupled: return 'decoupled'
        return 'corner'   # fallback (shouldn't happen if names list is correct)

    def refresh(self, hp_dict: dict,
                arb_dict: dict | None = None,
                heave_dict: dict | None = None,
                decoupled_dict: dict | None = None):
        self._hp = {k: v.copy() for k, v in hp_dict.items()}
        if arb_dict is not None:
            self._arb = {k: v.copy() for k, v in arb_dict.items()}
        if heave_dict is not None:
            self._heave = {k: v.copy() for k, v in heave_dict.items()}
        if decoupled_dict is not None:
            self._decoupled = {k: v.copy() for k, v in decoupled_dict.items()}
        # If the key set changed (e.g. topology switch swaps the hp dict's
        # keys entirely), rebuild the table.
        prev_all = self._all_names
        self._rebuild_name_lists()
        if self._all_names != prev_all:
            try:
                old = self._table
                old.setParent(None)
                old.deleteLater()
            except Exception:
                pass
            self._build()
        self._fill()

    def highlight_row(self, name: str):
        if name in self._all_names:
            self._table.selectRow(self._all_names.index(name))

    def _build(self):
        n = len(self._all_names)
        self._table = QTableWidget(n, 4)
        self._table.setHorizontalHeaderLabels(['Point', 'X', 'Y', 'Z'])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in (1, 2, 3):
            self._table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
            self._table.setColumnWidth(col, 66)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(
            QTableWidget.EditTrigger.DoubleClicked |
            QTableWidget.EditTrigger.AnyKeyPressed)
        self._table.cellChanged.connect(self._on_cell)
        self._table.currentCellChanged.connect(
            lambda r, *_: self.row_selected.emit(self._all_names[r])
            if 0 <= r < len(self._all_names) else None)
        self.add_widget(self._table)
        self._fill()

    def _fill(self):
        self._updating = True
        for row, name in enumerate(self._all_names):
            cat = self._category_for(name)
            if cat == 'corner':
                vals_mm = self._hp[name] * 1000.0
                is_chassis = name in CHASSIS_PTS
                col_str = C_BLUE if is_chassis else C_RED
            elif cat == 'arb':
                vals_mm = self._arb[name] * 1000.0
                col_str = '#FFB300'   # amber for ARB
            elif cat == 'heave':
                vals_mm = self._heave[name] * 1000.0
                col_str = '#42E0A0'   # green for HEAVE 3rd element
            else:   # decoupled
                vals_mm = self._decoupled[name] * 1000.0
                col_str = '#B85FFF'   # purple for DECOUPLED bellcrank/dampers

            item = QTableWidgetItem(name)
            item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            item.setForeground(QColor(col_str))
            self._table.setItem(row, 0, item)

            for col, v in enumerate(vals_mm, start=1):
                cell = QTableWidgetItem(f'{v:.2f}')
                cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._table.setItem(row, col, cell)

        self._table.resizeRowsToContents()
        self._updating = False

    def _on_cell(self, row, col):
        if self._updating or col == 0 or row >= len(self._all_names):
            return
        name = self._all_names[row]
        item = self._table.item(row, col)
        if not item: return
        try:
            val_m = float(item.text()) / 1000.0
            cat = self._category_for(name)
            target = {'corner': self._hp, 'arb': self._arb,
                      'heave': self._heave,
                      'decoupled': self._decoupled}[cat]
            pt = target[name].copy()
            pt[col - 1] = val_m
            target[name] = pt
            # Emit only the dict for the edited CATEGORY; receiver routes
            # it to the right stored dict on MainWindow.
            self.hp_changed.emit(
                {k: v.copy() for k, v in target.items()}, cat)
        except ValueError:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE VALUES TABLE
# ══════════════════════════════════════════════════════════════════════════════

class ValuesPanel(CollapsibleSection):
    """Button that opens a popup showing live metric values for all 4 corners."""

    _CORNERS = ('FL', 'FR', 'RL', 'RR')
    _EXTRA_ROWS = [
        ('max_bump_travel',  'Max Bump Travel',  'mm'),
        ('max_droop_travel', 'Max Droop Travel', 'mm'),
    ]

    def __init__(self):
        super().__init__('Live Values')
        self.set_info(section_info.VALUES)
        self._stroke_mm = 55.0
        # Per-axle sag in shock frame, mm (computed from preload + dynamics).
        # NOT a user input anymore.
        self._sag_shock_front_mm = 0.0
        self._sag_shock_rear_mm  = 0.0
        self._last_data: dict = {}
        self._dlg: QDialog | None = None
        # Single-model: only list metrics whose mechanism exists on the car.
        self._topology_flavors: set = {'standard'}
        self._build()

    def _visible_catalog(self):
        return [e for e in CATALOG
                if 'topologies' not in e
                or (e['topologies'] & self._topology_flavors)]

    def set_topology_flavors(self, flavors: set):
        flavors = set(flavors) if flavors else {'standard'}
        if flavors == self._topology_flavors:
            return
        self._topology_flavors = flavors
        # Row layout changes — drop the popup; it rebuilds on next open.
        if self._dlg is not None:
            try:
                self._dlg.close()
            except Exception:
                pass
            self._dlg = None
            self._popup_table = None

    def update_values(self, all_corners: dict):
        """Update stored data and refresh popup if open.

        all_corners: {'FL': {metric: val, ...}, 'FR': {...}, ...}
        """
        self._last_data = all_corners
        if self._dlg is not None and self._dlg.isVisible():
            self._fill_table()

    def update_damper_params(self,
                             stroke_mm: float,
                             sag_shock_front_mm: float,
                             sag_shock_rear_mm: float):
        """
        stroke_mm           : total damper stroke in shock frame (mm)
        sag_shock_front_mm  : computed static sag (front axle, shock frame, mm)
        sag_shock_rear_mm   : computed static sag (rear axle, shock frame, mm)

        Max bump travel (compression remaining) at wheel:
            = (stroke − sag) / MR
        Max droop travel (extension remaining) at wheel:
            = sag / MR
        """
        self._stroke_mm = stroke_mm
        self._sag_shock_front_mm = sag_shock_front_mm
        self._sag_shock_rear_mm  = sag_shock_rear_mm
        # Repaint the popup if open so numbers reflect the new sag.
        if self._dlg is not None and self._dlg.isVisible():
            self._fill_table()

    def _build(self):
        btn = QPushButton('Show Live Values')
        btn.setStyleSheet(
            'QPushButton { background: #1a1a1a; color: #e0e0e0; '
            'border: 1px solid #444; padding: 6px 16px; border-radius: 3px; '
            'font-weight: bold; }'
            'QPushButton:hover { background: #2a2a2a; }')
        btn.clicked.connect(self._show_popup)
        self.add_widget(btn)

    def _show_popup(self):
        if self._dlg is not None and self._dlg.isVisible():
            self._dlg.raise_()
            return

        self._dlg = QDialog(self)
        self._dlg.setWindowTitle('Live Kinematic Values — All Corners')
        self._dlg.resize(620, 700)
        self._dlg.setStyleSheet(
            'QDialog { background: #0a0a0a; }'
            'QTableWidget { background: #0a0a0a; color: #e0e0e0; '
            'gridline-color: #222; font-size: 11px; }'
            'QHeaderView::section { background: #111; color: #aaa; '
            'border: 1px solid #222; padding: 3px; font-weight: bold; }')

        lay = QVBoxLayout(self._dlg)
        visible = self._visible_catalog()
        n_rows = len(visible) + len(self._EXTRA_ROWS)
        n_cols = 1 + len(self._CORNERS) + 1
        self._popup_table = QTableWidget(n_rows, n_cols)
        headers = ['Metric'] + list(self._CORNERS) + ['Unit']
        self._popup_table.setHorizontalHeaderLabels(headers)
        self._popup_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        for ci in range(1, 1 + len(self._CORNERS)):
            self._popup_table.horizontalHeader().setSectionResizeMode(
                ci, QHeaderView.ResizeMode.Fixed)
            self._popup_table.setColumnWidth(ci, 82)
        self._popup_table.horizontalHeader().setSectionResizeMode(
            n_cols - 1, QHeaderView.ResizeMode.Fixed)
        self._popup_table.setColumnWidth(n_cols - 1, 40)
        self._popup_table.verticalHeader().setVisible(False)
        self._popup_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._popup_table.setAlternatingRowColors(True)

        # Populate labels
        prev_cat = None
        for row, entry in enumerate(visible):
            cat = entry['category']
            ni = QTableWidgetItem(entry['label'])
            ni.setFlags(Qt.ItemFlag.ItemIsEnabled)
            if cat != prev_cat:
                ni.setForeground(QColor('#aaaaaa'))
            prev_cat = cat
            self._popup_table.setItem(row, 0, ni)
            for ci in range(len(self._CORNERS)):
                vi = QTableWidgetItem('—')
                vi.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                vi.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self._popup_table.setItem(row, 1 + ci, vi)
            ui = QTableWidgetItem(entry['unit'])
            ui.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            ui.setFlags(Qt.ItemFlag.ItemIsEnabled)
            ui.setForeground(QColor(C_SUB))
            self._popup_table.setItem(row, n_cols - 1, ui)

        base = len(visible)
        for i, (key, label, unit) in enumerate(self._EXTRA_ROWS):
            row = base + i
            ni = QTableWidgetItem(label)
            ni.setFlags(Qt.ItemFlag.ItemIsEnabled)
            ni.setForeground(QColor('#aaaaaa'))
            self._popup_table.setItem(row, 0, ni)
            for ci in range(len(self._CORNERS)):
                vi = QTableWidgetItem('—')
                vi.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                vi.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self._popup_table.setItem(row, 1 + ci, vi)
            ui = QTableWidgetItem(unit)
            ui.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            ui.setFlags(Qt.ItemFlag.ItemIsEnabled)
            ui.setForeground(QColor(C_SUB))
            self._popup_table.setItem(row, n_cols - 1, ui)

        self._popup_table.resizeRowsToContents()
        lay.addWidget(self._popup_table)

        self._fill_table()
        self._dlg.show()

    @staticmethod
    def _avg_finite(values):
        """Mean of the finite entries in `values`; NaN if none are finite.
        Used for axle/vehicle scope reduction so per-corner numerical
        drift in fns like anti-dive doesn't show up as 4 "almost equal"
        cells."""
        finite = [v for v in values if not np.isnan(v)]
        if not finite:
            return float('nan')
        return float(sum(finite) / len(finite))

    def _fill_table(self):
        """Render the live-values table.  Each catalog entry has a
        ``scope`` controlling how its single value is laid out across
        the four corner columns:

            'corner'  → four independent values, one per cell.
            'axle'    → two values (one per axle); FL=FR=front, RL=RR=rear.
                        Per-corner L/R drift is averaged out.
            'front'   → one value at the front axle; FL=FR=value, RL/RR blank.
            'rear'    → one value at the rear axle; RL=RR=value, FL/FR blank.
            'vehicle' → one value across all four cells.
        """
        if self._popup_table is None:
            return

        for row, entry in enumerate(self._visible_catalog()):
            key = entry['key']
            scope = entry.get('scope', 'corner')

            # Pull the per-corner samples once.
            samples = {c: self._last_data.get(c, {}).get(key, float('nan'))
                       for c in self._CORNERS}

            # Decide what to write into each cell based on scope.
            if scope == 'corner':
                cell_vals = samples

            elif scope == 'axle':
                # One value per axle, averaged across L/R to cancel
                # numerical drift in the per-corner fn.
                front = self._avg_finite([samples['FL'], samples['FR']])
                rear  = self._avg_finite([samples['RL'], samples['RR']])
                cell_vals = {'FL': front, 'FR': front,
                             'RL': rear,  'RR': rear}

            elif scope == 'front':
                # Front-axle property — single number.  Rear cells blank.
                front = self._avg_finite([samples['FL'], samples['FR']])
                cell_vals = {'FL': front, 'FR': front,
                             'RL': float('nan'), 'RR': float('nan')}

            elif scope == 'rear':
                # Rear-axle property — single number.  Front cells blank.
                rear  = self._avg_finite([samples['RL'], samples['RR']])
                cell_vals = {'FL': float('nan'), 'FR': float('nan'),
                             'RL': rear,  'RR': rear}

            elif scope == 'vehicle':
                # Single car-level number.  Same in every cell.
                avg = self._avg_finite(list(samples.values()))
                cell_vals = {c: avg for c in self._CORNERS}

            else:   # unknown scope → fall back to per-corner
                cell_vals = samples

            for ci, corner in enumerate(self._CORNERS):
                v = cell_vals[corner]
                item = self._popup_table.item(row, 1 + ci)
                if item:
                    item.setText(f'{v:.4f}' if not np.isnan(v) else '—')

        # ── Extra rows (bump/droop travel) are always per-corner ────────
        base = len(self._visible_catalog())
        for ci, corner in enumerate(self._CORNERS):
            vals = self._last_data.get(corner, {})
            mr = vals.get('motion_ratio', float('nan'))
            sag_shock = (self._sag_shock_front_mm if corner in ('FL', 'FR')
                         else self._sag_shock_rear_mm)
            if not np.isnan(mr) and mr > 1e-6:
                # Physical: at static, damper is compressed by sag_shock from
                # full droop.  Remaining bump room = stroke − sag, remaining
                # droop room = sag.  Convert to wheel frame: / MR.
                sag_clamped = max(0.0, min(self._stroke_mm, sag_shock))
                bump  = (self._stroke_mm - sag_clamped) / mr
                droop = sag_clamped / mr
            else:
                bump = droop = float('nan')
            for ei, (ek, _, _) in enumerate(self._EXTRA_ROWS):
                val = bump if 'bump' in ek else droop
                item = self._popup_table.item(base + ei, 1 + ci)
                if item:
                    item.setText(f'{val:.1f}' if not np.isnan(val) else '—')


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH PICKER
# ══════════════════════════════════════════════════════════════════════════════

class GraphPickerPanel(CollapsibleSection):
    """Select which metrics to plot and which corners to show."""
    selection_changed = pyqtSignal(list)
    corners_changed   = pyqtSignal(list)

    def __init__(self):
        super().__init__('Graph Selection', header_color='#cccccc')
        self.set_info(section_info.GRAPH_PICKER)
        self._build()

    def get_selected_keys(self) -> list:
        keys = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item and item.checkState() == Qt.CheckState.Checked:
                keys.append(item.data(Qt.ItemDataRole.UserRole))
        return keys

    def get_selected_corners(self) -> list:
        return [lbl for lbl, cb in self._corner_cbs.items() if cb.isChecked()]

    def _build(self):
        # ── corner selector ───────────────────────────────────────────────
        corner_row = QHBoxLayout()
        corner_row.addWidget(QLabel('Corners:'))
        self._corner_cbs = {}
        for lbl in ('FL', 'FR', 'RL', 'RR'):
            cb = QCheckBox(lbl)
            cb.setChecked(True)
            cb.stateChanged.connect(
                lambda _, self=self: self.corners_changed.emit(self.get_selected_corners()))
            self._corner_cbs[lbl] = cb
            corner_row.addWidget(cb)
        corner_row.addStretch()
        self.add_layout(corner_row)

        # ── metric picker list ────────────────────────────────────────────
        # Topology-filtered (single-model GUI): CATALOG entries tagged with a
        # 'topologies' set are listed ONLY when the active car has a matching
        # flavour ('standard' / 'decoupled' / 'heave_tbar' / 'tbar').
        # Untagged entries are generic and always listed.
        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.setMaximumHeight(200)
        self._topology_flavors: set = {'standard'}
        self._checked_keys: set = set(DEFAULT_Y_KEYS)
        self._rebuild_list()

        self._list.itemChanged.connect(self._on_item_changed)
        self.add_widget(self._list)

        row = QHBoxLayout()
        for label, fn in [('All', self._all), ('None', self._none)]:
            b = QPushButton(label); b.setFixedWidth(55)
            b.clicked.connect(fn); row.addWidget(b)
        row.addStretch()
        self.add_layout(row)

    def _all(self):  self._set(Qt.CheckState.Checked)
    def _none(self): self._set(Qt.CheckState.Unchecked)

    def _set(self, state):
        self._list.blockSignals(True)
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item and item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                item.setCheckState(state)
                key = item.data(Qt.ItemDataRole.UserRole)
                if state == Qt.CheckState.Checked:
                    self._checked_keys.add(key)
                else:
                    self._checked_keys.discard(key)
        self._list.blockSignals(False)
        self.selection_changed.emit(self.get_selected_keys())

    def _on_item_changed(self, item):
        key = item.data(Qt.ItemDataRole.UserRole)
        if key is not None:
            if item.checkState() == Qt.CheckState.Checked:
                self._checked_keys.add(key)
            else:
                self._checked_keys.discard(key)
        self.selection_changed.emit(self.get_selected_keys())

    def _rebuild_list(self):
        """(Re)populate the metric list for the active topology flavours.
        Check states survive rebuilds via self._checked_keys."""
        self._list.blockSignals(True)
        self._list.clear()
        prev_cat = None
        for entry in CATALOG:
            topos = entry.get('topologies')
            if topos is not None and not (topos & self._topology_flavors):
                continue   # metric's mechanism doesn't exist on this car
            if entry['category'] != prev_cat:
                hdr = QListWidgetItem(f'  {entry["category"]}')
                hdr.setFlags(Qt.ItemFlag.NoItemFlags)
                hdr.setForeground(QColor('#aaaaaa'))
                f = hdr.font(); f.setBold(True); hdr.setFont(f)
                self._list.addItem(hdr)
                prev_cat = entry['category']
            item = QListWidgetItem(f'  {entry["label"]}  ({entry["unit"]})')
            item.setData(Qt.ItemDataRole.UserRole, entry['key'])
            item.setCheckState(
                Qt.CheckState.Checked if entry['key'] in self._checked_keys
                else Qt.CheckState.Unchecked)
            self._list.addItem(item)
        self._list.blockSignals(False)

    def set_topology_flavors(self, flavors: set):
        """Filter the metric list to the active topology's mechanisms and
        re-emit the (possibly reduced) selection."""
        flavors = set(flavors) if flavors else {'standard'}
        if flavors == self._topology_flavors:
            return
        self._topology_flavors = flavors
        self._rebuild_list()
        self.selection_changed.emit(self.get_selected_keys())


# ══════════════════════════════════════════════════════════════════════════════
#  ALIGNMENT PANEL
# ══════════════════════════════════════════════════════════════════════════════

class AlignmentPanel(CollapsibleSection):
    """
    Static alignment targets. Clicking Apply triggers a Newton solve in
    main_window to adjust tie_rod_inner (toe) and UCA pivot X (camber).

    Signals:
        alignment_changed(dict) — {'front_toe_deg', 'front_camber_deg',
                                    'rear_toe_deg',  'rear_camber_deg'}
    """
    alignment_changed = pyqtSignal(dict)

    def __init__(self):
        super().__init__('Alignment (Static)')
        self.set_info(section_info.ALIGNMENT)
        self._build()

    def get_params(self) -> dict:
        return {
            'front_toe_deg':    self._ft.value(),
            'front_camber_deg': self._fc.value(),
            'rear_toe_deg':     self._rt.value(),
            'rear_camber_deg':  self._rc.value(),
        }

    def _build(self):
        grid = QGridLayout(); grid.setSpacing(4)

        grid.addWidget(QLabel('Front toe:'),    0, 0)
        self._ft = _spin(-90.0, 90.0, 0.0, ' deg', dec=2, step=0.1)
        grid.addWidget(self._ft, 0, 1)

        grid.addWidget(QLabel('Front camber:'), 1, 0)
        self._fc = _spin(-90.0, 90.0, 0.0, ' deg', dec=2, step=0.1)
        grid.addWidget(self._fc, 1, 1)

        grid.addWidget(QLabel('Rear toe:'),     2, 0)
        self._rt = _spin(-90.0, 90.0, 0.0, ' deg', dec=2, step=0.1)
        grid.addWidget(self._rt, 2, 1)

        grid.addWidget(QLabel('Rear camber:'),  3, 0)
        self._rc = _spin(-90.0, 90.0, 0.0, ' deg', dec=2, step=0.1)
        grid.addWidget(self._rc, 3, 1)

        self.add_layout(grid)

        btn = QPushButton('Apply Alignment')
        btn.clicked.connect(lambda: self.alignment_changed.emit(self.get_params()))
        self.add_widget(btn)

        note = QLabel('Adjusts tie-rod inner X (toe) and UCA pivot X (camber) '
                      'via Newton solve at static position.')
        note.setWordWrap(True)
        note.setStyleSheet('color: #888888; font-size: 11px;')
        self.add_widget(note)


# ── helper ─────────────────────────────────────────────────────────────────────

class _NoScrollSpin(QDoubleSpinBox):
    """QDoubleSpinBox that ignores scroll events unless explicitly focused."""
    def wheelEvent(self, e):
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()


class _NoScrollCombo(QComboBox):
    """QComboBox that ignores scroll events unless explicitly focused."""
    def wheelEvent(self, e):
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()


def _spin(lo, hi, val, suffix='', dec=1, step=5.0) -> QDoubleSpinBox:
    sb = _NoScrollSpin()
    # NOTE: setDecimals() must come BEFORE setValue() — otherwise the
    # value is silently rounded to the default 2-decimal precision before
    # the higher-precision setting takes effect (e.g. 1.225 → 1.23 with
    # dec=3).  Same with setRange when extreme values are involved.
    sb.setDecimals(dec)
    sb.setRange(lo, hi)
    sb.setValue(val)
    sb.setSuffix(suffix)
    sb.setSingleStep(step)
    sb.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    return sb


# ══════════════════════════════════════════════════════════════════════════════
#  INVERSE KINEMATICS PANEL
# ══════════════════════════════════════════════════════════════════════════════

# Metrics the IK panel exposes as targets
IK_METRICS = [
    ('anti_dive',    'Anti-Dive',          '%',   0, 100),
    ('anti_squat',   'Anti-Squat',         '%',   0, 100),
    ('anti_lift',    'Anti-Lift',          '%',   0, 100),
    ('camber',       'Camber Angle',       '°',  -5, 2),
    ('toe',          'Bump Steer (Toe)',   '°',  -2, 2),
    ('ackermann',    'Ackermann %',        '%',  50, 150),
    ('rc_height',    'Roll Centre Height', 'mm', -20, 80),
    ('caster',       'Caster Angle',       '°',   0, 15),
    ('trail',        'Caster Trail',       'mm', -10, 60),
    ('motion_ratio', 'Motion Ratio',       '-',   0.3, 1.5),
    ('arb_mr',       'ARB Motion Ratio',  '-',   0.1, 2.0),
]

# Hardpoints the user can select (inboard chassis points that matter)
IK_HARDPOINTS = [
    'uca_front', 'uca_rear', 'uca_outer',
    'lca_front', 'lca_rear', 'lca_outer',
    'tie_rod_inner', 'tie_rod_outer',
    'pushrod_outer', 'pushrod_inner',
    'rocker_pivot', 'rocker_spring_pt',
    'arb_drop_top', 'arb_arm_end', 'arb_pivot',
]


class _SolutionPickerDialog(QDialog):
    """Modal dialog showing multiple IK solutions for the user to pick from."""

    def __init__(self, solutions: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle('Pick a Solution')
        self.setMinimumSize(650, 400)
        self._solutions = solutions
        self._chosen_idx = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('Multiple solutions found. Pick one to apply:'))

        # ── Summary table ────────────────────────────────────────────────
        self._table = QTableWidget(len(solutions), 0)
        self._table.verticalHeader().setVisible(False)

        # Build columns: Bounds | Max Error | Total Movement | per-variable deltas
        primary = solutions[0].get('primary_metric', '?')
        all_vars = solutions[0].get('variables', [])
        var_labels = [v.label for v in all_vars]

        headers = ['Bounds', f'Max Err ({primary})', 'Total Move']
        headers.extend(var_labels)
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)

        for row, sol in enumerate(solutions):
            bound_mm = sol.get('bound_label', '?')
            max_err = sol.get('primary_max_error', 0.0)
            deltas = sol.get('deltas_mm', [])
            total_move = sum(abs(float(d)) for d in deltas)

            self._table.setItem(row, 0, QTableWidgetItem(str(bound_mm)))
            self._table.setItem(row, 1, QTableWidgetItem(f'{max_err:.3f}'))
            self._table.setItem(row, 2, QTableWidgetItem(f'{total_move:.1f} mm'))

            for c, d in enumerate(deltas):
                item = QTableWidgetItem(f'{float(d):+.2f}')
                self._table.setItem(row, 3 + c, item)

        self._table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection)
        self._table.selectRow(0)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self._table)

        # ── Buttons ──────────────────────────────────────────────────────
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self.setStyleSheet("""
            QDialog { background: #111111; color: #e0e0e0; }
            QTableWidget { background: #0a0a0a; color: #e0e0e0;
                           gridline-color: #2a2a2a; }
            QHeaderView::section { background: #1a1a1a; color: #ccc;
                                   border: 1px solid #2a2a2a; padding: 3px; }
            QLabel { color: #e0e0e0; }
            QPushButton { background: #333; color: #e0e0e0; padding: 5px 12px;
                          border: 1px solid #555; border-radius: 3px; }
            QPushButton:hover { background: #444; }
        """)

    def chosen_result(self) -> dict | None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return None
        idx = rows[0].row()
        return self._solutions[idx]


class InverseKinematicsPanel(CollapsibleSection):
    """
    Inverse Kinematics panel.

    Workflow: select metric → set target value → select hardpoints → solve.
    Emits solve_requested with (metric_key, target_value, hardpoints, bound_mm, axle).
    """
    solve_requested = pyqtSignal(dict)   # full spec dict
    apply_requested = pyqtSignal(dict)   # optimised hp dict + axle

    def __init__(self):
        super().__init__('Inverse Kinematics', header_color='#cccccc')
        self.set_info(section_info.INVERSE_KINEMATICS)
        self._build()

    def _build(self):
        grid = QGridLayout()
        grid.setSpacing(4)

        r = 0
        # Axle selector
        grid.addWidget(QLabel('Axle:'), r, 0)
        self._axle = _NoScrollCombo()
        self._axle.addItems(['Front', 'Rear'])
        grid.addWidget(self._axle, r, 1); r += 1

        # Motion type
        grid.addWidget(QLabel('Motion:'), r, 0)
        self._motion = _NoScrollCombo()
        self._motion.addItems(['Heave', 'Roll', 'Pitch', 'Steer'])
        grid.addWidget(self._motion, r, 1); r += 1

        # Range — auto-clamps to damper limits when set_damper_limits() called
        grid.addWidget(QLabel('Min:'), r, 0)
        self._range_lo = _spin(-1e6, 1e6, -18, ' mm', dec=0, step=5)
        grid.addWidget(self._range_lo, r, 1); r += 1
        grid.addWidget(QLabel('Max:'), r, 0)
        self._range_hi = _spin(-1e6, 1e6, 42, ' mm', dec=0, step=5)
        grid.addWidget(self._range_hi, r, 1); r += 1

        # Metric selector
        grid.addWidget(QLabel('Target Metric:'), r, 0)
        self._metric = _NoScrollCombo()
        for key, label, unit, *_ in IK_METRICS:
            self._metric.addItem(f'{label} ({unit})', key)
        self._metric.currentIndexChanged.connect(self._on_metric_change)
        grid.addWidget(self._metric, r, 1); r += 1

        # Target start / end (linear ramp across travel range)
        grid.addWidget(QLabel('Target @ Min:'), r, 0)
        self._target_lo = _spin(-500, 500, 30.0, dec=2, step=0.5)
        grid.addWidget(self._target_lo, r, 1); r += 1

        grid.addWidget(QLabel('Target @ Max:'), r, 0)
        self._target_hi = _spin(-500, 500, 30.0, dec=2, step=0.5)
        grid.addWidget(self._target_hi, r, 1); r += 1

        # Curve shape between the two endpoints — a NONLINEAR target (e.g.
        # progressive/exponential MR) for a rising-rate that resists aero
        # heave and holds ride height.  Linear reproduces the old ramp.
        grid.addWidget(QLabel('Curve:'), r, 0)
        self._curve = _NoScrollCombo()
        self._curve.addItems(['Linear', 'Progressive', 'Digressive',
                              'Exponential'])
        self._curve.setToolTip(
            'Shape of the target between @Min (droop) and @Max (bump):\n'
            ' Linear      — straight ramp\n'
            ' Progressive — rises slowly then fast (τ^p): most stiffening DEEP '
            'in bump → best for holding ride height under downforce\n'
            ' Digressive  — rises fast then plateaus\n'
            ' Exponential — smooth exp curve')
        self._curve.currentIndexChanged.connect(self._on_curve_change)
        grid.addWidget(self._curve, r, 1); r += 1
        grid.addWidget(QLabel('Curvature:'), r, 0)
        self._curvature = _spin(0.2, 8.0, 2.0, dec=1, step=0.2)
        self._curvature.setToolTip('Strength of the nonlinearity (power p / '
                                   'exp k).  1 = linear-ish; higher = sharper. '
                                   'Ignored for Linear.')
        self._curvature.setEnabled(False)
        grid.addWidget(self._curvature, r, 1); r += 1

        # Bound (how far points can move)
        grid.addWidget(QLabel('Max Movement:'), r, 0)
        self._bound = _spin(0.01, 1e6, 10, ' mm', dec=1, step=1)
        grid.addWidget(self._bound, r, 1); r += 1

        # Method
        grid.addWidget(QLabel('Method:'), r, 0)
        self._method = _NoScrollCombo()
        self._method.addItems(['staged', 'hybrid', 'local', 'global'])
        grid.addWidget(self._method, r, 1); r += 1

        self.add_layout(grid)

        # ── Lock constraints: keep other metrics from drifting ────────────
        lock_hdr = QHBoxLayout()
        lock_label = QLabel('Lock (hold within tolerance):')
        lock_label.setStyleSheet(f'color: {C_BLUE}; font-weight: bold; font-size: 11px;')
        lock_hdr.addWidget(lock_label)
        lock_hdr.addStretch()
        lock_hdr.addWidget(QLabel('Tol:'))
        self._lock_tol = _spin(0.001, 1e6, 5.0, '', dec=1, step=1.0)
        self._lock_tol.setMaximumWidth(55)
        self._lock_tol.setToolTip(
            'Tolerance band for locked metrics.\n'
            'Units match each metric (°, %, mm).\n'
            'Larger = more freedom for primary target.')
        lock_hdr.addWidget(self._lock_tol)
        self.add_layout(lock_hdr)

        self._lock_checks: dict[str, QCheckBox] = {}
        lock_row = QGridLayout(); lock_row.setSpacing(2)
        # Default: lock everything except camber (the usual primary target)
        _default_locks = {'anti_dive', 'anti_squat', 'anti_lift', 'toe',
                          'rc_height', 'caster', 'trail', 'motion_ratio',
                          'arb_mr'}
        for i, (key, label, unit, *_) in enumerate(IK_METRICS):
            cb = QCheckBox(label)
            cb.setChecked(key in _default_locks)
            cb.setStyleSheet('font-size: 11px;')
            self._lock_checks[key] = cb
            lock_row.addWidget(cb, i // 2, i % 2)
        self.add_layout(lock_row)

        # ── Hardpoint selection ───────────────────────────────────────────
        hp_label = QLabel('Hardpoints to adjust:')
        hp_label.setStyleSheet(f'color: {C_BLUE}; font-weight: bold; font-size: 11px;')
        self.add_widget(hp_label)

        self._hp_checks: dict[str, QCheckBox] = {}
        _default_hp = {'uca_front', 'uca_rear', 'uca_outer',
                        'lca_front', 'lca_rear', 'lca_outer'}
        for hp_name in IK_HARDPOINTS:
            cb = QCheckBox(hp_name.replace('_', ' ').title())
            cb.setChecked(hp_name in _default_hp)
            self._hp_checks[hp_name] = cb
            self.add_widget(cb)

        # Coordinate axes to vary
        ax_row = QHBoxLayout()
        ax_row.addWidget(QLabel('Coords:'))
        self._ax_x = QCheckBox('X'); self._ax_x.setChecked(True)
        self._ax_y = QCheckBox('Y'); self._ax_y.setChecked(True)
        self._ax_z = QCheckBox('Z'); self._ax_z.setChecked(True)
        ax_row.addWidget(self._ax_x)
        ax_row.addWidget(self._ax_y)
        ax_row.addWidget(self._ax_z)
        self.add_layout(ax_row)

        # Auto-select preset button
        self._auto_btn = QPushButton('Auto-Select Hardpoints')
        self._auto_btn.clicked.connect(self._auto_select)
        self._auto_btn.setStyleSheet(f'background: #1a1a1a; color: {C_BLUE}; '
                                     'border: 1px solid #444; padding: 4px;')
        self.add_widget(self._auto_btn)

        # ── Tube diameters (for collision detection) ─────────────────────
        tube_label = QLabel('Tube OD (collision):')
        tube_label.setStyleSheet(f'color: {C_BLUE}; font-weight: bold; font-size: 11px;')
        self.add_widget(tube_label)
        tube_grid = QGridLayout(); tube_grid.setSpacing(2)
        self._tube_od_spins: dict[str, QDoubleSpinBox] = {}
        _tube_defaults = [
            ('UCA arms',      ['uca_front_arm', 'uca_rear_arm'], 25.4),
            ('LCA arms',      ['lca_front_arm', 'lca_rear_arm'], 25.4),
            ('Tie rod',       ['tie_rod'],                       19.0),
            ('Pushrod',       ['pushrod'],                       19.0),
            ('Spring/damper', ['spring_damper'],                  50.8),
        ]
        for i, (label, keys, default_mm) in enumerate(_tube_defaults):
            tube_grid.addWidget(QLabel(label), i, 0)
            sp = _spin(0, 1e6, default_mm, ' mm', dec=1, step=1.0)
            sp.setMaximumWidth(80)
            tube_grid.addWidget(sp, i, 1)
            for k in keys:
                self._tube_od_spins[k] = sp   # shared spinner for grouped
        self.add_layout(tube_grid)

        # Solve button
        self._solve_btn = QPushButton('  Solve  ')
        self._solve_btn.setStyleSheet(
            'background: #555555; color: white; font-weight: bold; '
            'border-radius: 4px; padding: 6px; font-size: 13px;')
        self._solve_btn.clicked.connect(self._on_solve)
        self.add_widget(self._solve_btn)

        # Status / results area
        self._status = QLabel('Ready')
        self._status.setWordWrap(True)
        self._status.setStyleSheet('color: #aaa; font-size: 11px;')
        self.add_widget(self._status)

        # Results table (hidden until solve completes)
        self._results_table = QTableWidget(0, 3)
        self._results_table.setHorizontalHeaderLabels(['Variable', 'Delta', 'New Value'])
        self._results_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self._results_table.verticalHeader().setVisible(False)
        self._results_table.setMaximumHeight(200)
        self._results_table.setVisible(False)
        self.add_widget(self._results_table)

        # Apply button (hidden until solve completes)
        self._apply_btn = QPushButton('Apply to Model')
        self._apply_btn.setStyleSheet(
            'background: #444444; color: white; font-weight: bold; '
            'border-radius: 4px; padding: 5px;')
        self._apply_btn.setVisible(False)
        self._apply_btn.clicked.connect(self._on_apply)
        self.add_widget(self._apply_btn)

        # "Find Solutions" button (hidden until solve shows target not met)
        self._find_btn = QPushButton('Find Solutions (wider search)')
        self._find_btn.setStyleSheet(
            'background: #555; color: white; font-weight: bold; '
            'border-radius: 4px; padding: 5px;')
        self._find_btn.setVisible(False)
        self._find_btn.clicked.connect(self._on_find_solutions)
        self.add_widget(self._find_btn)

        self._last_result = None
        # Default to Camber Angle with 0° @ min, -2° @ max
        self._metric.setCurrentIndex(3)   # camber
        self._on_metric_change(3)
        self._target_lo.setValue(0.0)
        self._target_hi.setValue(-2.0)

    def _on_curve_change(self, _idx):
        # curvature only matters for the nonlinear shapes
        self._curvature.setEnabled(self._curve.currentText() != 'Linear')

    def _on_metric_change(self, _idx):
        key = self._metric.currentData()
        for mkey, _, unit, lo, hi in IK_METRICS:
            if mkey == key:
                self._target_lo.setRange(lo - 50, hi + 50)
                self._target_hi.setRange(lo - 50, hi + 50)
                self._target_lo.setSuffix(f' {unit}')
                self._target_hi.setSuffix(f' {unit}')
                # Default: constant target at a sensible mid-range value
                default = (lo + hi) / 2
                self._target_lo.setValue(default)
                self._target_hi.setValue(default)
                break
        # Ackermann only works in steer mode — auto-switch
        if key == 'ackermann':
            self._motion.setCurrentText('Steer')

    def _auto_select(self):
        """Auto-select hardpoints based on the chosen metric."""
        from vahan.optimizer import PRESETS
        key = self._metric.currentData()
        # For Ackermann, default to rack_position (tie_rod_inner only)
        # so the optimizer moves the rack without changing tie_rod_outer.
        preset_key = 'rack_position' if key == 'ackermann' else key
        preset = PRESETS.get(preset_key, [])
        # Uncheck all first
        for cb in self._hp_checks.values():
            cb.setChecked(False)
        # Check relevant ones
        points_in_preset = {s['point'] for s in preset}
        for hp_name, cb in self._hp_checks.items():
            if hp_name in points_in_preset:
                cb.setChecked(True)
        # Also set coordinate axes from preset
        coords_in_preset = {s['coord'] for s in preset}
        self._ax_x.setChecked(0 in coords_in_preset)
        self._ax_y.setChecked(1 in coords_in_preset)
        self._ax_z.setChecked(2 in coords_in_preset)

    def get_spec(self) -> dict:
        """Collect the full solve specification from the UI."""
        coords = []
        if self._ax_x.isChecked(): coords.append(0)
        if self._ax_y.isChecked(): coords.append(1)
        if self._ax_z.isChecked(): coords.append(2)

        hp_names = [n for n, cb in self._hp_checks.items() if cb.isChecked()]

        # Locked metrics: those checked AND not the primary target
        primary = self._metric.currentData()
        lock_metrics = [k for k, cb in self._lock_checks.items()
                        if cb.isChecked() and k != primary]

        # Tube ODs for collision detection (mm → metres)
        tube_od = {}
        for key, sp in self._tube_od_spins.items():
            tube_od[key] = sp.value() / 1000.0

        return {
            'axle':         'front' if self._axle.currentIndex() == 0 else 'rear',
            'motion':       self._motion.currentText().lower(),
            'range_lo':     self._range_lo.value(),
            'range_hi':     self._range_hi.value(),
            'metric_key':   self._metric.currentData(),
            'target_lo':    self._target_lo.value(),
            'target_hi':    self._target_hi.value(),
            'target_shape': self._curve.currentText().lower(),
            'target_curvature': self._curvature.value(),
            'bound_mm':     self._bound.value(),
            'method':       self._method.currentText(),
            'hp_names':     hp_names,
            'coords':       coords,
            'lock_metrics': lock_metrics,
            'lock_tol':     self._lock_tol.value(),
            'tube_od':      tube_od,
        }

    def set_damper_limits(self,
                          stroke_mm: float,
                          sag_shock_mm: float,
                          mr: float = 1.0):
        """Clamp the IK sweep range to physical damper travel.

        stroke_mm    : total damper stroke length (shock frame, mm)
        sag_shock_mm : static sag (shock frame, mm).  At ride height,
                       the damper is compressed by this much from full
                       extension (full droop).
        mr           : motion ratio (d_spring/d_wheel) at static.

        Sign convention in this codebase:
            positive travel = bump   (wheel up, spring compresses)
            negative travel = droop  (wheel down, spring extends)

        Available travel from ride height:
            droop (extend) room : sag / MR   (wheel moves down by this)
            bump  (compress)    : (stroke − sag) / MR
        So sweep range:  lo = −sag/MR   .. hi = +(stroke−sag)/MR
        """
        sag = max(0.0, min(stroke_mm, sag_shock_mm))
        mr_safe = max(mr, 1e-6)
        lo = -sag / mr_safe                   # max droop (wheel can drop this much)
        hi = (stroke_mm - sag) / mr_safe      # max bump  (wheel can rise this much)
        self._range_lo.setRange(lo - 5, hi + 5)
        self._range_hi.setRange(lo - 5, hi + 5)
        self._range_lo.setValue(lo)
        self._range_hi.setValue(hi)

    def _on_solve(self):
        spec = self.get_spec()
        if not spec['hp_names']:
            self._status.setText('Select at least one hardpoint to adjust.')
            return
        if not spec['coords']:
            self._status.setText('Select at least one coordinate axis (X/Y/Z).')
            return
        self._status.setText('Solving...')
        self._solve_btn.setEnabled(False)
        self.solve_requested.emit(spec)

    def show_result(self, result: dict | None, error: str = ''):
        """Called by main_window after solve completes."""
        self._solve_btn.setEnabled(True)
        self._find_btn.setVisible(False)
        if error:
            self._status.setText(f'Error: {error}')
            return
        if result is None:
            self._status.setText('No result.')
            return

        self._last_result = result
        cost = result['cost']
        mid = len(result['travel_mm']) // 2
        targets = result['targets']
        achieved = {}
        for k, curve in result['curves'].items():
            achieved[k] = curve[mid]

        lines = [f'Cost: {cost:.4f}']
        for k, tgt in targets.items():
            tgt_val = tgt[mid]
            ach_val = achieved.get(k, float('nan'))
            lines.append(f'{k}: target={tgt_val:.2f}  achieved={ach_val:.2f}')

        # ── Saturation check: did any vars hit their movement limit? ─────
        saturated = result.get('saturated', [])
        max_err = result.get('primary_max_error', 0.0)
        metric_key = result.get('primary_metric', '')

        # Determine tolerance for "target met" based on metric unit
        unit = ''
        for mkey, _, u, *_ in IK_METRICS:
            if mkey == metric_key:
                unit = u; break
        tol = 0.15 if unit == '°' else (2.0 if unit == '%' else 1.0)

        if saturated and max_err > tol:
            sat_labels = [f"  {s['label']}  ({s['delta_mm']:+.1f} / "
                          f"+-{s['bound_mm']:.0f}mm)" for s in saturated]
            lines.append('')
            lines.append(f'Target not fully met (max error: {max_err:.2f}{unit})')
            lines.append('Hardpoints at limit:')
            lines.extend(sat_labels)
            self._find_btn.setVisible(True)

        # ── Collision warning ────────────────────────────────────────────
        collisions = result.get('collisions', [])
        if collisions:
            lines.append('')
            lines.append('COLLISION DETECTED:')
            for c in collisions:
                lines.append(
                    f"  {c['member_a']} / {c['member_b']}  "
                    f"(overlap {c['overlap_mm']:.1f} mm)")

        self._status.setText('\n'.join(lines))

        # Fill results table
        n = len(result['variables'])
        self._results_table.setRowCount(n)
        for i, (v, d) in enumerate(zip(result['variables'], result['deltas_mm'])):
            self._results_table.setItem(i, 0, QTableWidgetItem(v.label))
            self._results_table.setItem(i, 1, QTableWidgetItem(f'{d:+.3f} mm'))
            new_val = result['x'][i] * 1000
            self._results_table.setItem(i, 2, QTableWidgetItem(f'{new_val:.2f} mm'))
        self._results_table.setVisible(True)
        self._apply_btn.setVisible(True)

    def _on_find_solutions(self):
        """Run multiple solves at wider bounds and let user pick."""
        spec = self.get_spec()
        spec['explore'] = True   # tells main_window to run multi-solution search
        self._find_btn.setVisible(False)
        self._solve_btn.setEnabled(False)
        self._status.setText('Searching for solutions at wider bounds...')
        self.solve_requested.emit(spec)

    def show_solutions(self, solutions: list[dict]):
        """Show a picker dialog for multiple IK solutions."""
        self._solve_btn.setEnabled(True)
        if not solutions:
            self._status.setText('No solutions found.')
            return

        dlg = _SolutionPickerDialog(solutions, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            chosen = dlg.chosen_result()
            if chosen is not None:
                self._last_result = chosen
                self.show_result(chosen)

    def _on_solve(self):
        spec = self.get_spec()
        if not spec['hp_names']:
            self._status.setText('Select at least one hardpoint to adjust.')
            return
        if not spec['coords']:
            self._status.setText('Select at least one coordinate axis (X/Y/Z).')
            return
        self._status.setText('Solving...')
        self._solve_btn.setEnabled(False)
        self.solve_requested.emit(spec)

    def _on_apply(self):
        if self._last_result is None:
            return
        axle = 'front' if self._axle.currentIndex() == 0 else 'rear'
        self.apply_requested.emit({
            'hp': self._last_result['hp'],
            'axle': axle,
        })


# ══════════════════════════════════════════════════════════════════════════════
#  DYNAMICS PANEL
# ══════════════════════════════════════════════════════════════════════════════

_DYNAMICS_HELP = """<h3 style="color:#4FC3F7;">Dynamics Panel Reference</h3>

<h4 style="color:#e07b30;">Buttons</h4>
<table cellspacing="4">
<tr><td style="color:#4FC3F7;font-weight:bold;">Solve</td>
<td>Computes the steady-state equilibrium at the specified lateral/longitudinal g.
Iterates: roll angle &rarr; per-corner travel &rarr; kinematic solve (RC migration,
camber change) &rarr; load transfer &rarr; updated roll. Converges in 2-3 iterations.</td></tr>
<tr><td style="color:#4FC3F7;font-weight:bold;">Sweep</td>
<td>Runs Solve at many g-levels (default 0&ndash;2 g, 41 points) and plots
all outputs vs. acceleration. Select <b>Lateral</b> or <b>Longitudinal</b> sweep mode.
Lateral sweep = load transfer diagram for understeer/oversteer tuning.
Longitudinal sweep = pitch, front/rear load shift under braking/accel.</td></tr>
</table>

<h4 style="color:#e07b30;">Input Parameters</h4>
<table cellspacing="4">
<tr><td style="color:#4FC3F7;">Total mass</td>
<td>Car + driver, fully loaded (kg).</td></tr>
<tr><td style="color:#4FC3F7;">Sprung mass</td>
<td>Everything supported by the springs: chassis, engine, driver, etc. Total mass minus all 4 unsprung corners.</td></tr>
<tr><td style="color:#4FC3F7;">Unsprung F/R (axle)</td>
<td>Mass of both wheels + uprights + hubs + brakes + half-links on one axle (kg). Acts at wheel-center height.</td></tr>
<tr><td style="color:#4FC3F7;">Spring rate F/R</td>
<td>Linear spring rate at the spring itself (lbf/in). Wheel rate = spring rate &times; MR&sup2;. MR is read from your geometry automatically. Converted to N/m internally (1 lbf/in = 175.127 N/m).</td></tr>
<tr><td style="color:#4FC3F7;">Tire rate</td>
<td>Vertical stiffness of the tire carcass (lbf/in). In series with the spring: ride rate = (wheel rate &times; tire rate) / (wheel rate + tire rate).</td></tr>
<tr><td style="color:#4FC3F7;">ARB rate F/R</td>
<td>Anti-roll bar equivalent wheel rate (lbf/in). Force at one wheel per inch of single-side deflection (other side fixed). Set to 0 for no ARB.</td></tr>
<tr><td style="color:#4FC3F7;">Lateral g</td>
<td>Centripetal acceleration in units of g (9.81 m/s&sup2;). 1.0 g = steady-state cornering at roughly 1.0 &times; 9.81 m/s&sup2;.</td></tr>
<tr><td style="color:#4FC3F7;">Longitudinal g</td>
<td>Braking (negative) or acceleration (positive) in g. Shifts load front/rear via pitch load transfer.</td></tr>
<tr><td style="color:#4FC3F7;">Power (wheel)</td>
<td>Peak wheel horsepower (hp). After all drivetrain losses. Used to compute torque, drive force, and power-limited max acceleration.</td></tr>
<tr><td style="color:#4FC3F7;">Engine RPM</td>
<td>Engine RPM at the operating point. With gear ratio and tire radius, this gives vehicle speed. Speed + turn radius &rarr; lateral g is auto-calculated.</td></tr>
<tr><td style="color:#4FC3F7;">Total ratio</td>
<td>Overall drivetrain ratio from engine to wheel. For single-speed FSAE: primary &times; final. E.g. if primary = 2.8 and sprocket = 3.6, total = 10.08.</td></tr>
<tr><td style="color:#4FC3F7;">Tire radius</td>
<td>Loaded tire radius (mm). For FSAE 10&rdquo; wheels: ~203 mm. Used in speed and force calculations.</td></tr>
<tr><td style="color:#4FC3F7;">Turn radius</td>
<td>Corner radius (m). With speed from RPM, this auto-calculates lateral g = v&sup2;/(R&times;9.81). FSAE hairpin ~4.5m, skidpad ~7.6m.</td></tr>
<tr><td style="color:#4FC3F7;">Max steer</td>
<td>Maximum front wheel steer angle (deg, not rack). Used to compute minimum turn radius: R_min = wheelbase / tan(steer_max).</td></tr>
<tr><td style="color:#4FC3F7;">Drivetrain</td>
<td>RWD, FWD, or AWD. Determines which tires provide traction force. RWD = rear axle only, FWD = front axle, AWD = all four.</td></tr>
</table>

<h4 style="color:#e07b30;">Auto-sourced (not entered here)</h4>
<table cellspacing="4">
<tr><td style="color:#66BB6A;">Motion ratio</td>
<td>Computed from your pushrod/rocker geometry at design position. d(spring length)/d(wheel travel).</td></tr>
<tr><td style="color:#66BB6A;">Track, wheelbase, CG</td>
<td>Read from the Car Parameters panel on the left sidebar.</td></tr>
<tr><td style="color:#66BB6A;">Roll centre height</td>
<td>Queried from the kinematic solver at each iteration's travel. Migrates with roll.</td></tr>
<tr><td style="color:#66BB6A;">Camber at load</td>
<td>Queried from the kinematic solver at the operating travel.</td></tr>
</table>

<h4 style="color:#e07b30;">Results Table</h4>
<table cellspacing="4">
<tr><td style="color:#4FC3F7;">Fz (N)</td>
<td>Vertical load on each tire. Positive = compression. Sum of all 4 = total weight.</td></tr>
<tr><td style="color:#4FC3F7;">Travel (mm)</td>
<td>Suspension travel at each corner from body roll. Positive = bump (compression).</td></tr>
<tr><td style="color:#4FC3F7;">Camber (deg)</td>
<td>Wheel camber at the operating travel. Negative = top of wheel leans inboard.</td></tr>
<tr><td style="color:#4FC3F7;">Utilization</td>
<td>Fraction of available tire grip used. &gt;1.0 means that corner has exceeded its peak lateral force &mdash; the car is sliding.</td></tr>
<tr><td style="color:#4FC3F7;">LT Geo (N)</td>
<td><b>Geometric load transfer.</b> Force path through the roll centre directly to the chassis &mdash; no body roll needed. Proportional to RC height. Higher RC = more geometric LT = less roll, but less tunability.</td></tr>
<tr><td style="color:#4FC3F7;">LT Elastic (N)</td>
<td><b>Elastic load transfer.</b> Force path through springs + ARB. Proportional to each axle's share of total roll stiffness. <b>This is what you tune with ARBs.</b> More front elastic LT = more understeer.</td></tr>
<tr><td style="color:#4FC3F7;">LT Unsprung (N)</td>
<td><b>Unsprung load transfer.</b> Direct inertia of unsprung mass (wheels, uprights, brakes) at axle height. Small but not negligible.</td></tr>
</table>

<h4 style="color:#e07b30;">Sweep Controls</h4>
<table cellspacing="4">
<tr><td style="color:#4FC3F7;">Sweep axes</td>
<td><b>Lateral</b> = sweep cornering g. <b>Longitudinal</b> = sweep braking/accel g.
<b>Both checked</b> = combined: sweeps lateral g while also applying the longitudinal g from the spinner.
This is the real peak load case &mdash; trail braking into a corner, or accelerating out.</td></tr>
<tr><td style="color:#4FC3F7;">Graphs</td>
<td>Check/uncheck which plots to show. Pitch and Understeer Gradient are new additions.</td></tr>
<tr><td style="color:#4FC3F7;">Corners</td>
<td>Toggle FL/FR/RL/RR visibility on per-corner plots (Fz, Travel, Camber, Utilization).</td></tr>
</table>

<h4 style="color:#e07b30;">New Plots</h4>
<table cellspacing="4">
<tr><td style="color:#4FC3F7;">Pitch Angle</td>
<td>Nose-down (braking) or nose-up (accel) angle from longitudinal load transfer through pitch stiffness. K_pitch = 2 &times; (K_wheel_F &times; a&sup2; + K_wheel_R &times; b&sup2;).</td></tr>
<tr><td style="color:#4FC3F7;">Understeer Gradient</td>
<td>Front avg slip angle minus rear avg slip angle (deg). Positive = understeer (front tires need more SA for the same g). Computed by inverting the tire model: given the required Fy at each corner's Fz and camber, find what SA the tire needs. Requires tire data loaded.</td></tr>
</table>

<h4 style="color:#e07b30;">Summary Line</h4>
<table cellspacing="4">
<tr><td style="color:#4FC3F7;">Roll</td>
<td>Body roll angle (deg). Roll = (sprung mass &times; ay &times; moment arm) / total roll stiffness.</td></tr>
<tr><td style="color:#4FC3F7;">LLTD</td>
<td>Lateral Load Transfer Distribution (% front). The fraction of total lateral LT carried by the front axle. Higher LLTD = front tires saturate first = understeer. Typical FSAE target: 50&ndash;58%.</td></tr>
<tr><td style="color:#4FC3F7;">RC</td>
<td>Roll centre height front/rear (mm) at the current operating point.</td></tr>
</table>

<h4 style="color:#e07b30;">Tire Data Columns</h4>
<p>Your tire data file (.mat, .csv, or .xlsx) should contain these columns.
The TTC .mat files from your zip already have all of them.</p>
<table cellspacing="4">
<tr><td style="color:#4FC3F7;font-weight:bold;">SA</td>
<td><b>Slip Angle</b> (deg). The angle between where the tire is pointing and where it is actually traveling. This generates lateral force. Positive = tire pointing inboard of travel direction.</td></tr>
<tr><td style="color:#4FC3F7;font-weight:bold;">FZ</td>
<td><b>Normal Load</b> (N). Vertical force pushing the tire into the ground. In TTC raw data, FZ is negative (compression = downward). The loader auto-detects this and flips it to positive.</td></tr>
<tr><td style="color:#4FC3F7;font-weight:bold;">FY</td>
<td><b>Lateral Force</b> (N). The cornering force the tire generates perpendicular to its heading. This is what keeps you on the track in a turn.</td></tr>
<tr><td style="color:#4FC3F7;font-weight:bold;">IA</td>
<td><b>Inclination Angle / Camber</b> (deg). Tilt of the wheel from vertical. 0 = perfectly upright. Affects the shape of the Fy vs SA curve and peak grip.</td></tr>
<tr><td style="color:#4FC3F7;font-weight:bold;">MZ</td>
<td><b>Aligning Moment</b> (Nm). Torque about the tire's vertical axis &mdash; what you feel through the steering wheel as self-aligning torque. Optional.</td></tr>
<tr><td style="color:#4FC3F7;font-weight:bold;">MX</td>
<td><b>Overturning Moment</b> (Nm). Torque about the tire's longitudinal axis from lateral force offset. Optional.</td></tr>
<tr><td style="color:#4FC3F7;font-weight:bold;">V</td>
<td><b>Velocity</b> (kph). Test speed. Used to filter out stationary data points at the start of the test run. Optional &mdash; if missing, no filter applied.</td></tr>
<tr><td style="color:#4FC3F7;font-weight:bold;">P</td>
<td><b>Pressure</b> (kPa). Tire inflation pressure during the test. Optional.</td></tr>
</table>
"""


class DynamicsPanel(CollapsibleSection):
    """
    Steady-state dynamics panel with ? help popup.
    """
    solve_requested        = pyqtSignal(dict)
    sweep_requested        = pyqtSignal(dict)
    tire_file_changed      = pyqtSignal(str)
    tire_plots_requested   = pyqtSignal()   # show tire/grip characterization plots
    params_changed         = pyqtSignal(dict)
    graph_selection_changed = pyqtSignal(list)   # selected graph keys
    corners_changed        = pyqtSignal(list)    # selected corners
    apply_aero_toggled     = pyqtSignal(bool)    # True = aero on

    def __init__(self):
        super().__init__('Dynamics', header_color='#4FC3F7')
        self.set_info(section_info.DYNAMICS)
        self._tire_path = ''
        self._build()

    def _build(self):
        # ── Help button (?) in top-right ─────────────────────────────────
        help_row = QHBoxLayout()
        help_row.addStretch()
        self._help_btn = QPushButton('?')
        self._help_btn.setFixedSize(24, 24)
        self._help_btn.setStyleSheet(
            'QPushButton { background: #1a1a1a; color: #4FC3F7; border: 1px solid #333; '
            'border-radius: 12px; font-weight: bold; font-size: 14px; }'
            'QPushButton:hover { background: #2a2a2a; border-color: #4FC3F7; }')
        self._help_btn.setToolTip('Click for detailed help on every field')
        self._help_btn.clicked.connect(self._show_help)
        help_row.addWidget(self._help_btn)
        self.add_layout(help_row)

        # ── Tire model loader ────────────────────────────────────────────
        tire_row = QHBoxLayout()
        tire_row.addWidget(QLabel('Tire data:'))
        self._tire_label = QLabel('No file loaded')
        self._tire_label.setStyleSheet(f'color: {C_SUB}; font-size: 11px;')
        self._tire_label.setWordWrap(True)
        tire_row.addWidget(self._tire_label, 1)
        self._tire_btn = QPushButton('Load')
        self._tire_btn.setMaximumWidth(60)
        self._tire_btn.clicked.connect(self._on_tire_browse)
        tire_row.addWidget(self._tire_btn)
        self.add_layout(tire_row)

        # Format hint
        fmt_label = QLabel(
            'Accepts .mat (TTC), .csv, .xlsx  \u2014  click ? for column details')
        fmt_label.setStyleSheet(f'color: {C_SUB}; font-size: 10px;')
        fmt_label.setWordWrap(True)
        self.add_widget(fmt_label)

        # ── Vehicle params grid ──────────────────────────────────────────
        g = QGridLayout(); g.setSpacing(4)
        # spinbox -> its QLabel, so apply_topology() can show/hide/relabel
        # whole rows per the active suspension topology (single-model GUI).
        self._row_labels: dict = {}

        def row(label, lo, hi, val, suf, r, dec=1, step=1.0):
            lbl = QLabel(label)
            g.addWidget(lbl, r, 0)
            sb = _spin(lo, hi, val, suf, dec=dec, step=step)
            sb.valueChanged.connect(lambda _: self.params_changed.emit(self.get_params()))
            g.addWidget(sb, r, 1)
            self._row_labels[sb] = lbl
            return sb

        r = 0
        # Mass — total is DERIVED from sprung + both unsprung values so they
        # can never silently disagree.  Total is shown as a read-only label
        # immediately below the three input rows.
        self._sprung_mass     = row('Sprung mass:',        0.1, 1e6, 223.8,  ' kg',   r, dec=1, step=1); r += 1
        self._us_front        = row('Unsprung F (axle):',  0.01, 1e6,  26.5,  ' kg',   r, dec=1, step=0.5); r += 1
        self._us_rear         = row('Unsprung R (axle):',  0.01, 1e6,  40.05, ' kg',   r, dec=2, step=0.5); r += 1
        # Derived total mass (read-only)
        g.addWidget(QLabel('Total mass (=Σ):'), r, 0)
        self._total_mass_lbl = QLabel('— kg')
        self._total_mass_lbl.setStyleSheet(
            f'color:{C_TEXT}; padding:2px 6px; background:#0a0a0a; '
            f'border:1px solid #2a2a2a; border-radius:3px; '
            "font-family:'Consolas','SF Mono',monospace;")
        g.addWidget(self._total_mass_lbl, r, 1); r += 1
        # Wire the recompute: any of the three mass inputs change → relabel
        for sb in (self._sprung_mass, self._us_front, self._us_rear):
            sb.valueChanged.connect(self._refresh_total_mass)
        self._refresh_total_mass()
        self._spring_f        = row('Spring rate F:',      0.1, 1e6, 200,   ' lbf/in', r, dec=0, step=10); r += 1
        self._spring_r        = row('Spring rate R:',      0.1, 1e6, 200,   ' lbf/in', r, dec=0, step=10); r += 1
        self._tire_rate       = row('Tire rate:',          0.1, 1e6, 909,   ' lbf/in', r, dec=0, step=25); r += 1
        # ── DECOUPLED-only damper rates ─────────────────────────────────
        # Only consumed when the topology is DECOUPLED.  Otherwise these
        # values are ignored.  Labels say "DECOUPLED" so the user knows
        # they're topology-specific and won't expect them to act under
        # STANDARD / PUSHROD setups.
        self._decoupled_heave_f = row('[DECOUPLED] Heave damper F:', 0.0, 1e6, 0.0, ' lbf/in', r, dec=0, step=10); r += 1
        self._decoupled_roll_f  = row('[DECOUPLED] Roll  damper F:', 0.0, 1e6, 0.0, ' lbf/in', r, dec=0, step=10); r += 1
        self._decoupled_heave_r = row('[DECOUPLED] Heave damper R:', 0.0, 1e6, 0.0, ' lbf/in', r, dec=0, step=10); r += 1
        self._decoupled_roll_r  = row('[DECOUPLED] Roll  damper R:', 0.0, 1e6, 0.0, ' lbf/in', r, dec=0, step=10); r += 1
        # ── HEAVE_TBAR-only 3rd-element rates ───────────────────────────
        # Consumed only when an axle is set to HEAVE_TBAR spring config.
        # The motion ratio is derived geometrically by vahan.tbar.HeaveTBarSolver
        # (perturb symmetric wheel Z, measure spring compression); the user
        # only enters the SPRING rate here.
        self._heave_3rd_f = row('[HEAVE_TBAR] 3rd-elem rate F:', 0.0, 1e6, 0.0, ' lbf/in', r, dec=0, step=10); r += 1
        self._heave_3rd_r = row('[HEAVE_TBAR] 3rd-elem rate R:', 0.0, 1e6, 0.0, ' lbf/in', r, dec=0, step=10); r += 1
        # ── ARB geometry (per axle) ─────────────────────────────────────
        # The wheel-rate contribution of each anti-roll bar is DERIVED
        # from the bar geometry rather than typed in directly:
        #     J = π·D⁴/32                  polar second moment   (mm⁴)
        #     I = π·D⁴/64                  area second moment    (mm⁴)
        #     K_torsion = G·J / (A²·L)     end-of-arm torsion    (N/mm)
        #     K_armBend = 3·E·I / A³       cantilever arm bend   (N/mm)
        #     K_arb     = (K_t·K_a)/(K_t+K_a)   series combination
        #     K_wheel   = K_arb / MR_arb²        wheel reduction
        # See workbook ‘Springs, Dampers, Anti-roll’: rows 49–64.
        #
        # Of the geometry inputs, ONLY the bar OD/ID and material
        # constants are owned by the panel.  The arm length, half-length
        # (active twist span) and MR are all redundant — they are
        # determined by the kinematic hardpoints (`arb_pivot`,
        # `arb_arm_end`, `arb_drop_top`) and the rocker chain, so
        # main_window pulls them from the kinematic model and pushes
        # them in through :meth:`set_derived_arb_geometry` before any
        # call to :meth:`get_params`.
        #
        # Hollow bars: ID > 0 switches the cross-section to a tube.
        # Default ID = 0 (solid) preserves the workbook reference rates.
        self._arb_OD_f = row('ARB F bar OD:', 0.1, 1e6,  12.70,   ' mm',    r, dec=2, step=0.1);  r += 1
        self._arb_ID_f = row('ARB F bar ID:', 0.0, 1e6,   9.65,   ' mm',    r, dec=2, step=0.1);  r += 1
        self._arb_OD_r = row('ARB R bar OD:', 0.1, 1e6,  12.70,   ' mm',    r, dec=2, step=0.1);  r += 1
        self._arb_ID_r = row('ARB R bar ID:', 0.0, 1e6,   9.65,   ' mm',    r, dec=2, step=0.1);  r += 1
        self._arb_G    = row('Bar G (shear):', 1, 1e9, 79300.0, ' N/mm²', r, dec=0, step=500);  r += 1
        self._arb_E    = row('Bar E (Young):', 1, 1e9, 207000.0,' N/mm²', r, dec=0, step=500);  r += 1
        # Blade-type ARB arms: a flat leaf whose WEAK-AXIS bending is the
        # dominant series compliance (and the trackside adjuster).  w or t
        # = 0 keeps the legacy model (arm bends with the BAR tube section).
        self._arb_blade_w_f = row('ARB F blade width:', 0.0, 1e6, 0.0, ' mm', r, dec=1, step=0.5); r += 1
        self._arb_blade_t_f = row('ARB F blade thick:', 0.0, 1e6, 0.0, ' mm', r, dec=2, step=0.1); r += 1
        self._arb_blade_w_r = row('ARB R blade width:', 0.0, 1e6, 0.0, ' mm', r, dec=1, step=0.5); r += 1
        self._arb_blade_t_r = row('ARB R blade thick:', 0.0, 1e6, 0.0, ' mm', r, dec=2, step=0.1); r += 1
        for _bw in (self._arb_blade_w_f, self._arb_blade_w_r):
            _bw.setToolTip('Blade (bar arm) flat width.\n0 = legacy model: '
                           'the arm bends with the bar tube section.')
        for _bt in (self._arb_blade_t_f, self._arb_blade_t_r):
            _bt.setToolTip('Blade (bar arm) flat thickness — the WEAK axis.\n'
                           'Arm bending stiffness = 3·E·(w·t³/12)/A³ when '
                           'both w and t > 0.\n0 = legacy tube-section arm.')
        # ID = 0 ⇒ solid bar.  Any ID ≥ OD collapses the cross-section
        # to a zero-wall tube → wheel rate = 0 (caught in get_params).
        self._arb_ID_f.setToolTip(
            'Inner diameter of front anti-roll bar.\n'
            '0 = solid bar.  Must be < OD; otherwise the bar contributes 0 stiffness.\n'
            'Hollow bars cut mass dramatically with only modest stiffness loss '
            '(stiffness scales with OD⁴−ID⁴ — the outer fibres carry almost all the load).')
        self._arb_ID_r.setToolTip(self._arb_ID_f.toolTip().replace('front', 'rear'))
        self.add_layout(g)

        # Cached derived geometry (populated by set_derived_arb_geometry).
        # Defaults match the workbook so the panel behaves sensibly on
        # first launch before the kinematic model has been queried.
        self._derived_arb_geom = {
            'F': {'arm_length_mm':  84.33, 'half_length_mm': 249.43, 'mr': 2.500},
            'R': {'arm_length_mm': 104.78, 'half_length_mm': 286.26, 'mr': 3.000},
        }
        # Read-only readout for derived ARB geometry — same styling as the
        # other auto-info labels in the panel.
        self._arb_geom_label = QLabel('')
        self._arb_geom_label.setStyleSheet(
            f'color: #66BB6A; font-size: 10px; font-style: italic;')
        self._arb_geom_label.setWordWrap(True)
        self._refresh_arb_geom_label()
        self.add_widget(self._arb_geom_label)

        # ── Powertrain ───────────────────────────────────────────────────
        pw = QGridLayout(); pw.setSpacing(4)
        pr = 0
        def prow(label, lo, hi, val, suf, r, dec=1, step=1.0):
            pw.addWidget(QLabel(label), r, 0)
            sb = _spin(lo, hi, val, suf, dec=dec, step=step)
            sb.valueChanged.connect(self._on_driving_changed)
            pw.addWidget(sb, r, 1)
            return sb
        # Powertrain type combo (ICE vs EV) — placed before the spec rows
        # because it conditionally shows/hides the peak-torque input.
        pt_row = QHBoxLayout()
        pt_row.addWidget(QLabel('Powertrain:'))
        self._powertrain_type = QComboBox()
        self._powertrain_type.addItems(['ICE', 'EV'])
        self._powertrain_type.setMaximumWidth(80)
        self._powertrain_type.setStyleSheet(
            'QComboBox { background: #1a1a1a; color: #e0e0e0; border: 1px solid #333; '
            'border-radius: 3px; padding: 2px 6px; }'
            'QComboBox::drop-down { border: none; }'
            'QComboBox QAbstractItemView { background: #1a1a1a; color: #e0e0e0; '
            'selection-background-color: #1a5276; }')
        self._powertrain_type.currentIndexChanged.connect(self._on_powertrain_type_changed)
        pt_row.addWidget(self._powertrain_type)
        pt_row.addStretch(1)
        self.add_layout(pt_row)

        # Power / RPM / torque / ratio / tire — collapsed to a single FINAL
        # drive ratio (no separate primary + sprocket).  drivetrain
        # efficiency is explicit so wheel power = engine power × eff.
        self._power_hp        = prow('Peak power:',       0,  1e6,   75.0, ' hp',  pr, dec=1, step=5); pr += 1
        self._engine_rpm      = prow('RPM at peak pwr:',  0,  1e6, 10000,  ' rpm', pr, dec=0, step=100); pr += 1
        self._peak_torque     = prow('Peak motor torque:',0,  1e6,    0.0, ' Nm',  pr, dec=1, step=1); pr += 1
        self._final_ratio     = prow('Final drive ratio:',0.01,1e6, 12.59, ':1',   pr, dec=2, step=0.1); pr += 1
        self._drivetrain_eff  = prow('Drivetrain eff:',   0.01,1.0,  0.92, '',     pr, dec=2, step=0.01); pr += 1
        self._tire_radius     = prow('Tire radius:',    0.1,1e6,   203,    ' mm', pr, dec=0, step=1); pr += 1
        self._rim_clear_dia   = prow('Rim clear dia:',  0.1,1e6,   241.3,  ' mm', pr, dec=1, step=1); pr += 1
        self._rim_clear_dia.setToolTip(
            'Clear inner diameter of the wheel rim (interference envelope).\n'
            'The kingpin ball joints + tie-rod end must fit within this circle\n'
            'about the wheel spin axis or the upright cannot be built.\n'
            'Default 241.3 mm = 9.5 in clear inside a 10 in rim.')
        self._turn_radius     = prow('Turn radius:',   0.01,1e6,    10.0,  ' m',  pr, dec=1, step=0.5); pr += 1
        self.add_layout(pw)

        # Derived display: wheel power (= engine_power × eff)
        self._wheel_power_lbl = QLabel('— W')
        self._wheel_power_lbl.setStyleSheet(
            f'color:{C_SUB}; font-size:10.5px; padding:2px 4px; '
            "font-family:'Consolas','SF Mono',monospace;")
        self.add_widget(self._wheel_power_lbl)
        for sb in (self._power_hp, self._drivetrain_eff):
            sb.valueChanged.connect(self._refresh_wheel_power)
        self._refresh_wheel_power()
        # Set the initial enabled/disabled state of the EV peak-torque input
        # based on the default powertrain_type (ICE → torque input dimmed).
        self._on_powertrain_type_changed()

        # Drive axle selector (RWD / FWD / AWD)
        dt_row = QHBoxLayout()
        dt_row.addWidget(QLabel('Drive axle:'))
        self._drivetrain = QComboBox()
        self._drivetrain.addItems(['RWD', 'FWD', 'AWD'])
        self._drivetrain.setMaximumWidth(80)
        self._drivetrain.setStyleSheet(
            'QComboBox { background: #1a1a1a; color: #e0e0e0; border: 1px solid #333; '
            'border-radius: 3px; padding: 2px 6px; }'
            'QComboBox::drop-down { border: none; }'
            'QComboBox QAbstractItemView { background: #1a1a1a; color: #e0e0e0; '
            'selection-background-color: #1a5276; }')
        self._drivetrain.currentIndexChanged.connect(
            lambda _: self.params_changed.emit(self.get_params()))
        dt_row.addWidget(self._drivetrain)
        dt_row.addStretch()
        self.add_layout(dt_row)

        # Computed driving readout
        self._driving_info = QLabel('')
        self._driving_info.setStyleSheet(f'color: #66BB6A; font-size: 10px; font-style: italic;')
        self._driving_info.setWordWrap(True)
        self.add_widget(self._driving_info)

        # Note about auto-sourced params
        auto_note = QLabel('MR from geometry. Track/WB/CG from Car Params.')
        auto_note.setStyleSheet(f'color: #66BB6A; font-size: 10px; font-style: italic;')
        self.add_widget(auto_note)

        # ── Computed dynamics constants (auto-updated) ───────────────────
        self._dyn_constants = QLabel('')
        self._dyn_constants.setStyleSheet(
            'color: #4FC3F7; font-size: 10px; font-family: monospace;'
            'background: #0a0a0a; padding: 4px; border: 1px solid #1a1a1a;'
            'border-radius: 3px;')
        self._dyn_constants.setWordWrap(True)
        self.add_widget(self._dyn_constants)

        # ── Acceleration inputs ──────────────────────────────────────────
        # Lat-g is a Cornering-only input (held value for the single-
        # point Solve; the swept value comes from the lat-g range).
        # Lon-g serves BOTH modes:
        #   Cornering : held lon-g for the combined sweep (sign drives
        #               sub-case: 0 = pure lat, − = +braking, + = +accel)
        #   Straights : applied lon-g for the trajectory (sign drives
        #               direction: + = acceleration, − = braking)
        # Each lives in its own row host so they can be hidden
        # independently — Lat-g hides in Straights, Lon-g stays.
        lat_row = QHBoxLayout()
        lat_row.setContentsMargins(0, 0, 0, 0)
        self._lat_g_label = QLabel('Lateral g:')
        lat_row.addWidget(self._lat_g_label)
        self._lat_g = _spin(-99, 99, 1.0, ' g', dec=2, step=0.05)
        lat_row.addWidget(self._lat_g)
        lat_row.addStretch()
        self._lat_row_host = QWidget()
        self._lat_row_host.setLayout(lat_row)
        self.add_widget(self._lat_row_host)

        lon_row = QHBoxLayout()
        lon_row.setContentsMargins(0, 0, 0, 0)
        self._lon_g_label = QLabel('Longitudinal g:')
        lon_row.addWidget(self._lon_g_label)
        self._lon_g = _spin(-99, 99, 0.0, ' g', dec=2, step=0.05)
        lon_row.addWidget(self._lon_g)
        lon_row.addStretch()
        self._lon_row_host = QWidget()
        self._lon_row_host.setLayout(lon_row)
        self.add_widget(self._lon_row_host)
        # Backwards-compat alias — anything that referenced
        # `_acc_grid_host` will still work via this attribute.
        self._acc_grid_host = self._lat_row_host

        # ── Buttons ──────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._solve_btn = QPushButton('Solve')
        self._solve_btn.clicked.connect(self._on_solve)
        self._solve_btn.setStyleSheet(
            'QPushButton { background: #1a5276; color: white; padding: 6px 16px; '
            'border-radius: 3px; font-weight: bold; }'
            'QPushButton:hover { background: #1f6da0; }')
        btn_row.addWidget(self._solve_btn)

        self._sweep_btn = QPushButton('Sweep')
        self._sweep_btn.clicked.connect(self._on_sweep)
        self._sweep_btn.setStyleSheet(
            'QPushButton { background: #1a5276; color: white; padding: 6px 16px; '
            'border-radius: 3px; font-weight: bold; }'
            'QPushButton:hover { background: #1f6da0; }')
        btn_row.addWidget(self._sweep_btn)

        # Tire / grip characterization plots (Fy-SA, Cα-Fz, Mz-SA, friction
        # circle) — uses the loaded tire model + the current lat/lon g operating
        # point for the friction circle.
        self._tire_btn = QPushButton('Tire / Grip Plots')
        self._tire_btn.clicked.connect(lambda: self.tire_plots_requested.emit())
        self._tire_btn.setStyleSheet(
            'QPushButton { background: #37474F; color: white; padding: 6px 16px; '
            'border-radius: 3px; font-weight: bold; }'
            'QPushButton:hover { background: #455A64; }')
        btn_row.addWidget(self._tire_btn)
        self.add_layout(btn_row)

        # ── Friction-circle options (feeds the Tire / Grip Plots popup) ────
        # User-settable vertical loads for the friction-circle panel, plus an
        # optional 3D view (Fy-Fx circles stacked over the Fz range).
        fc_row = QHBoxLayout()
        fc_row.addWidget(QLabel('Friction circle Fz (N):'))
        self._fc_fz_edit = QLineEdit('300, 650, 1000')
        self._fc_fz_edit.setToolTip(
            'Comma-separated vertical loads (N) — one friction circle is '
            'drawn per load. Leave blank for automatic levels from the '
            'tire data range.')
        self._fc_fz_edit.setMaximumWidth(140)
        fc_row.addWidget(self._fc_fz_edit)
        self._fc_3d_chk = QCheckBox('3D')
        self._fc_3d_chk.setToolTip(
            'Draw the friction circle as a 3D surface: lateral force x '
            'longitudinal force stacked over vertical load.')
        fc_row.addWidget(self._fc_3d_chk)
        fc_row.addStretch(1)
        self.add_layout(fc_row)

        # ── Apply Aero toggle ────────────────────────────────────────────
        aero_row = QHBoxLayout()
        self._apply_aero_btn = QPushButton('Apply Aero')
        self._apply_aero_btn.setCheckable(True)
        self._apply_aero_btn.setChecked(False)
        self._apply_aero_btn.setStyleSheet(
            'QPushButton { background: #1a1a1a; color: #888; padding: 5px 14px; '
            'border: 1px solid #333; border-radius: 3px; font-weight: bold; }'
            'QPushButton:checked { background: #6A1B9A; color: white; border-color: #CE93D8; }'
            'QPushButton:hover { background: #2a2a2a; }')
        self._apply_aero_btn.setToolTip(
            'Include aero downforce in dynamics solve/sweep.  Source is\n'
            'controlled by the "Aero source" combobox below — either the\n'
            'Aero Load Targets inverse-solved deficit, or user-supplied\n'
            'CFD numbers (Custom).')
        self._apply_aero_btn.toggled.connect(self._on_aero_toggle)
        aero_row.addWidget(self._apply_aero_btn)
        self._aero_label = QLabel('OFF')
        self._aero_label.setStyleSheet('color: #666; font-size: 10px;')
        aero_row.addWidget(self._aero_label)
        aero_row.addStretch()
        self.add_layout(aero_row)

        # ── Aero source toggle (Solved | Custom CFD) ─────────────────────
        # The original "Apply Aero" path only supports loading the deficit
        # produced by the Aero Load Targets inverse solver — useful for
        # sizing an aero target, but the WRONG direction for validating a
        # CFD result against measured handling.  "Custom" mode lets the
        # user type in measured/CFD downforce + CoP and drive the same
        # V²-scaled aero injection pipeline with those numbers.
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel('Aero source:'))
        self._aero_source = QComboBox()
        self._aero_source.addItem('Solved (from Aero panel)', 'solved')
        self._aero_source.addItem('Custom (CFD validation)',  'custom')
        self._aero_source.setStyleSheet(
            'QComboBox { background: #1a1a1a; color: #e0e0e0; '
            'border: 1px solid #333; border-radius: 3px; padding: 2px 6px; }'
            'QComboBox QAbstractItemView { background: #1a1a1a; color: #e0e0e0; '
            'selection-background-color: #6A1B9A; }')
        self._aero_source.currentIndexChanged.connect(self._on_aero_source_changed)
        src_row.addWidget(self._aero_source, 1)
        self.add_layout(src_row)

        # ── Custom aero inputs (visible only when "Custom" is selected) ──
        # The user enters the downforce produced at a chosen reference
        # speed plus the CoP location.  Internally we convert to a CL·A
        # so the V²-scaling pipeline downstream is unchanged:
        #     CL·A = 2·F_ref / (ρ·V_ref²)
        #     F_DF(g) at constant turn-radius R is then
        #         0.5·ρ·V²·CL·A   with   V² = g·g_earth·R
        # which still satisfies the existing "Fz per g" convention.
        self._aero_custom_box = QGroupBox('Custom aero (CFD validation)')
        self._aero_custom_box.setStyleSheet(
            'QGroupBox { color: #CE93D8; font-size: 11px; border: 1px solid #333; '
            'border-radius: 3px; margin-top: 6px; padding: 6px; }'
            'QGroupBox::title { left: 6px; padding: 0 4px; }')
        cag = QGridLayout(self._aero_custom_box); cag.setSpacing(4)
        cr = 0

        cag.addWidget(QLabel('Total DF @ ref:'), cr, 0)
        self._aero_F_ref = _spin(0.0, 1e6, 250.0, ' N', dec=1, step=10)
        self._aero_F_ref.setToolTip(
            'Total aero downforce (both axles combined) at the reference\n'
            'speed below.  Sign convention: positive = pushes the car DOWN.')
        cag.addWidget(self._aero_F_ref, cr, 1); cr += 1

        cag.addWidget(QLabel('Ref speed:'), cr, 0)
        self._aero_V_ref = _spin(0.1, 1e6, 60.0, ' km/h', dec=1, step=5)
        self._aero_V_ref.setToolTip(
            'Speed at which the downforce above was measured / simulated.\n'
            'CL·A is back-calculated as  2·F_ref / (ρ·V_ref²),  then\n'
            'downforce scales with V² at any other speed.')
        cag.addWidget(self._aero_V_ref, cr, 1); cr += 1

        cag.addWidget(QLabel('CoP (% rear):'), cr, 0)
        self._aero_cop_rear = _spin(0.0, 100.0, 50.0, ' %', dec=1, step=1)
        self._aero_cop_rear.setToolTip(
            'Centre-of-pressure expressed as the fraction of total\n'
            'downforce on the rear axle.  50% = balanced;  60% = rear-\n'
            'biased (typical for high-rake cars), 40% = front-biased.')
        cag.addWidget(self._aero_cop_rear, cr, 1); cr += 1

        cag.addWidget(QLabel('Air density:'), cr, 0)
        self._aero_rho = _spin(0.01, 10.0, 1.225, ' kg/m³', dec=3, step=0.005)
        self._aero_rho.setToolTip(
            'Air density at the operating point.  1.225 kg/m³ is ISA sea\n'
            'level / 15 °C.  Drops to ~1.18 in hot weather, ~1.10 at\n'
            'altitude — meaningful for outdoor events.')
        cag.addWidget(self._aero_rho, cr, 1); cr += 1

        # Read-only readout: equivalent CL·A + downforce at reference V
        # so the user can sanity-check against their CFD output.
        self._aero_custom_info = QLabel('')
        self._aero_custom_info.setStyleSheet(
            'color: #CE93D8; font-size: 10px; font-style: italic;')
        self._aero_custom_info.setWordWrap(True)
        cag.addWidget(self._aero_custom_info, cr, 0, 1, 2); cr += 1

        # Recompute the readout whenever a custom-aero spinbox changes,
        # and fire params_changed so dependent code (live wheel-rate
        # display, etc.) is not stuck on stale numbers.
        for sb in (self._aero_F_ref, self._aero_V_ref,
                   self._aero_cop_rear, self._aero_rho):
            sb.valueChanged.connect(self._on_custom_aero_changed)

        self.add_widget(self._aero_custom_box)
        # Hidden until Custom is selected.  We don't want it eating
        # vertical space when the panel is in Solved mode (the default).
        self._aero_custom_box.setVisible(False)
        self._on_custom_aero_changed()   # populate readout

        # ── Test mode: Cornering or Straights ──────────────────────────
        # Two top-level scenarios.  Sub-mode is INFERRED from input
        # signs — no extra dropdowns:
        #
        #   Cornering   : sweep lat-g range; sign of Lon-g spin selects
        #                 the cornering sub-case automatically:
        #                     lon = 0  →  pure lat-g
        #                     lon < 0  →  lat-g + braking
        #                     lon > 0  →  lat-g + acceleration
        #
        #   Straights   : time-domain trajectory.  Start speed selects:
        #                     start = 0       →  acceleration from rest
        #                     start > 0       →  braking from velocity
        #
        # Hidden state — kept for backwards compat with legacy save
        # files / sweep code paths but no longer surfaced in the UI.
        # The new test combo drives all visibility.
        self._sweep_lat_cb = QCheckBox()
        self._sweep_lat_cb.setChecked(True)
        self._sweep_lat_cb.setVisible(False)
        self._sweep_lon_cb = QCheckBox()
        self._sweep_lon_cb.setChecked(False)
        self._sweep_lon_cb.setVisible(False)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel('Test:'))
        self._test_mode = QComboBox()
        self._test_mode.addItem('Cornering',  'cornering')
        self._test_mode.addItem('Straights',  'straights')
        self._test_mode.setStyleSheet(
            'QComboBox { background: #1a1a1a; color: #e0e0e0; '
            'border: 1px solid #333; border-radius: 3px; padding: 2px 6px; }'
            'QComboBox QAbstractItemView { background: #1a1a1a; color: #e0e0e0; '
            'selection-background-color: #1a5276; }')
        self._test_mode.setMaximumWidth(140)
        self._test_mode.currentIndexChanged.connect(self._on_test_mode_changed)
        mode_row.addWidget(self._test_mode)
        mode_row.addStretch()
        self.add_layout(mode_row)

        # ── Sweep-by axis toggle ───────────────────────────────────────
        # User picks ONE independent variable for lateral / combined
        # sweeps; the other is derived from v² = a_y · g_e · R so the
        # operating points stay self-consistent.  No over-constraint:
        #   "by g"     : sweep lat-g, derive v = √(g·g_e·R)
        #   "by speed" : sweep v, derive lat_g = v²/(g_e·R)
        # Longitudinal trajectory mode ignores this — it sweeps time.
        # Wrap each row in a QWidget so the whole row (label + spinboxes)
        # can be hidden as a unit when not relevant.
        axis_row = QHBoxLayout()
        axis_row.setContentsMargins(0, 0, 0, 0)
        axis_row.addWidget(QLabel('Sweep by:'))
        self._sweep_axis = QComboBox()
        self._sweep_axis.addItem('Lat-g',  'g')
        self._sweep_axis.addItem('Speed',  'speed')
        self._sweep_axis.setStyleSheet(
            'QComboBox { background: #1a1a1a; color: #e0e0e0; '
            'border: 1px solid #333; border-radius: 3px; padding: 2px 6px; }'
            'QComboBox QAbstractItemView { background: #1a1a1a; color: #e0e0e0; '
            'selection-background-color: #1a5276; }')
        self._sweep_axis.setMaximumWidth(110)
        self._sweep_axis.currentIndexChanged.connect(self._on_sweep_axis_changed)
        axis_row.addWidget(self._sweep_axis)
        axis_row.addStretch()
        self._axis_host = QWidget()
        self._axis_host.setLayout(axis_row)
        self.add_widget(self._axis_host)

        sweep_row = QHBoxLayout()
        sweep_row.setContentsMargins(0, 0, 0, 0)
        sweep_row.addWidget(QLabel('Range:'))
        self._g_min = _spin(-99, 99, 0, ' g', dec=2, step=0.1)
        self._g_min.setMaximumWidth(65)
        sweep_row.addWidget(self._g_min)
        sweep_row.addWidget(QLabel('to'))
        self._g_max = _spin(-99, 99, 2.0, ' g', dec=2, step=0.1)
        self._g_max.setMaximumWidth(65)
        sweep_row.addWidget(self._g_max)
        sweep_row.addStretch()
        self._g_sweep_host = QWidget()
        self._g_sweep_host.setLayout(sweep_row)
        self.add_widget(self._g_sweep_host)

        # Speed-sweep range — visible only when the axis combo is set
        # to "Speed".  Initial values are intentionally 0..60 mph so a
        # newly-built panel doesn't accidentally sweep a meaningless
        # zero-to-zero range.
        v_sweep_row = QHBoxLayout()
        v_sweep_row.setContentsMargins(0, 0, 0, 0)
        v_sweep_row.addWidget(QLabel('Range:'))
        self._v_min = _spin(0.0, 1e4, 0.0, ' mph', dec=1, step=5.0)
        self._v_min.setMaximumWidth(85)
        v_sweep_row.addWidget(self._v_min)
        v_sweep_row.addWidget(QLabel('to'))
        self._v_max = _spin(0.0, 1e4, 60.0, ' mph', dec=1, step=5.0)
        self._v_max.setMaximumWidth(85)
        v_sweep_row.addWidget(self._v_max)
        v_sweep_row.addStretch()
        self._v_sweep_host = QWidget()
        self._v_sweep_host.setLayout(v_sweep_row)
        self._v_sweep_host.setVisible(False)
        self.add_widget(self._v_sweep_host)

        # Wire sweep-mode checkboxes to update visibility too — the
        # "Sweep by" combo + range rows only apply when at least one
        # of {Lateral, combined} is being swept.  Pure longitudinal is
        # a time-domain trajectory and ignores all three.
        self._sweep_lat_cb.stateChanged.connect(self._on_sweep_mode_changed)
        self._sweep_lon_cb.stateChanged.connect(self._on_sweep_mode_changed)

        # ── Acceleration trajectory inputs ─────────────────────────────
        # Longitudinal sweep is a time-domain trajectory:
        #   F_engine = P/v   (power-limited, capped by traction at low v)
        #   F_drag   = ½·ρ·CdA·v²
        #   a(t)     = (F_engine − F_drag) / m
        # The integrator runs from start_speed_mph until achievable
        # accel decays toward terminal (drag balances power) — no
        # hardcoded duration, the trajectory ends when physics says.
        # Lateral sweeps ignore all three of these inputs.
        sv_row = QHBoxLayout()
        sv_row.addWidget(QLabel('Start speed:'))
        self._start_speed = _spin(0.0, 1e4, 0.0, ' mph', dec=1, step=5.0)
        self._start_speed.setMaximumWidth(85)
        self._start_speed.setToolTip(
            'Initial velocity at t = 0 of the longitudinal trajectory.\n'
            'The car accelerates / decelerates from this speed.\n'
            'Lateral sweeps ignore this (speed comes from a_y·R).')
        sv_row.addWidget(self._start_speed)
        sv_row.addWidget(QLabel('End speed:'))
        self._end_speed = _spin(0.0, 1e4, 0.0, ' mph', dec=1, step=5.0)
        self._end_speed.setMaximumWidth(85)
        self._end_speed.setToolTip(
            'Optional ceiling for the longitudinal trajectory.\n'
            'When > 0, the run terminates at this speed (or earlier if\n'
            'physics says).  When 0, the trajectory runs to its natural\n'
            'end (terminal speed for accel, v = 0 for braking).')
        sv_row.addWidget(self._end_speed)
        sv_row.addStretch()
        self.add_layout(sv_row)

        # ── Trajectory direction (longitudinal mode) ───────────────────
        # Picks whether the time-domain run is an acceleration (engine
        # propelling against drag) or a braking event (max-grip
        # deceleration of all 4 tires until v = 0).  Lateral sweeps
        # ignore this — they're steady-state operating-point sweeps.
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel('Direction:'))
        self._traj_direction = QComboBox()
        self._traj_direction.addItem('Accelerate', 'accel')
        self._traj_direction.addItem('Brake',      'brake')
        self._traj_direction.setStyleSheet(
            'QComboBox { background: #1a1a1a; color: #e0e0e0; '
            'border: 1px solid #333; border-radius: 3px; padding: 2px 6px; }'
            'QComboBox QAbstractItemView { background: #1a1a1a; color: #e0e0e0; '
            'selection-background-color: #1a5276; }')
        self._traj_direction.setMaximumWidth(110)
        self._traj_direction.setToolTip(
            'Longitudinal trajectory direction:\n'
            '  • Accelerate — v grows from start_speed, engine vs drag,\n'
            '    terminates at terminal speed (P = drag·v).\n'
            '  • Brake — v drops from start_speed at full 4-tire μ·g until\n'
            '    v = 0.  Use start_speed > 0 for a meaningful run.')
        dir_row.addWidget(self._traj_direction)
        dir_row.addStretch()
        self.add_layout(dir_row)

        aero_row = QHBoxLayout()
        aero_row.addWidget(QLabel('CdA:'))
        self._cda = _spin(0.001, 1e3, 1.0, ' m²', dec=3, step=0.05)
        self._cda.setMaximumWidth(95)
        self._cda.setToolTip(
            'Drag coefficient × frontal area (m²).\n'
            '  Sedan ≈ 0.6 · 2.2 = 1.3   |   FSAE no aero ≈ 1.0\n'
            '  FSAE with aero ≈ 1.5–2.0  |   F1 ≈ 1.3–1.5\n'
            'Bounds the longitudinal-trajectory terminal speed via\n'
            '   v_term = ∛(2·P / (ρ·CdA))')
        aero_row.addWidget(self._cda)
        aero_row.addWidget(QLabel('Air density:'))
        self._air_rho = _spin(0.01, 100.0, 1.225, ' kg/m³', dec=3, step=0.005)
        self._air_rho.setMaximumWidth(95)
        self._air_rho.setToolTip(
            'Air density ρ (kg/m³).  ISA sea level / 15 °C = 1.225.\n'
            'Drops to ~1.18 in hot weather, ~1.10 at altitude.')
        aero_row.addWidget(self._air_rho)
        aero_row.addStretch()
        self.add_layout(aero_row)

        # ── Graph picker ─────────────────────────────────────────────────
        self._graph_checks = {}
        _DYN_GRAPHS = [
            ('fz',             'Corner Loads (Fz)'),
            ('roll',           'Roll Angle'),
            ('pitch',          'Pitch Angle'),
            ('travel',         'Suspension Travel'),
            ('camber',         'Camber'),
            ('lt',             'Load Transfer'),
            ('rc',             'Roll Centre Height'),
            ('utilization',    'Tire Utilization'),
            ('understeer',     'Understeer Gradient'),
            ('steer_correction', 'Steer Correction'),
            ('steering_wheel_angle', 'Steering Wheel Angle'),
            ('path_deviation',   'Path Deviation'),
            ('speed',          'Speed'),
        ]
        _DEFAULT_ON = {'fz', 'roll', 'travel', 'lt', 'utilization'}

        gpick_row = QHBoxLayout()
        gpick_row.addWidget(QLabel('Graphs:'))
        gpick_row.addStretch()
        self.add_layout(gpick_row)

        gpick_grid = QGridLayout(); gpick_grid.setSpacing(2)
        for idx, (key, label) in enumerate(_DYN_GRAPHS):
            cb = QCheckBox(label)
            cb.setChecked(key in _DEFAULT_ON)
            cb.setStyleSheet('QCheckBox { font-size: 11px; }')
            cb.stateChanged.connect(self._on_graph_changed)
            self._graph_checks[key] = cb
            gpick_grid.addWidget(cb, idx // 2, idx % 2)
        self.add_layout(gpick_grid)

        # ── Corner selector ──────────────────────────────────────────────
        corner_row = QHBoxLayout()
        corner_row.addWidget(QLabel('Corners:'))
        self._corner_cbs = {}
        for lbl in ('FL', 'FR', 'RL', 'RR'):
            cb = QCheckBox(lbl)
            cb.setChecked(True)
            cb.setStyleSheet('QCheckBox { font-size: 11px; }')
            cb.stateChanged.connect(self._on_corners_changed)
            self._corner_cbs[lbl] = cb
            corner_row.addWidget(cb)
        corner_row.addStretch()
        self.add_layout(corner_row)

        # ── Results table ────────────────────────────────────────────────
        self._result_table = QTableWidget(7, 4)
        self._result_table.setHorizontalHeaderLabels(['FL', 'FR', 'RL', 'RR'])
        self._result_table.setVerticalHeaderLabels([
            'Fz (N)', 'Travel (mm)', 'Camber (deg)',
            'Utilization', 'LT Geo (N)', 'LT Elastic (N)', 'LT Unsprung (N)',
        ])
        self._result_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._result_table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Fixed)
        self._result_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._result_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._result_table.setMaximumHeight(210)
        self.add_widget(self._result_table)

        # ── Summary line ─────────────────────────────────────────────────
        self._summary = QLabel('')
        self._summary.setStyleSheet(f'color: {C_TEXT}; font-size: 11px;')
        self._summary.setWordWrap(True)
        self.add_widget(self._summary)

        self._status = QLabel('')
        self._status.setStyleSheet(f'color: {C_SUB}; font-size: 11px;')
        self.add_widget(self._status)

        # Set initial input visibility based on the default test mode.
        try:
            self._on_test_mode_changed()
        except Exception:
            pass

    # ── Public API ───────────────────────────────────────────────────────

    @staticmethod
    def _compute_arb_wheel_rate_Npm(OD_mm: float, ID_mm: float,
                                    L_half_mm: float, A_mm: float,
                                    MR: float, G_Npmm2: float,
                                    E_Npmm2: float,
                                    blade_w_mm: float = 0.0,
                                    blade_t_mm: float = 0.0) -> float:
        """Wheel-rate contribution of one anti-roll bar (N/m).

        Hollow-shaft form — the area moments scale with (OD⁴−ID⁴) so a
        solid bar is the special case ID = 0.  Combines the torsional
        stiffness of the half-bar and the cantilever bending stiffness
        of the arm in series, then reduces through the bar→wheel motion
        ratio.

            J = π·(OD⁴ − ID⁴) / 32          polar 2nd moment (hollow)
            I = π·(OD⁴ − ID⁴) / 64          area  2nd moment (hollow)
            K_t = G·J / (A²·L_half)         torsion bar (force at arm tip)
            K_a = 3·E·I / A³                arm cantilever bending
            K_arb = (K_t·K_a) / (K_t + K_a) two springs in series
            K_w   = K_arb / MR_arb²         wheel reflection

        Returns 0 if OD/L/A/MR are non-positive (so the panel never
        crashes mid-typing) or if ID ≥ OD (degenerate "tube wall = 0").
        ID < 0 is silently clamped to 0.
        """
        import math
        OD = float(OD_mm); ID = max(0.0, float(ID_mm))
        L  = float(L_half_mm); A = float(A_mm); MR = float(MR)
        G  = float(G_Npmm2); E = float(E_Npmm2)
        if min(OD, L, A, MR) <= 0.0:
            return 0.0
        if ID >= OD:                       # zero or negative wall thickness
            return 0.0
        D4 = OD ** 4 - ID ** 4             # mm⁴ (positive by check above)
        J = math.pi * D4 / 32.0            # mm⁴
        I = math.pi * D4 / 64.0            # mm⁴
        # Blade-type arm: a flat leaf bending about its WEAK axis is the
        # arm's series compliance (and the trackside stiffness adjuster).
        # w or t = 0 → legacy model (arm bends with the bar tube section).
        bw = max(0.0, float(blade_w_mm)); bt = max(0.0, float(blade_t_mm))
        if bw > 0.0 and bt > 0.0:
            I_arm = bw * bt ** 3 / 12.0    # mm⁴ (weak axis)
        else:
            I_arm = I
        K_t = G * J / (A * A * L)          # N/mm at arm tip (torsion bar)
        K_a = 3.0 * E * I_arm / (A * A * A)  # N/mm at arm tip (arm bending)
        if K_t > 0.0 and K_a > 0.0:
            K_arb = (K_t * K_a) / (K_t + K_a)
        else:
            K_arb = max(K_t, K_a)
        K_w_Npmm = K_arb / (MR * MR)       # N/mm at wheel
        return K_w_Npmm * 1000.0           # → N/m

    # ── Topology gating (single-model GUI) ──────────────────────────────
    def apply_topology(self, topo) -> None:
        """Show/hide/relabel the spring + bar inputs so the panel presents
        EXACTLY the elements the active topology physically has — not the
        same generic fields for every car (single-model philosophy).

        Per axle:
          * CORNER (std)   -> 'Spring rate' corner coil; ARB rows per ARBType.
          * central T-bar  -> corner coil ON the bellcrank ('Corner coil
            (plain TBAR)      rate'); bar rows relabelled 'T-bar OD/ID'
                              (the twist IS the roll device).
          * HEAVE_TBAR     -> same as central T-bar + the 3rd (heave) spring
                              rate row.
          * DECOUPLED      -> NO corner coil and NO bar: heave + roll spring
                              rates of the two cross-car coilovers instead.
        Also seeds a sensible non-zero default rate the first time a
        topology-specific field becomes active (so a fresh decoupled /
        heave-T-bar project is genuinely that flavour from the get-go)."""
        from vahan.topology import (DamperActuation as _DA, ARBType as _AT,
                                    SpringConfig as _SC)

        def _flavor(a):
            if a.spring_config == _SC.DECOUPLED:
                return 'decoupled'
            if a.spring_config == _SC.HEAVE_TBAR:
                return 'heave_tbar'
            if (a.arb_type == _AT.TBAR
                    and a.damper_actuation in (_DA.PUSHROD, _DA.PULLROD)):
                return 'tbar'
            return 'standard'

        def _row(sb, visible, label=None):
            lbl = self._row_labels.get(sb)
            sb.setVisible(visible)
            if lbl is not None:
                lbl.setVisible(visible)
                if label is not None:
                    lbl.setText(label)

        def _seed(sb, value):
            if sb.value() <= 0.0:
                sb.setValue(value)

        any_bar = False
        for a, tag, sp, dh, dr, h3, od, idd in (
            (topo.front, 'F', self._spring_f, self._decoupled_heave_f,
             self._decoupled_roll_f, self._heave_3rd_f,
             self._arb_OD_f, self._arb_ID_f),
            (topo.rear, 'R', self._spring_r, self._decoupled_heave_r,
             self._decoupled_roll_r, self._heave_3rd_r,
             self._arb_OD_r, self._arb_ID_r),
        ):
            fl = _flavor(a)
            # corner spring / coil
            if fl == 'decoupled':
                _row(sp, False)
            elif fl in ('heave_tbar', 'tbar'):
                _row(sp, True, f'Corner coil rate {tag} (bellcrank):')
            else:
                _row(sp, True, f'Spring rate {tag}:')
            # decoupled cross-car coilovers (these are SPRING rates)
            _row(dh, fl == 'decoupled', f'Heave spring rate {tag}:')
            _row(dr, fl == 'decoupled', f'Roll spring rate {tag}:')
            if fl == 'decoupled':
                _seed(dh, 200.0); _seed(dr, 200.0)
            # 3rd (heave) element
            _row(h3, fl == 'heave_tbar', f'3rd (heave) spring rate {tag}:')
            if fl == 'heave_tbar':
                _seed(h3, 200.0)
            # bar rows: the T-bar twist for tbar/heave_tbar; the U-bar for
            # bellcrank/control-arm; none for decoupled or ARB=NONE.
            if fl in ('heave_tbar', 'tbar') or (
                    fl == 'standard' and a.arb_type == _AT.TBAR):
                _row(od, True, f'T-bar {tag} OD:'); _row(idd, True, f'T-bar {tag} ID:')
                any_bar = True
            elif fl == 'standard' and a.arb_type in (_AT.BELLCRANK,
                                                     _AT.CONTROL_ARM):
                _row(od, True, f'ARB {tag} bar OD:'); _row(idd, True, f'ARB {tag} bar ID:')
                any_bar = True
            else:
                _row(od, False); _row(idd, False)
        _row(self._arb_G, any_bar)
        _row(self._arb_E, any_bar)

    def set_derived_arb_geometry(self, front: dict, rear: dict) -> None:
        """Push kinematic-derived ARB arm length / half-length / MR.

        Called by main_window before any :meth:`get_params` so the wheel-rate
        formula always uses the latest geometry from the kinematic model
        rather than stale numbers.  Each dict must carry the keys
        ``'arm_length_mm'``, ``'half_length_mm'``, ``'mr'``.
        """
        for key in ('arm_length_mm', 'half_length_mm', 'mr'):
            if key not in front or key not in rear:
                return                              # ignore malformed update
        self._derived_arb_geom['F'] = {k: float(front[k]) for k in
                                       ('arm_length_mm', 'half_length_mm', 'mr')}
        self._derived_arb_geom['R'] = {k: float(rear[k]) for k in
                                       ('arm_length_mm', 'half_length_mm', 'mr')}
        self._refresh_arb_geom_label()

    def _refresh_arb_geom_label(self) -> None:
        """Refresh the read-only ARB geometry readout from the cache."""
        if not hasattr(self, '_arb_geom_label'):
            return
        f = self._derived_arb_geom['F']
        r = self._derived_arb_geom['R']
        self._arb_geom_label.setText(
            f"Auto ARB:  F  half {f['half_length_mm']:.1f}  arm {f['arm_length_mm']:.1f} mm   "
            f"MR {f['mr']:.3f}\n"
            f"           R  half {r['half_length_mm']:.1f}  arm {r['arm_length_mm']:.1f} mm   "
            f"MR {r['mr']:.3f}"
        )

    def rim_clear_diameter_m(self) -> float:
        """Wheel-rim clear inner diameter (m) — the interference envelope the
        kingpin ball joints + tie-rod end must fit inside.  A packaging input,
        kept OUT of get_params()/VehicleParams (that is dynamics only)."""
        return float(self._rim_clear_dia.value()) / 1000.0

    def get_params(self) -> dict:
        """Return dynamics parameters dict for VehicleParams construction.

        ``arb_rate_*_Npm`` are *derived* from the bar geometry inputs
        via :meth:`_compute_arb_wheel_rate_Npm`; nothing in the panel
        types in a wheel-rate ARB number directly any more.  Of the
        geometry inputs, the panel only owns D / G / E — the arm length,
        half-length and MR are pulled from the kinematic model and
        pushed in via :meth:`set_derived_arb_geometry`.
        """
        G  = self._arb_G.value()
        E  = self._arb_E.value()
        f_geom = self._derived_arb_geom['F']
        r_geom = self._derived_arb_geom['R']
        arb_f_Npm = self._compute_arb_wheel_rate_Npm(
            OD_mm=self._arb_OD_f.value(),
            ID_mm=self._arb_ID_f.value(),
            L_half_mm=f_geom['half_length_mm'],
            A_mm=f_geom['arm_length_mm'],
            MR=f_geom['mr'],
            G_Npmm2=G, E_Npmm2=E,
            blade_w_mm=self._arb_blade_w_f.value(),
            blade_t_mm=self._arb_blade_t_f.value())
        arb_r_Npm = self._compute_arb_wheel_rate_Npm(
            OD_mm=self._arb_OD_r.value(),
            ID_mm=self._arb_ID_r.value(),
            L_half_mm=r_geom['half_length_mm'],
            A_mm=r_geom['arm_length_mm'],
            MR=r_geom['mr'],
            G_Npmm2=G, E_Npmm2=E,
            blade_w_mm=self._arb_blade_w_r.value(),
            blade_t_mm=self._arb_blade_t_r.value())
        params = {
            # total_mass_kg is now DERIVED from the three below — no longer
            # an independent input.  VehicleParams computes it as a @property.
            'sprung_mass_kg':         self._sprung_mass.value(),
            'unsprung_mass_front_kg': self._us_front.value(),
            'unsprung_mass_rear_kg':  self._us_rear.value(),
            'spring_rate_front_Npm':  self._spring_f.value() * 175.127,  # lbf/in → N/m
            'spring_rate_rear_Npm':   self._spring_r.value() * 175.127,
            'tire_rate_Npm':          self._tire_rate.value() * 175.127,
            'arb_rate_front_Npm':     arb_f_Npm,
            'arb_rate_rear_Npm':      arb_r_Npm,
            # DECOUPLED-only fields -- ignored unless topology says so.
            # 175.127 converts lbf/in -> N/m.  Stored under the names
            # VehicleParams reads; main_window's _apply_topology_to_dyn_params
            # will use these directly instead of the spring_rate / arb_rate
            # reuse trick the previous build relied on.
            'decoupled_heave_rate_front_Npm': self._decoupled_heave_f.value() * 175.127,
            'decoupled_roll_rate_front_Npm':  self._decoupled_roll_f.value()  * 175.127,
            'decoupled_heave_rate_rear_Npm':  self._decoupled_heave_r.value() * 175.127,
            'decoupled_roll_rate_rear_Npm':   self._decoupled_roll_r.value()  * 175.127,
            # HEAVE_TBAR-only 3rd-element rates -- ignored unless topology says so
            'heave_3rd_rate_front_Npm':       self._heave_3rd_f.value() * 175.127,
            'heave_3rd_rate_rear_Npm':        self._heave_3rd_r.value() * 175.127,
            'power_hp':               self._power_hp.value(),
            'engine_rpm':             self._engine_rpm.value(),
            'total_drive_ratio':      self._final_ratio.value(),   # single final ratio
            'tire_radius_m':          self._tire_radius.value() / 1000,
            'drivetrain':             self._drivetrain.currentText(),
            'drivetrain_efficiency':  self._drivetrain_eff.value(),
            'powertrain_type':        self._powertrain_type.currentText(),
            'peak_torque_Nm':         self._peak_torque.value(),
        }
        # Aero drag — bounds terminal speed in time-domain trajectory.
        # Guarded with hasattr because get_params can fire from another
        # widget's signal during _build() before these spinboxes exist
        # (custom-aero readout calls params_changed which calls
        # get_params).  When missing, omit the key — VehicleParams uses
        # its own defaults for both fields, no magic numbers here.
        if hasattr(self, '_cda'):
            params['cda_m2'] = self._cda.value()
        if hasattr(self, '_air_rho'):
            params['air_density_kg_m3'] = self._air_rho.value()
        return params

    # ── Save / load (every input on the panel) ───────────────────────────

    def get_state(self) -> dict:
        """JSON-friendly snapshot of every user-facing input on the panel.

        Used by the project save/load flow.  Excludes values that are
        *derived* — the ARB arm length / half-length / MR readout is
        rebuilt from the kinematic model on load, the dyn-constants /
        driving-info labels are recomputed.  Includes the tire-data file
        path, the graph-picker selection and the corner selection so
        re-opening a project restores the user's full UI state, not just
        the numerical inputs.
        """
        return {
            # Vehicle masses + springs.  total_mass is DERIVED (= sum) and
            # written to the saved state for backward-compat with older
            # readers, but on load it is IGNORED — the saved sprung +
            # unsprung values are the source of truth.
            'total_mass_kg':       float(self._sprung_mass.value()
                                         + self._us_front.value()
                                         + self._us_rear.value()),
            'sprung_mass_kg':      float(self._sprung_mass.value()),
            'unsprung_front_kg':   float(self._us_front.value()),
            'unsprung_rear_kg':    float(self._us_rear.value()),
            'spring_front_lbfin':  float(self._spring_f.value()),
            'spring_rear_lbfin':   float(self._spring_r.value()),
            'tire_rate_lbfin':     float(self._tire_rate.value()),
            'decoupled_heave_f_lbfin': float(self._decoupled_heave_f.value()),
            'decoupled_roll_f_lbfin':  float(self._decoupled_roll_f.value()),
            'decoupled_heave_r_lbfin': float(self._decoupled_heave_r.value()),
            'decoupled_roll_r_lbfin':  float(self._decoupled_roll_r.value()),
            'heave_3rd_f_lbfin':       float(self._heave_3rd_f.value()),
            'heave_3rd_r_lbfin':       float(self._heave_3rd_r.value()),
            # ARB material / diameters (arm-length / half-length / MR are
            # *derived* from the kinematic model on load).  ID = 0 is a
            # solid bar — the saved key set is the same regardless of
            # cross-section so older v2 hollow-less files still load.
            'arb_OD_f_mm':         float(self._arb_OD_f.value()),
            'arb_ID_f_mm':         float(self._arb_ID_f.value()),
            'arb_OD_r_mm':         float(self._arb_OD_r.value()),
            'arb_ID_r_mm':         float(self._arb_ID_r.value()),
            'arb_blade_w_f_mm':    float(self._arb_blade_w_f.value()),
            'arb_blade_t_f_mm':    float(self._arb_blade_t_f.value()),
            'arb_blade_w_r_mm':    float(self._arb_blade_w_r.value()),
            'arb_blade_t_r_mm':    float(self._arb_blade_t_r.value()),
            'arb_G_Npmm2':         float(self._arb_G.value()),
            'arb_E_Npmm2':         float(self._arb_E.value()),
            # Powertrain — primary_ratio + sprocket_* are obsolete (collapsed
            # into final_drive_ratio).  Kept in save dict for backward read
            # by older readers but ignored on load.
            'power_hp':              float(self._power_hp.value()),
            'engine_rpm':            float(self._engine_rpm.value()),
            'final_drive_ratio':     float(self._final_ratio.value()),
            'drivetrain_efficiency': float(self._drivetrain_eff.value()),
            'powertrain_type':       str(self._powertrain_type.currentText()),
            'peak_torque_Nm':        float(self._peak_torque.value()),
            'tire_radius_mm':        float(self._tire_radius.value()),
            'rim_clear_dia_mm':      float(self._rim_clear_dia.value()),
            'turn_radius_m':         float(self._turn_radius.value()),
            'drivetrain':          self._drivetrain.currentText(),
            # Solve / sweep setpoints
            'lateral_g':           float(self._lat_g.value()),
            'longitudinal_g':      float(self._lon_g.value()),
            'sweep_lat':           bool(self._sweep_lat_cb.isChecked()),
            'sweep_lon':           bool(self._sweep_lon_cb.isChecked()),
            'g_min':               float(self._g_min.value()),
            'g_max':               float(self._g_max.value()),
            'v_min_mph':           float(self._v_min.value()),
            'v_max_mph':           float(self._v_max.value()),
            'sweep_axis':          self._sweep_axis.currentData() or 'g',
            'test_mode':           self._test_mode.currentData() or 'cornering',
            'start_speed_mph':     float(self._start_speed.value()),
            'end_speed_mph':       float(self._end_speed.value()),
            'cda_m2':              float(self._cda.value()),
            'air_density_kg_m3':   float(self._air_rho.value()),
            'traj_direction':      self._traj_direction.currentData() or 'accel',
            # UI selections
            'apply_aero':          bool(self._apply_aero_btn.isChecked()),
            'aero_source':         self.get_aero_source(),
            'aero_F_ref_N':        float(self._aero_F_ref.value()),
            'aero_V_ref_kph':      float(self._aero_V_ref.value()),
            'aero_cop_rear_pct':   float(self._aero_cop_rear.value()),
            'aero_air_density':    float(self._aero_rho.value()),
            'graphs':              [k for k, cb in self._graph_checks.items()
                                    if cb.isChecked()],
            'corners':             [lbl for lbl, cb in self._corner_cbs.items()
                                    if cb.isChecked()],
            # Tire-data file (absolute path; falls back gracefully if
            # the file moved between save & load)
            'tire_path':           str(self._tire_path),
        }

    def set_state(self, d: dict) -> None:
        """Restore every input from a dict produced by :meth:`get_state`.

        Missing keys are silently left at their current (default) value
        so an older save file that lacks newer fields still loads.  All
        widget signals are blocked during the bulk update so we don't
        emit ``params_changed`` 25 times — a single
        :meth:`_on_driving_changed` call at the end refreshes everything
        downstream.
        """
        if not isinstance(d, dict):
            return

        def _set_spin(sb, key):
            if key in d:
                try:
                    sb.setValue(float(d[key]))
                except (TypeError, ValueError):
                    pass

        # Block signals on every widget that emits during set so the
        # bulk load is atomic from the controller's perspective.
        widgets = [
            self._sprung_mass, self._us_front, self._us_rear,
            self._spring_f, self._spring_r, self._tire_rate,
            self._decoupled_heave_f, self._decoupled_roll_f,
            self._decoupled_heave_r, self._decoupled_roll_r,
            self._heave_3rd_f, self._heave_3rd_r,
            self._arb_OD_f, self._arb_ID_f, self._arb_OD_r, self._arb_ID_r,
            self._arb_G, self._arb_E,
            self._power_hp, self._engine_rpm,
            self._final_ratio, self._drivetrain_eff, self._peak_torque,
            self._tire_radius, self._rim_clear_dia, self._turn_radius,
            self._lat_g, self._lon_g, self._g_min, self._g_max,
            self._drivetrain, self._sweep_lat_cb, self._sweep_lon_cb,
            self._apply_aero_btn,
            self._aero_source, self._aero_F_ref, self._aero_V_ref,
            self._aero_cop_rear, self._aero_rho,
        ]
        for w in widgets:
            w.blockSignals(True)
        try:
            # total_mass_kg is IGNORED on load (it's a derived sum now).
            # Sprung + unsprung values below are the source of truth.
            _set_spin(self._sprung_mass,     'sprung_mass_kg')
            _set_spin(self._us_front,        'unsprung_front_kg')
            _set_spin(self._us_rear,         'unsprung_rear_kg')
            _set_spin(self._spring_f,        'spring_front_lbfin')
            _set_spin(self._spring_r,        'spring_rear_lbfin')
            _set_spin(self._tire_rate,       'tire_rate_lbfin')
            _set_spin(self._decoupled_heave_f, 'decoupled_heave_f_lbfin')
            _set_spin(self._decoupled_roll_f,  'decoupled_roll_f_lbfin')
            _set_spin(self._decoupled_heave_r, 'decoupled_heave_r_lbfin')
            _set_spin(self._decoupled_roll_r,  'decoupled_roll_r_lbfin')
            _set_spin(self._heave_3rd_f,       'heave_3rd_f_lbfin')
            _set_spin(self._heave_3rd_r,       'heave_3rd_r_lbfin')
            # ARB OD/ID — back-compat: old v2 saves used a single
            # diameter per axle (arb_D_f_mm / arb_D_r_mm) which we
            # treat as OD with ID = 0 (solid bar).
            if 'arb_OD_f_mm' in d:
                _set_spin(self._arb_OD_f,    'arb_OD_f_mm')
            elif 'arb_D_f_mm' in d:
                _set_spin(self._arb_OD_f,    'arb_D_f_mm')
                self._arb_ID_f.setValue(0.0)
            if 'arb_OD_r_mm' in d:
                _set_spin(self._arb_OD_r,    'arb_OD_r_mm')
            elif 'arb_D_r_mm' in d:
                _set_spin(self._arb_OD_r,    'arb_D_r_mm')
                self._arb_ID_r.setValue(0.0)
            _set_spin(self._arb_ID_f,        'arb_ID_f_mm')
            _set_spin(self._arb_ID_r,        'arb_ID_r_mm')
            _set_spin(self._arb_blade_w_f,   'arb_blade_w_f_mm')
            _set_spin(self._arb_blade_t_f,   'arb_blade_t_f_mm')
            _set_spin(self._arb_blade_w_r,   'arb_blade_w_r_mm')
            _set_spin(self._arb_blade_t_r,   'arb_blade_t_r_mm')
            _set_spin(self._arb_G,           'arb_G_Npmm2')
            _set_spin(self._arb_E,           'arb_E_Npmm2')
            _set_spin(self._power_hp,        'power_hp')
            _set_spin(self._engine_rpm,      'engine_rpm')
            # Final drive ratio replaces primary + sprocket_drive +
            # sprocket_driven.  Older files store those three; if so, fold
            # them into the new single ratio.  New files have
            # 'final_drive_ratio' directly.
            if 'final_drive_ratio' in d:
                _set_spin(self._final_ratio, 'final_drive_ratio')
            elif ('primary_ratio' in d and 'sprocket_drive' in d
                    and 'sprocket_driven' in d):
                try:
                    pr  = float(d['primary_ratio'])
                    sd  = max(float(d['sprocket_drive']), 1)
                    sdn = float(d['sprocket_driven'])
                    self._final_ratio.setValue(pr * sdn / sd)
                except (TypeError, ValueError):
                    pass
            _set_spin(self._drivetrain_eff,  'drivetrain_efficiency')
            _set_spin(self._peak_torque,     'peak_torque_Nm')
            if 'powertrain_type' in d:
                idx = self._powertrain_type.findText(str(d['powertrain_type']))
                if idx >= 0:
                    self._powertrain_type.setCurrentIndex(idx)
            _set_spin(self._tire_radius,     'tire_radius_mm')
            _set_spin(self._rim_clear_dia,   'rim_clear_dia_mm')
            _set_spin(self._turn_radius,     'turn_radius_m')
            _set_spin(self._lat_g,           'lateral_g')
            _set_spin(self._lon_g,           'longitudinal_g')
            _set_spin(self._g_min,           'g_min')
            _set_spin(self._g_max,           'g_max')
            _set_spin(self._v_min,           'v_min_mph')
            _set_spin(self._v_max,           'v_max_mph')
            _set_spin(self._start_speed,     'start_speed_mph')
            _set_spin(self._cda,             'cda_m2')
            _set_spin(self._air_rho,         'air_density_kg_m3')

            # Sweep-axis combo
            if 'sweep_axis' in d:
                target = str(d['sweep_axis'])
                for i in range(self._sweep_axis.count()):
                    if self._sweep_axis.itemData(i) == target:
                        self._sweep_axis.setCurrentIndex(i)
                        break
            # Trajectory direction combo (legacy hidden)
            if 'traj_direction' in d:
                target = str(d['traj_direction'])
                for i in range(self._traj_direction.count()):
                    if self._traj_direction.itemData(i) == target:
                        self._traj_direction.setCurrentIndex(i)
                        break
            # Test mode combo
            if 'test_mode' in d:
                target = str(d['test_mode'])
                for i in range(self._test_mode.count()):
                    if self._test_mode.itemData(i) == target:
                        self._test_mode.setCurrentIndex(i)
                        break

            if 'drivetrain' in d:
                idx = self._drivetrain.findText(str(d['drivetrain']))
                if idx >= 0:
                    self._drivetrain.setCurrentIndex(idx)

            if 'sweep_lat' in d:
                self._sweep_lat_cb.setChecked(bool(d['sweep_lat']))
            if 'sweep_lon' in d:
                self._sweep_lon_cb.setChecked(bool(d['sweep_lon']))
            if 'apply_aero' in d:
                self._apply_aero_btn.setChecked(bool(d['apply_aero']))

            # Aero source + custom inputs.  Tolerate missing keys for
            # forward compat: an older save without aero_source loads
            # in 'solved' mode (current behaviour) and the custom-aero
            # spinboxes keep their defaults.
            if 'aero_source' in d:
                target = str(d['aero_source'])
                for i in range(self._aero_source.count()):
                    if self._aero_source.itemData(i) == target:
                        self._aero_source.setCurrentIndex(i)
                        break
            _set_spin(self._aero_F_ref,     'aero_F_ref_N')
            _set_spin(self._aero_V_ref,     'aero_V_ref_kph')
            _set_spin(self._aero_cop_rear,  'aero_cop_rear_pct')
            _set_spin(self._aero_rho,       'aero_air_density')

            # Graph & corner checkboxes — block per-widget too because
            # the parent block above doesn't cover the dict children.
            if isinstance(d.get('graphs'), list):
                wanted = set(d['graphs'])
                for key, cb in self._graph_checks.items():
                    cb.blockSignals(True)
                    cb.setChecked(key in wanted)
                    cb.blockSignals(False)
            if isinstance(d.get('corners'), list):
                wanted = set(d['corners'])
                for key, cb in self._corner_cbs.items():
                    cb.blockSignals(True)
                    cb.setChecked(key in wanted)
                    cb.blockSignals(False)

            # Tire data file — only update the label if a path was
            # actually saved; empty string means "no tire loaded".
            if 'tire_path' in d:
                path = str(d['tire_path'])
                self._tire_path = path
                if path:
                    name = path.split('/')[-1].split('\\')[-1]
                    self._tire_label.setText(name)
                    self._tire_label.setStyleSheet(
                        f'color: {C_TEXT}; font-size: 11px;')
                else:
                    self._tire_label.setText('No file loaded')
                    self._tire_label.setStyleSheet(
                        f'color: {C_SUB}; font-size: 11px;')
        finally:
            for w in widgets:
                w.blockSignals(False)

        # Refresh derived label (mass inputs were set with signals blocked)
        try:
            self._refresh_total_mass()
        except Exception:
            pass

        # Refresh dependent UI now that all widgets are bulk-loaded:
        # custom-aero readout + show/hide the custom group based on the
        # restored source, and run the driving-info / dyn-constants
        # pipeline.  Either step is best-effort — a panel reconstruction
        # bug shouldn't block the file load itself.
        try:
            self._on_aero_source_changed()
            self._on_custom_aero_changed()
        except Exception:
            pass
        try:
            self._on_driving_changed()
        except Exception:
            self.params_changed.emit(self.get_params())

    def update_constants(self, veh, rc_front_m=None, rc_rear_m=None):
        """Update the computed dynamics constants display from a VehicleParams.
        rc_front_m / rc_rear_m: design roll-centre heights from the kinematic
        model (optional) — enable the roll-gradient (deg/g) readout."""
        import math
        lines = []
        # Wheel rates
        wr_f = veh.wheel_rate_front_Npm
        wr_r = veh.wheel_rate_rear_Npm
        lines.append(f'Wheel rate:  F {wr_f/175.127:.0f}  R {wr_r/175.127:.0f} lbf/in')
        # ARB wheel rates (derived from geometry — sanity check against
        # workbook B63/C63: should land near those numbers when the
        # geometry inputs match the workbook).
        arb_f = veh.arb_rate_front_Npm
        arb_r = veh.arb_rate_rear_Npm
        lines.append(f'ARB wheel:   F {arb_f/175.127:.1f}  R {arb_r/175.127:.1f} lbf/in'
                     f'   ({arb_f/1000:.2f} / {arb_r/1000:.2f} N/mm)')
        # Ride rates (series spring+tire)
        rr_f = veh.ride_rate_front_Npm
        rr_r = veh.ride_rate_rear_Npm
        lines.append(f'Ride rate:   F {rr_f/175.127:.0f}  R {rr_r/175.127:.0f} lbf/in')
        # Natural frequencies
        m_f = veh.sprung_mass_kg * veh.front_weight_fraction / 2  # per corner
        m_r = veh.sprung_mass_kg * veh.rear_weight_fraction / 2
        if m_f > 0 and rr_f > 0:
            f_f = math.sqrt(rr_f / m_f) / (2 * math.pi)
        else:
            f_f = 0
        if m_r > 0 and rr_r > 0:
            f_r = math.sqrt(rr_r / m_r) / (2 * math.pi)
        else:
            f_r = 0
        lines.append(f'Ride freq:   F {f_f:.2f}  R {f_r:.2f} Hz')
        # Roll stiffness
        rs_f = veh.roll_stiffness_front_Npm_rad
        rs_r = veh.roll_stiffness_rear_Npm_rad
        rs_t = rs_f + rs_r
        lines.append(f'Roll stiff:  F {rs_f:.0f}  R {rs_r:.0f}  T {rs_t:.0f} Nm/rad')
        # Roll stiffness distribution
        rsd = rs_f / rs_t * 100 if rs_t > 0 else 50
        lines.append(f'Roll dist:   {rsd:.1f}% front')
        # Roll gradient (deg/g) — the standard design-judge number:
        #   phi/ay = m_s · h_arm / K_roll_total, h_arm = sprung CG height
        #   above the roll axis evaluated under the CG.
        try:
            if rc_front_m is not None and rc_rear_m is not None and rs_t > 0:
                a_frac = (veh.cg_to_front_axle_m
                          / max(veh.wheelbase_m, 1e-6))
                h_ra = rc_front_m + (rc_rear_m - rc_front_m) * a_frac
                h_arm = max(veh.sprung_cg_height_m - h_ra, 0.0)
                grad = math.degrees(
                    veh.sprung_mass_kg * 9.80665 * h_arm / rs_t)
                lines.append(f'Roll grad:   {grad:.2f} deg/g'
                             f'   (arm {h_arm*1000:.0f} mm)')
        except Exception:
            pass
        # Motion ratios
        lines.append(f'MR:          F {veh.motion_ratio_front:.3f}  R {veh.motion_ratio_rear:.3f}')
        # Weight fractions
        lines.append(f'Weight:      {veh.front_weight_fraction*100:.1f}F / {veh.rear_weight_fraction*100:.1f}R')

        self._dyn_constants.setText('\n'.join(lines))

    def get_tire_path(self) -> str:
        return self._tire_path

    def show_result(self, result):
        """Populate the table from a SteadyStateResult."""
        cols = ['FL', 'FR', 'RL', 'RR']
        rows_data = [
            [f'{result.Fz.get(c, 0):.1f}' for c in cols],
            [f'{result.travel.get(c, 0):.2f}' for c in cols],
            [f'{result.camber.get(c, 0):.3f}' for c in cols],
            [f'{result.utilization.get(c, 0):.2f}' for c in cols],
        ]
        rows_data.append([
            f'{result.geometric_lt_front_N:.1f}', f'{result.geometric_lt_front_N:.1f}',
            f'{result.geometric_lt_rear_N:.1f}', f'{result.geometric_lt_rear_N:.1f}',
        ])
        rows_data.append([
            f'{result.elastic_lt_front_N:.1f}', f'{result.elastic_lt_front_N:.1f}',
            f'{result.elastic_lt_rear_N:.1f}', f'{result.elastic_lt_rear_N:.1f}',
        ])
        rows_data.append([
            f'{result.unsprung_lt_front_N:.1f}', f'{result.unsprung_lt_front_N:.1f}',
            f'{result.unsprung_lt_rear_N:.1f}', f'{result.unsprung_lt_rear_N:.1f}',
        ])

        for r, row in enumerate(rows_data):
            for c, val in enumerate(row):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._result_table.setItem(r, c, item)

        total_lt_f = (result.geometric_lt_front_N + result.elastic_lt_front_N
                      + result.unsprung_lt_front_N)
        total_lt_r = (result.geometric_lt_rear_N + result.elastic_lt_rear_N
                      + result.unsprung_lt_rear_N)
        total_lt = total_lt_f + total_lt_r
        lltd = total_lt_f / total_lt * 100 if total_lt > 0 else 50
        pitch_str = f'  |  Pitch: {result.pitch_angle_deg:.3f} deg' if abs(result.pitch_angle_deg) > 0.0001 else ''
        us_str = ''
        if abs(result.understeer_gradient_deg) > 0.001:
            us_val = result.understeer_gradient_deg
            us_str = f'  |  US: {us_val:+.2f}°'
            # Add path deviation if turn radius is set
            R = self._turn_radius.value()
            L = 1.53  # default wheelbase, overridden below if possible
            try:
                # Try to get wheelbase from the main window
                mw = self.window()
                if hasattr(mw, '_car'):
                    L = mw._car.get('wheelbase_mm', 1530) / 1000
            except Exception:
                pass
            if R > 0 and L > 0:
                import math
                us_rad = math.radians(us_val)
                denom = 1.0 - R * us_rad / L
                if abs(denom) > 0.01:
                    r_actual = R / denom
                    dev = r_actual - R
                    if abs(dev) < 50:
                        us_str += f'  →  {dev:+.1f}m wide at {R:.0f}m'
                # Handwheel angle
                sr = getattr(self, '_cached_steer_ratio', 0)
                if sr > 0:
                    ack = math.degrees(L / R)
                    hw_needed = (ack + us_val) * sr
                    us_str += f'  |  HW: {hw_needed:.0f}° (Ack {ack*sr:.0f}°)'
        self._summary.setText(
            f'Roll: {result.roll_angle_deg:.3f} deg{pitch_str}  |  '
            f'LLTD: {lltd:.1f}% front{us_str}  |  '
            f'RC: {result.rc_height_front_m*1000:.1f}/{result.rc_height_rear_m*1000:.1f} mm  |  '
            f'{result.iterations} iter')

    def show_max_g(self, info: dict):
        """Display max acceleration info below the summary."""
        parts = []
        if info.get('traction_g', 0) > 0:
            parts.append(f'Traction: {info["traction_g"]:.2f}g')
        if info.get('power_g', 0) > 0:
            parts.append(f'Power: {info["power_g"]:.2f}g')
        if info.get('effective_g', 0) > 0:
            parts.append(f'Max accel: {info["effective_g"]:.2f}g')
        if info.get('braking_g', 0) > 0:
            parts.append(f'Max brake: {info["braking_g"]:.2f}g')
        if info.get('mu_front', 0) > 0:
            parts.append(f'mu: {info["mu_front"]:.2f}F / {info["mu_rear"]:.2f}R')
        if info.get('min_turn_radius_m', 0) > 0:
            parts.append(f'R_min: {info["min_turn_radius_m"]:.2f} m')
        if parts:
            self._summary.setText(self._summary.text() + '\n' + '  |  '.join(parts))

    def set_status(self, msg: str):
        self._status.setText(msg)

    def set_solving(self, busy: bool):
        self._solve_btn.setEnabled(not busy)
        self._sweep_btn.setEnabled(not busy)

    # ── Internals ────────────────────────────────────────────────────────

    def _show_help(self):
        from PyQt6.QtWidgets import QTextBrowser
        dlg = QDialog(self)
        dlg.setWindowTitle('Dynamics Reference')
        dlg.resize(620, 700)
        dlg.setStyleSheet(
            'QDialog { background: #0a0a0a; }'
            'QTextBrowser { background: #0a0a0a; color: #e0e0e0; border: none; '
            'font-size: 12px; }')
        lay = QVBoxLayout(dlg)
        tb = QTextBrowser()
        tb.setOpenExternalLinks(False)
        tb.setHtml(_DYNAMICS_HELP)
        lay.addWidget(tb)
        close_btn = QPushButton('Close')
        close_btn.clicked.connect(dlg.accept)
        close_btn.setStyleSheet(
            'QPushButton { background: #1a5276; color: white; padding: 6px 20px; '
            'border-radius: 3px; font-weight: bold; }'
            'QPushButton:hover { background: #1f6da0; }')
        lay.addWidget(close_btn)
        dlg.exec()

    def _on_tire_browse(self):
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select tire data file', '',
            'Tire data (*.mat *.csv *.xlsx *.xls);;'
            'MATLAB files (*.mat);;CSV files (*.csv);;'
            'Excel files (*.xlsx *.xls);;All files (*)')
        if path:
            self._tire_path = path
            name = path.split('/')[-1].split('\\')[-1]
            self._tire_label.setText(name)
            self._tire_label.setStyleSheet(f'color: {C_TEXT}; font-size: 11px;')
            self.tire_file_changed.emit(path)

    def _on_solve(self):
        self.solve_requested.emit({
            'lateral_g': self._lat_g.value(),
            'longitudinal_g': self._lon_g.value(),
        })

    def _on_sweep(self):
        # New test-mode dispatch:
        #   Cornering : always route through 'combined' so the friction-
        #               circle clamp applies regardless of lon-g sign.
        #               Sub-mode is implied by the lon-g spin value:
        #                  lon = 0  → pure lateral (clamp is a no-op here)
        #                  lon < 0  → cornering + braking
        #                  lon > 0  → cornering + acceleration
        #   Straights : longitudinal trajectory.  Direction inferred
        #               from start_speed (0 = accel from rest, > 0 =
        #               brake from velocity).
        test = self._test_mode.currentData() or 'cornering'
        if test == 'straights':
            # Pure straight line — lat-g forced to 0.  The trajectory's
            # applied longitudinal-g comes from the Lon-g spin (sign
            # picks accel/brake; magnitude is the target g, clamped
            # per-step by traction / power / μ-circle).  start_speed
            # is the initial v.  Nothing else.
            mode = 'longitudinal'
            lat_g_fixed = 0.0
            lon_g_fixed = self._lon_g.value()
        else:
            mode = 'combined'             # friction-circle clamp always on
            lat_g_fixed = self._lat_g.value()
            lon_g_fixed = self._lon_g.value()
        # Sweep axis: 'g' (lat-g sweep) or 'speed' (mph sweep with R fixed).
        # Longitudinal mode is always a time-domain trajectory and
        # ignores this — start/end derive from start_speed + drag.
        sweep_axis = self._sweep_axis.currentData()
        self.sweep_requested.emit({
            'mode': mode,
            'sweep_axis':       sweep_axis,
            'g_min':            self._g_min.value(),
            'g_max':            self._g_max.value(),
            'v_min_mph':        self._v_min.value(),
            'v_max_mph':        self._v_max.value(),
            'n_points':         41,
            'lateral_g':        lat_g_fixed,
            'longitudinal_g':   lon_g_fixed,
            'turn_radius_m':    self._turn_radius.value(),
            # Acceleration-trajectory inputs.  Only the longitudinal
            # sweep uses these.
            'start_speed_mph':  self._start_speed.value(),
            'end_speed_mph':    self._end_speed.value(),
            # Direction inferred from start_speed sign (+0 = accel from
            # rest, > 0 = brake from velocity).  No separate dropdown.
            'traj_direction':   ('brake' if self._start_speed.value() > 0
                                  else 'accel'),
        })

    def _on_aero_toggle(self, checked: bool):
        self.apply_aero_toggled.emit(checked)
        if checked:
            self._apply_aero_btn.setStyleSheet(
                'QPushButton { background: #6A1B9A; color: white; padding: 5px 14px; '
                'border: 1px solid #CE93D8; border-radius: 3px; font-weight: bold; }'
                'QPushButton:hover { background: #8E24AA; }')
        else:
            self._apply_aero_btn.setStyleSheet(
                'QPushButton { background: #1a1a1a; color: #888; padding: 5px 14px; '
                'border: 1px solid #333; border-radius: 3px; font-weight: bold; }'
                'QPushButton:checked { background: #6A1B9A; color: white; border-color: #CE93D8; }'
                'QPushButton:hover { background: #2a2a2a; }')
            self._aero_label.setText('OFF')
            self._aero_label.setStyleSheet('color: #666; font-size: 10px;')

    def update_aero_label(self, total_N: float):
        """Called by main_window when aero state changes."""
        if self._apply_aero_btn.isChecked() and total_N > 0:
            self._aero_label.setText(f'+{total_N:.0f} N applied')
            self._aero_label.setStyleSheet('color: #CE93D8; font-size: 10px; font-weight: bold;')
        else:
            self._aero_label.setText('OFF')
            self._aero_label.setStyleSheet('color: #666; font-size: 10px;')

    # ── Aero source: Solved (inverse) vs Custom (CFD) ───────────────────

    def _on_test_mode_changed(self, *_):
        """Sync the hidden Lateral/Longitudinal checkboxes with the new
        Test combo, and refresh visibility of mode-specific inputs.

        Cornering  →  Lateral sweep + (combined when lon-g != 0)
        Straights  →  Longitudinal trajectory only
        """
        is_straights = (self._test_mode.currentData() == 'straights')
        # Update hidden checkboxes for the existing _on_sweep dispatch
        # to read.  Cornering = always lateral checked, longitudinal
        # off (combined is detected later by lon-g sign != 0).
        self._sweep_lat_cb.blockSignals(True)
        self._sweep_lon_cb.blockSignals(True)
        if is_straights:
            self._sweep_lat_cb.setChecked(False)
            self._sweep_lon_cb.setChecked(True)
        else:
            self._sweep_lat_cb.setChecked(True)
            self._sweep_lon_cb.setChecked(False)
        self._sweep_lat_cb.blockSignals(False)
        self._sweep_lon_cb.blockSignals(False)
        self._on_sweep_mode_changed()

    def _on_sweep_axis_changed(self, *_):
        """Defer to the mode-change handler — axis is forced to 'g' in
        the new Cornering mode anyway."""
        self._on_sweep_mode_changed()

    def _on_sweep_mode_changed(self, *_):
        """Show / hide mode-specific inputs based on sweep checkboxes.

        The dynamics panel has THREE distinct modes, mutually exclusive:

          1. Lateral only   — steady-state lat-g (or speed) sweep at
                              the panel's fixed lon-g.  Trajectory
                              inputs (Direction / Start / End / CdA /
                              ρ) are HIDDEN.
          2. Longitudinal only — time-domain acceleration or braking
                              trajectory.  `Sweep by`, `Range g`, and
                              `Range mph` are HIDDEN.
          3. Both checked (Combined) — steady-state lat-g sweep at the
                              user-set fixed lon-g.  STILL a steady-
                              state sweep, NOT a trajectory — trajectory
                              inputs are HIDDEN here too because they
                              don't affect the combined solve.

        Net rule:
          • Trajectory inputs visible iff long-only.
          • Lateral-axis inputs visible iff lateral checked.
        """
        is_cornering = (self._test_mode.currentData() == 'cornering')
        is_straights = not is_cornering

        # Cornering visible inputs:  lat-g range + lat-g spin + lon-g spin.
        # Straights visible inputs:  start_speed + lon-g spin only.
        # Everything else hidden in both modes (Poka Yoke — user can't
        # enter a value that doesn't apply to the active test).
        self._axis_host.setVisible(False)
        self._g_sweep_host.setVisible(is_cornering)
        self._v_sweep_host.setVisible(False)

        # Lat-g spin: cornering only (Straights is straight-line, lat=0)
        if hasattr(self, '_lat_row_host'):
            self._lat_row_host.setVisible(is_cornering)
        # Lon-g spin: visible in BOTH modes —
        #   Cornering : held lon-g for the combined sweep
        #   Straights : applied lon-g for the trajectory
        if hasattr(self, '_lon_row_host'):
            self._lon_row_host.setVisible(True)

        # Straights-only: start_speed.  end_speed / CdA / ρ no longer
        # surface in the UI per the "two inputs only" spec — they
        # remain VehicleParams defaults internally.
        if hasattr(self, '_traj_direction'):
            self._traj_direction.setVisible(False)
        if hasattr(self, '_start_speed'):
            self._start_speed.setVisible(is_straights)
            if hasattr(self, '_start_speed_label'):
                self._start_speed_label.setVisible(is_straights)
        for w in (getattr(self, '_end_speed', None),
                  getattr(self, '_cda', None),
                  getattr(self, '_air_rho', None)):
            if w is not None:
                w.setVisible(False)

    def _on_aero_source_changed(self, *_):
        """Show/hide the custom-aero group based on the source combobox.
        Also re-emit ``apply_aero_toggled`` so main_window re-computes the
        aero load with the now-active source."""
        is_custom = (self._aero_source.currentData() == 'custom')
        self._aero_custom_box.setVisible(is_custom)
        # Re-fire the toggle signal so downstream wiring re-queries the
        # active source — without this, switching from Solved → Custom
        # while "Apply Aero" is on leaves stale numbers in the dynamics.
        if self._apply_aero_btn.isChecked():
            self.apply_aero_toggled.emit(True)

    def _on_custom_aero_changed(self, *_):
        """Refresh the read-only ``CL·A`` + downforce-at-ref readout.

        Same V²-scaling identity used downstream:
            CL·A = 2 · F_ref / (ρ · V_ref²)
            F(V) = 0.5 · ρ · V² · CL·A
        Echoes back the equivalent CL·A in m² and the downforce at a few
        useful reference speeds so the user can sanity-check the numbers
        against their CFD output before switching Apply Aero on.
        """
        if not hasattr(self, '_aero_custom_info'):
            return
        F_ref = float(self._aero_F_ref.value())
        V_ref_kph = float(self._aero_V_ref.value())
        rho = float(self._aero_rho.value())
        rear_pct = float(self._aero_cop_rear.value())
        V_ref_ms = V_ref_kph / 3.6
        if V_ref_ms <= 0 or rho <= 0:
            self._aero_custom_info.setText('Invalid ref speed or air density.')
            return
        CLA = 2.0 * F_ref / (rho * V_ref_ms ** 2)
        # Echo at three speed points: 30/60/100 km/h
        F_30  = 0.5 * rho * (30/3.6) ** 2 * CLA
        F_60  = 0.5 * rho * (60/3.6) ** 2 * CLA
        F_100 = 0.5 * rho * (100/3.6) ** 2 * CLA
        f_split = (1.0 - rear_pct / 100.0)
        r_split = (rear_pct / 100.0)
        self._aero_custom_info.setText(
            f"CL·A = {CLA:.3f} m²   |   "
            f"F at 30/60/100 km/h = {F_30:.0f} / {F_60:.0f} / {F_100:.0f} N\n"
            f"Per-axle split @ ref:  F = {F_ref*f_split:.0f} N   "
            f"R = {F_ref*r_split:.0f} N  ({rear_pct:.0f}% rear)"
        )
        self.params_changed.emit(self.get_params())

    def get_custom_aero_params(self) -> dict:
        """Public getter for main_window's `_get_aero_Fz_per_g` to read
        when the source combobox is in Custom mode.  Returns a plain
        dict so the wiring code doesn't need to know about the spinboxes
        directly."""
        return {
            'F_ref_N':       float(self._aero_F_ref.value()),
            'V_ref_kph':     float(self._aero_V_ref.value()),
            'cop_rear_pct':  float(self._aero_cop_rear.value()),
            'air_density':   float(self._aero_rho.value()),
        }

    def get_aero_source(self) -> str:
        """Returns 'solved' or 'custom'."""
        return str(self._aero_source.currentData() or 'solved')

    def _on_graph_changed(self):
        self.graph_selection_changed.emit(self.get_selected_graphs())

    def _on_corners_changed(self):
        self.corners_changed.emit(self.get_selected_corners())

    def _refresh_total_mass(self):
        """Recompute and display the derived total mass = sum of inputs."""
        total = (self._sprung_mass.value()
                 + self._us_front.value()
                 + self._us_rear.value())
        self._total_mass_lbl.setText(f'{total:.2f} kg')

    def tire_plot_inputs(self) -> dict:
        """Friction-circle options for the Tire / Grip Plots popup:
        user Fz levels (N) or None for automatic, and the 3D toggle."""
        fz_levels = None
        try:
            txt = self._fc_fz_edit.text().strip()
            if txt:
                vals = [float(t) for t in txt.replace(';', ',').split(',')
                        if t.strip()]
                fz_levels = [v for v in vals if v > 0] or None
        except Exception:
            fz_levels = None
        return dict(fz_levels=fz_levels, fc_3d=self._fc_3d_chk.isChecked())

    def _refresh_wheel_power(self):
        """Show wheel power = engine power × drivetrain efficiency."""
        hp = self._power_hp.value()
        eff = self._drivetrain_eff.value()
        wheel_hp = hp * eff
        wheel_kw = wheel_hp * 0.7457
        self._wheel_power_lbl.setText(
            f'  wheel power = {wheel_hp:.1f} hp  ({wheel_kw:.1f} kW)')

    def _on_powertrain_type_changed(self):
        """Show/hide the EV-only peak-torque input + refresh wheel-power."""
        # Guard against firing during __init__ before sibling widgets exist
        if not hasattr(self, '_peak_torque'):
            return
        is_ev = (self._powertrain_type.currentText() == 'EV')
        self._peak_torque.setEnabled(is_ev)
        self._peak_torque.setStyleSheet(
            '' if is_ev else 'color:#555;')
        try:
            self._refresh_wheel_power()
        except Exception:
            pass
        # Only emit params_changed if the rest of the panel has finished
        # being constructed.
        if hasattr(self, '_drivetrain_eff') and hasattr(self, '_final_ratio'):
            try:
                self.params_changed.emit(self.get_params())
            except Exception:
                pass

    def _on_driving_changed(self):
        """Auto-calculate g values from RPM, gear ratio, turn radius."""
        import math
        hp = self._power_hp.value()
        rpm = self._engine_rpm.value()
        ratio = self._final_ratio.value()          # single final drive ratio
        eff = self._drivetrain_eff.value()
        r_tire = self._tire_radius.value() / 1000  # m
        R_turn = self._turn_radius.value()
        # Total mass is the derived sum (no longer an input field).
        mass = (self._sprung_mass.value()
                + self._us_front.value() + self._us_rear.value())

        parts = [f'ratio: {ratio:.2f}:1']

        if rpm > 0 and ratio > 0 and r_tire > 0:
            speed_ms = rpm * 2 * math.pi * r_tire / (ratio * 60)
            speed_mph = speed_ms * 2.23694
            parts.append(f'{speed_mph:.1f} mph')

            if R_turn > 0:
                lat_g = speed_ms ** 2 / (R_turn * 9.81)
                parts.append(f'lat: {lat_g:.2f}g')

        if hp > 0 and rpm > 0 and ratio > 0 and r_tire > 0 and mass > 0:
            # Wheel power = engine power × drivetrain efficiency.  Drive
            # force at GEARED TOP SPEED = wheel_power / v_top.  EV: cap
            # by motor-torque-limited wheel force at low speed.
            P_engine_W = hp * 745.7
            P_wheel_W  = P_engine_W * eff
            omega      = rpm * 2 * math.pi / 60
            T_engine   = P_engine_W / omega     # engine-shaft torque (Nm)
            v_top_ms   = rpm * 2 * math.pi * r_tire / (60 * ratio)
            F_pwr_top  = P_wheel_W / max(v_top_ms, 0.5)
            # EV traction floor
            if self._powertrain_type.currentText() == 'EV':
                T_motor = self._peak_torque.value()
                F_torque_max = T_motor * ratio * eff / max(r_tire, 1e-3)
                F_drive = min(F_torque_max, F_pwr_top)
            else:
                F_drive = F_pwr_top
            accel_g = F_drive / (mass * 9.81)
            parts.append(f'accel@top: {accel_g:.2f}g')
            parts.append(f'T_eng: {T_engine:.1f} Nm')

            # 0-60 mph estimate (constant accel approximation)
            v_60 = 26.82  # m/s
            if accel_g > 0:
                # Traction-limited approx: t = v / a
                t_traction = v_60 / (accel_g * 9.81)
                # Power-limited (energy method): t = m·v² / (2·P_wheel)
                t_power = mass * v_60 ** 2 / (2 * P_wheel_W) if P_wheel_W > 0 else 99
                # Real 0-60 is between these; the limiting one dominates.
                t_060 = max(t_traction, t_power)
                parts.append(f'0-60: {t_060:.1f}s')

        # R_min is computed from geometry in _build_dynamics_solver
        if hasattr(self, '_cached_r_min') and self._cached_r_min > 0:
            parts.append(f'R_min: {self._cached_r_min:.2f} m')

        # Weight distribution from CG position + wheelbase
        try:
            mw = self.window()
            if hasattr(mw, '_car'):
                car = mw._car
                wb = car.get('wheelbase_mm', 0) / 1000
                cg_y = car.get('cg_y_mm', 0) / 1000  # CG distance from front axle
                if wb > 0 and cg_y > 0:
                    front_pct = (wb - cg_y) / wb * 100
                    parts.insert(0, f'W: {front_pct:.1f}F / {100-front_pct:.1f}R')
        except Exception:
            pass

        self._driving_info.setText('  |  '.join(parts) if parts else '')

        # Update computed dynamics constants
        try:
            from vahan.dynamics import VehicleParams
            p = self.get_params()
            mw = self.window()
            if hasattr(mw, '_car'):
                car = mw._car
                p['front_track_m'] = car.get('track_f_mm', 1222) / 1000
                p['rear_track_m'] = car.get('track_r_mm', 1200) / 1000
                p['wheelbase_m'] = car.get('wheelbase_mm', 1537) / 1000
                p['cg_height_m'] = car.get('cg_z_mm', 280) / 1000
                p['cg_to_front_axle_m'] = car.get('cg_y_mm', 1100) / 1000
                if 'front_brake_bias_pct' in car:
                    p['front_brake_bias'] = car['front_brake_bias_pct'] / 100
            veh = VehicleParams(**{k: v for k, v in p.items()
                                   if k in VehicleParams.__dataclass_fields__})
            self.update_constants(veh)
        except Exception:
            pass

        self.params_changed.emit(self.get_params())

    def get_selected_graphs(self) -> list:
        return [k for k, cb in self._graph_checks.items() if cb.isChecked()]

    def get_selected_corners(self) -> list:
        return [lbl for lbl, cb in self._corner_cbs.items() if cb.isChecked()]



# ══════════════════════════════════════════════════════════════════════════════
#  AERO DOWNFORCE PANEL
# ══════════════════════════════════════════════════════════════════════════════

class AeroPanel(CollapsibleSection):
    """Per-corner additional Fz needed for target utilization."""
    solve_requested = pyqtSignal(dict)   # lateral_g, longitudinal_g, target_util
    sweep_requested = pyqtSignal(dict)

    def __init__(self):
        super().__init__('Aero Load Targets', header_color='#CE93D8')
        self.set_info(section_info.AERO)
        self._build()

    def _build(self):
        g = QGridLayout(); g.setSpacing(4); r = 0
        g.addWidget(QLabel('Lateral g:'), r, 0)
        self._lat_g = _spin(-99, 99, 1.5, ' g', dec=2, step=0.1); g.addWidget(self._lat_g, r, 1); r += 1
        g.addWidget(QLabel('Long. g:'), r, 0)
        self._lon_g = _spin(-99, 99, 0, ' g', dec=2, step=0.1); g.addWidget(self._lon_g, r, 1); r += 1
        g.addWidget(QLabel('Target util:'), r, 0)
        self._tgt = _spin(0.1, 1.0, 0.80, '', dec=2, step=0.05); g.addWidget(self._tgt, r, 1); r += 1
        self.add_layout(g)

        btn_row = QHBoxLayout()
        self._solve_btn = QPushButton('Solve')
        self._solve_btn.setStyleSheet(
            'QPushButton { background: #6A1B9A; color: white; padding: 5px 14px; '
            'border-radius: 3px; font-weight: bold; }'
            'QPushButton:hover { background: #8E24AA; }')
        self._solve_btn.clicked.connect(self._on_solve)
        btn_row.addWidget(self._solve_btn)
        self._sweep_btn = QPushButton('Sweep')
        self._sweep_btn.setStyleSheet(
            'QPushButton { background: #4A148C; color: white; padding: 5px 14px; '
            'border-radius: 3px; font-weight: bold; }'
            'QPushButton:hover { background: #6A1B9A; }')
        self._sweep_btn.clicked.connect(self._on_sweep)
        btn_row.addWidget(self._sweep_btn)
        self.add_layout(btn_row)

        self._status = QLabel('')
        self._status.setStyleSheet('color: #888; font-size: 11px;')
        self._status.setWordWrap(True)
        self.add_widget(self._status)

        # Results table
        self._tbl = QTableWidget(5, 3)
        self._tbl.setHorizontalHeaderLabels(['', 'Addl Fz (N)', 'Util after'])
        self._tbl.verticalHeader().setVisible(False)
        self._tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._tbl.setMaximumHeight(320)
        self._tbl.setStyleSheet(
            'QTableWidget { background: #0d0d0d; color: #e0e0e0; gridline-color: #222; '
            'font-size: 11px; } QHeaderView::section { background: #1a1a1a; color: #aaa; '
            'border: 1px solid #222; font-size: 10px; }')
        self.add_widget(self._tbl)

        self._summary = QLabel('')
        self._summary.setStyleSheet('color: #CE93D8; font-size: 11px; font-weight: bold;')
        self._summary.setWordWrap(True)
        self.add_widget(self._summary)

    def _on_solve(self):
        self.solve_requested.emit({
            'lateral_g': self._lat_g.value(),
            'longitudinal_g': self._lon_g.value(),
            'target_util': self._tgt.value(),
        })

    def _on_sweep(self):
        self.sweep_requested.emit({
            'lateral_g': self._lat_g.value(),
            'longitudinal_g': self._lon_g.value(),
            'target_util': self._tgt.value(),
        })

    # ── Save / load (every input on the panel) ───────────────────────
    def get_state(self) -> dict:
        """JSON-friendly snapshot of every input on the aero panel."""
        return {
            'lateral_g':      float(self._lat_g.value()),
            'longitudinal_g': float(self._lon_g.value()),
            'target_util':    float(self._tgt.value()),
        }

    def set_state(self, d: dict) -> None:
        """Restore every input from a dict produced by :meth:`get_state`."""
        if not isinstance(d, dict):
            return
        for key, sb in (
            ('lateral_g',      self._lat_g),
            ('longitudinal_g', self._lon_g),
            ('target_util',    self._tgt),
        ):
            if key in d:
                try:
                    sb.blockSignals(True)
                    sb.setValue(float(d[key]))
                except (TypeError, ValueError):
                    pass
                finally:
                    sb.blockSignals(False)

    def show_result(self, r):
        """r: AeroResult — per-corner deficit, axle needs, total, rear bias."""
        df = r.downforce
        ut = r.utilization_aero

        self._tbl.setHorizontalHeaderLabels(['', 'Deficit (N)', 'Util after'])

        rows = [('FL', df.get('FL', 0), ut.get('FL', 0)),
                ('FR', df.get('FR', 0), ut.get('FR', 0)),
                ('RL', df.get('RL', 0), ut.get('RL', 0)),
                ('RR', df.get('RR', 0), ut.get('RR', 0))]

        front_need = r.front_axle_need_N
        rear_need  = r.rear_axle_need_N
        total      = r.total_downforce_N

        rows.append(('', '', None))
        rows.append(('Front axle need',  f'{front_need:.0f}', None))
        rows.append(('Rear axle need',   f'{rear_need:.0f}', None))
        rows.append(('Total deficit',    f'{total:.0f}', None))
        rows.append(('Rear aero bias',   f'{r.rear_aero_bias_pct:.1f}%', None))

        self._tbl.setRowCount(len(rows))
        for i, (lbl, val, u) in enumerate(rows):
            self._tbl.setItem(i, 0, QTableWidgetItem(str(lbl)))
            self._tbl.setItem(i, 1, QTableWidgetItem(
                f'{val:.1f}' if isinstance(val, (int, float)) else str(val)))
            self._tbl.setItem(i, 2, QTableWidgetItem(
                f'{u:.3f}' if isinstance(u, float) else (str(u) if u is not None else '')))

        cap = ''
        if r.capped:
            cap = f'  |  Capped: {", ".join(r.capped)}'

        self._summary.setText(
            f'Deficit F:{front_need:.0f}  R:{rear_need:.0f}  '
            f'Tot:{total:.0f} N  |  '
            f'Bias: {r.rear_aero_bias_pct:.0f}%{cap}')
        self._status.setText(
            f'Solved @ {r.lateral_g:.2f}g lat, {r.longitudinal_g:.2f}g lon')



# ══════════════════════════════════════════════════════════════════════════════
#  COMPONENT LOADS PANEL
# ══════════════════════════════════════════════════════════════════════════════

class LoadsPanel(CollapsibleSection):
    """
    Component force calculator.

    Separate front/rear brake params, upright geometry, bearing and
    caliper bolt force output with V/H decomposition.
    """
    loads_requested = pyqtSignal()   # emitted when user clicks Compute
    wheel_package_requested = pyqtSignal()   # opens the Wheel Package Load Analysis

    def __init__(self):
        super().__init__('Component Loads', header_color='#E53935')
        self.set_info(section_info.LOADS)
        self._build()

    # ── helpers ───────────────────────────────────────────────────────
    def _brake_section(self, title):
        """Build a grid of brake-parameter spinners, return (layout, dict_of_spins)."""
        g = QGridLayout(); g.setSpacing(4)
        lbl = QLabel(title)
        lbl.setStyleSheet('font-weight: bold; color: #FFA726;')
        g.addWidget(lbl, 0, 0, 1, 4)

        s = {}
        r = 1
        g.addWidget(QLabel('Pad mu:'), r, 0)
        s['pad_mu'] = _spin(0.1, 1.0, 0.45, '', dec=2, step=0.05)
        g.addWidget(s['pad_mu'], r, 1)
        g.addWidget(QLabel('Piston area:'), r, 2)
        s['piston_area'] = _spin(0.1, 1e6, 793.5, ' mm\u00b2', dec=1, step=50)
        g.addWidget(s['piston_area'], r, 3)

        r += 1
        g.addWidget(QLabel('Pad radius:'), r, 0)
        s['pad_radius'] = _spin(0.1, 1e6, 94.4, ' mm', dec=1, step=5)
        g.addWidget(s['pad_radius'], r, 1)
        g.addWidget(QLabel('Pistons/cal:'), r, 2)
        s['num_pistons'] = _spin(1, 99, 1, '', dec=0, step=1)
        g.addWidget(s['num_pistons'], r, 3)

        # ── CALIPER MOUNT geometry, straight off the caliper drawing ──────
        # Wilwood GP200-style radial mount.  These three numbers plus the rotor
        # diameter FIX where the caliper sits — the drawing's own
        #   D1 = disc_dia/2 - MOUNT HEIGHT   (radius to the bolt line)
        #   A  = sqrt(D1^2 + (MOUNT CENTER/2)^2)   (bolt circle radius)
        # Seward's l4 (pad centre offset from the bolt line) is then DERIVED, not
        # typed: l3 + l4 = R_pad identically.  It used to be a free 25 mm input,
        # which put the bolt line 23 mm from where the drawing puts it.
        r += 1
        g.addWidget(QLabel('Rotor dia:'), r, 0)
        s['rotor_dia'] = _spin(1, 1e6, 240.0, ' mm', dec=1, step=5)
        g.addWidget(s['rotor_dia'], r, 1)
        g.addWidget(QLabel('Mount centre l5:'), r, 2)
        s['bolt_spacing'] = _spin(0.1, 1e6, 60.5, ' mm', dec=1, step=5)
        g.addWidget(s['bolt_spacing'], r, 3)

        r += 1
        g.addWidget(QLabel('Mount height:'), r, 0)
        s['mount_height'] = _spin(0.0, 1e6, 27.9, ' mm', dec=1, step=1)
        s['mount_height'].setToolTip(
            'Drawing MOUNT HEIGHT — the caliper bolt line sits this far BELOW the '
            'disc outside diameter. Bolt-line radius D1 = rotor_dia/2 - this.')
        g.addWidget(s['mount_height'], r, 1)
        g.addWidget(QLabel('Mount offset:'), r, 2)
        s['mount_offset'] = _spin(0.0, 1e6, 21.8, ' mm', dec=1, step=1)
        s['mount_offset'].setToolTip(
            'Drawing MOUNT OFFSET — lateral offset of the bolt plane from the '
            'disc face.')
        g.addWidget(s['mount_offset'], r, 3)

        r += 1
        g.addWidget(QLabel('Rotor thickness:'), r, 0)
        s['rotor_thickness'] = _spin(0.5, 100.0, 6.35, ' mm', dec=2, step=1)
        s['rotor_thickness'].setToolTip(
            'Disc width. Does NOT change lockup torque or line pressure — it sets '
            'thermal mass, so a thicker rotor runs proportionally cooler per stop.')
        g.addWidget(s['rotor_thickness'], r, 1)

        # Read-out of what the drawing dimensions imply, so a wrong entry is
        # visible immediately instead of only showing up in the load numbers.
        r += 1
        s['_mount_readout'] = QLabel('')
        s['_mount_readout'].setStyleSheet('color:#888; font-size:11px;')
        g.addWidget(s['_mount_readout'], r, 0, 1, 4)

        def _upd_readout():
            from vahan.loads import BrakeParams
            bp = BrakeParams(pad_radius_mm=s['pad_radius'].value(),
                             rotor_dia_mm=s['rotor_dia'].value(),
                             caliper_bolt_spacing_mm=s['bolt_spacing'].value(),
                             caliper_mount_height_mm=s['mount_height'].value())
            s['_mount_readout'].setText(
                f'bolt line D1 {bp.bolt_line_radius_mm:.1f} mm · bolt circle A '
                f'{bp.bolt_circle_radius_mm:.1f} mm · derived pad offset l4 '
                f'{bp.caliper_l4_mm:.1f} mm')
        for _k in ('pad_radius', 'rotor_dia', 'bolt_spacing', 'mount_height'):
            s[_k].valueChanged.connect(lambda *_a: _upd_readout())
        _upd_readout()

        return g, s

    def _build(self):
        # ── Front brake ──────────────────────────────────────────────
        lay_f, self._brk_f = self._brake_section('Front Brakes')
        self.add_layout(lay_f)

        # ── Rear brake ───────────────────────────────────────────────
        lay_r, self._brk_r = self._brake_section('Rear Brakes')
        self.add_layout(lay_r)

        # ── Upright / bearing geometry ───────────────────────────────
        upr = QGridLayout(); upr.setSpacing(4)
        lbl = QLabel('Upright / Bearings')
        lbl.setStyleSheet('font-weight: bold; color: #FFA726;')
        upr.addWidget(lbl, 0, 0, 1, 4)

        r = 1
        upr.addWidget(QLabel('Bearing spacing:'), r, 0)
        self._brg_spacing = _spin(0.1, 1e6, 50.8, ' mm', dec=1, step=5)
        self._brg_spacing.setToolTip('Bearing center-to-center along the spindle '
                                     '(2.00 in on the outgoing car)')
        upr.addWidget(self._brg_spacing, r, 1)
        upr.addWidget(QLabel('Bearing inboard:'), r, 2)
        self._brg_inboard = _spin(0, 1e6, 39.4, ' mm', dec=1, step=2)
        self._brg_inboard.setToolTip(
            'How far the NEAR (outer) bearing sits inboard of the wheel '
            'centre-line. Both bearings are inboard, so the tyre load is '
            'overhung. 1.55 in on the outgoing car.')
        upr.addWidget(self._brg_inboard, r, 3)

        r += 1
        from PyQt6.QtWidgets import QCheckBox
        self._cal_vert = QCheckBox('Vertical mounts (tie-rod side)')
        self._cal_vert.setChecked(True)
        self._cal_vert.setToolTip(
            'Clock the caliper so its two mount bolts stack VERTICALLY, on the '
            'tie-rod side of the wheel \u2014 one vertical mounting boss, shared with '
            'the steering-arm face, simplifies the upright. Uncheck to set the '
            'clock angle manually.')
        upr.addWidget(self._cal_vert, r, 0, 1, 2)
        upr.addWidget(QLabel('Caliper angle:'), r, 2)
        self._cal_angle = _spin(0, 360, 45, '\u00b0', dec=0, step=15)
        self._cal_angle.setToolTip(
            'Manual caliper clock (degrees from top of disc, CW from outboard '
            'view) \u2014 used only when vertical mounts is unchecked')
        upr.addWidget(self._cal_angle, r, 3)

        self.add_layout(upr)

        # ── Compute button ───────────────────────────────────────────
        self._compute_btn = QPushButton('Compute Loads')
        self._compute_btn.setStyleSheet(
            'QPushButton { background: #8B0000; color: white; padding: 6px 16px; '
            'border-radius: 3px; font-weight: bold; }'
            'QPushButton:hover { background: #B22222; }')
        self._compute_btn.clicked.connect(lambda: self.loads_requested.emit())
        self.add_widget(self._compute_btn)

        # ── Wheel Package Load Analysis (Seward-style upright + arms) ──
        self._wpkg_btn = QPushButton('Wheel Package Load Analysis...')
        self._wpkg_btn.setToolTip(
            'Open the Seward-style wheel-package analysis: the UPRIGHT free body '
            '(bearings + caliper + ball-joint loads, component or resultant '
            'vectors) and the CONTROL-ARM axial forces, per case and corner.')
        self._wpkg_btn.setStyleSheet(
            'QPushButton { background: #37474F; color: white; padding: 6px 16px; '
            'border-radius: 3px; font-weight: bold; }'
            'QPushButton:hover { background: #455A64; }')
        self._wpkg_btn.clicked.connect(lambda: self.wheel_package_requested.emit())
        self.add_widget(self._wpkg_btn)

        # ── Status ───────────────────────────────────────────────────
        self._loads_status = QLabel('')
        self._loads_status.setStyleSheet('color: #888; font-size: 11px;')
        self.add_widget(self._loads_status)

    # ── param getters ────────────────────────────────────────────────
    def _bp_from(self, d):
        from vahan.loads import BrakeParams
        return BrakeParams(
            pad_mu=d['pad_mu'].value(),
            piston_area_mm2=d['piston_area'].value(),
            pad_radius_mm=d['pad_radius'].value(),
            num_pistons=int(d['num_pistons'].value()),
            caliper_bolt_spacing_mm=d['bolt_spacing'].value(),
            # Caliper MOUNT geometry off the caliper drawing.  l4 is NOT passed:
            # it is derived from the rotor and the mount height (l3 + l4 = R_pad),
            # so the bolt line cannot be set to something the drawing contradicts.
            rotor_dia_mm=(d['rotor_dia'].value() if 'rotor_dia' in d else 240.0),
            rotor_thickness_mm=(d['rotor_thickness'].value()
                                if 'rotor_thickness' in d else 6.35),
            caliper_mount_height_mm=(d['mount_height'].value()
                                     if 'mount_height' in d else 27.9),
            caliper_mount_offset_mm=(d['mount_offset'].value()
                                     if 'mount_offset' in d else 21.8),
        )

    def get_brake_params_front(self):
        return self._bp_from(self._brk_f)

    def get_brake_params_rear(self):
        return self._bp_from(self._brk_r)

    def get_upright_params(self):
        from vahan.loads import UprightParams
        return UprightParams(
            bearing_spacing_mm=self._brg_spacing.value(),
            bearing_inboard_offset_mm=self._brg_inboard.value(),
            caliper_angle_deg=self._cal_angle.value(),
            caliper_vertical_mounts=self._cal_vert.isChecked(),
        )

    # ── Save / load (every input on the panel) ───────────────────────
    @staticmethod
    def _brake_state(brk: dict) -> dict:
        """Snapshot one brake-section's spinbox dict to plain floats."""
        return {k: float(sb.value()) for k, sb in brk.items()}

    def _apply_brake_state(self, brk: dict, d: dict) -> None:
        """Restore one brake-section's spinbox dict from a saved dict."""
        if not isinstance(d, dict):
            return
        for k, sb in brk.items():
            if k in d:
                try:
                    sb.blockSignals(True)
                    sb.setValue(float(d[k]))
                finally:
                    sb.blockSignals(False)

    def get_state(self) -> dict:
        """JSON-friendly snapshot of every input on the loads panel:
        front + rear brake parameters and the upright/bearing geometry.
        These feed the static load calculator, so they're as much an
        "input" as the dynamics inputs.
        """
        return {
            'brake_front':    self._brake_state(self._brk_f),
            'brake_rear':     self._brake_state(self._brk_r),
            'bearing_spacing_mm':       float(self._brg_spacing.value()),
            'bearing_inboard_offset_mm': float(self._brg_inboard.value()),
            'caliper_angle_deg':        float(self._cal_angle.value()),
            'caliper_vertical_mounts':  bool(self._cal_vert.isChecked()),
        }

    def set_state(self, d: dict) -> None:
        """Restore every input from a dict produced by :meth:`get_state`.
        Tolerates missing keys."""
        if not isinstance(d, dict):
            return
        self._apply_brake_state(self._brk_f, d.get('brake_front', {}))
        self._apply_brake_state(self._brk_r, d.get('brake_rear', {}))
        for key, sb in (
            ('bearing_spacing_mm',       self._brg_spacing),
            ('bearing_inboard_offset_mm', self._brg_inboard),
            ('caliper_angle_deg',        self._cal_angle),
        ):
            if key in d:
                try:
                    sb.blockSignals(True)
                    sb.setValue(float(d[key]))
                except (TypeError, ValueError):
                    pass
                finally:
                    sb.blockSignals(False)
        if 'caliper_vertical_mounts' in d:
            self._cal_vert.blockSignals(True)
            self._cal_vert.setChecked(bool(d['caliper_vertical_mounts']))
            self._cal_vert.blockSignals(False)

    # ── results popup ────────────────────────────────────────────────
    def show_loads(self, loads: dict, lat_g: float = 0.0, lon_g: float = 0.0):
        """Display computed loads in a popup dialog with V/H decomposition."""
        self._loads_status.setText(
            f'Computed at {lat_g:.2f}g lateral, {lon_g:.2f}g longitudinal')

        # Row defs: (label, attr, unit, decimals)
        # 'header' = section header; attr=None = separator
        # 'vh_color' attrs get up/down coloring instead of tension/compression
        _AXIAL_ATTRS = frozenset({
            'uca_front_N', 'uca_rear_N', 'lca_front_N',
            'lca_rear_N', 'tierod_N', 'pushrod_N'})

        rows = [
            ('header', 'TIRE CONTACT PATCH LOADS', None, None),
            ('Fz  ground reaction (up+)',          'Fz_N',       'N',  0),
            ('Fy  lateral cornering force',        'Fy_N',       'N',  0),
            ('Fx  longitudinal (fwd+)',            'Fx_N',       'N',  0),
            ('', None, None, None),

            ('header', 'MEMBER AXIAL FORCES  (+ tension / - compression)', None, None),
            ('UCA front arm',  'uca_front_N', 'N', 0),
            ('UCA rear arm',   'uca_rear_N',  'N', 0),
            ('LCA front arm',  'lca_front_N', 'N', 0),
            ('LCA rear arm',   'lca_rear_N',  'N', 0),
            ('Tie rod',        'tierod_N',    'N', 0),
            ('Pushrod',        'pushrod_N',   'N', 0),
            ('Spring (comp+)', 'spring_force_N', 'N', 0),
            ('', None, None, None),

            ('header', 'BALL JOINT REACTIONS  (V=up+, H=fwd+)', None, None),
            ('UCA BJ  V',     'uca_bj_V',     'N', 0),
            ('UCA BJ  H',     'uca_bj_H',     'N', 0),
            ('LCA BJ  V',     'lca_bj_V',     'N', 0),
            ('LCA BJ  H',     'lca_bj_H',     'N', 0),
            ('Tie rod  V',    'tierod_bj_V',  'N', 0),
            ('Tie rod  H',    'tierod_bj_H',  'N', 0),
            ('Pushrod  V',    'pushrod_bj_V', 'N', 0),
            ('Pushrod  H',    'pushrod_bj_H', 'N', 0),
            ('', None, None, None),

            ('header', 'BEARING LOADS  (V=up+, H=fwd+)', None, None),
            ('Inner bearing V',  'bearing_inner_V', 'N', 0),
            ('Inner bearing H',  'bearing_inner_H', 'N', 0),
            ('Outer bearing V',  'bearing_outer_V', 'N', 0),
            ('Outer bearing H',  'bearing_outer_H', 'N', 0),
            ('', None, None, None),

            ('header', 'CALIPER MOUNT BOLTS  (V=up+, H=fwd+)', None, None),
            ('Upper bolt V',  'caliper_upper_V', 'N', 0),
            ('Upper bolt H',  'caliper_upper_H', 'N', 0),
            ('Lower bolt V',  'caliper_lower_V', 'N', 0),
            ('Lower bolt H',  'caliper_lower_H', 'N', 0),
            ('', None, None, None),

            ('header', 'BRAKE SYSTEM', None, None),
            ('Brake torque',                        'brake_torque_Nm',    'Nm',  1),
            ('Caliper clamp (both pads)',            'caliper_clamp_N',    'N',   0),
            ('Line pressure',                       'line_pressure_MPa',  'MPa', 2),
        ]

        # ── popup ────────────────────────────────────────────────────
        dlg = QDialog(self)
        dlg.setWindowTitle('Component Loads')
        dlg.setMinimumSize(780, 700)
        dlg.setStyleSheet('''
            QDialog { background: #000; color: #e0e0e0; }
            QLabel  { color: #e0e0e0; }
            QTableWidget { background: #0a0a0a; color: #e0e0e0;
                           gridline-color: #2a2a2a; border: none; font-size: 12px; }
            QHeaderView::section { background: #111; color: #ccc;
                                   border: 1px solid #2a2a2a; padding: 4px;
                                   font-weight: bold; font-size: 12px; }
        ''')
        lay = QVBoxLayout(dlg)

        # Operating conditions
        lon_desc = ''
        if lon_g < -0.01:
            lon_desc = f', {abs(lon_g):.2f}g braking'
        elif lon_g > 0.01:
            lon_desc = f', {lon_g:.2f}g acceleration'
        cond = QLabel(f'Operating point:  {lat_g:.2f}g lateral{lon_desc}')
        cond.setStyleSheet(
            'color: #FFA726; font-size: 14px; font-weight: bold; padding: 6px;')
        lay.addWidget(cond)

        # Table
        tbl = QTableWidget(len(rows), 4)
        tbl.setHorizontalHeaderLabels(['FL', 'FR', 'RL', 'RR'])
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        tbl.setSelectionMode(QTableWidget.SelectionMode.NoSelection)

        vlabels = []
        for ri, (label, attr, unit, dec) in enumerate(rows):
            # section header
            if label == 'header':
                vlabels.append('')
                for c in range(4):
                    it = QTableWidgetItem(attr if c == 0 else '')
                    if c == 0:
                        it.setForeground(QColor('#FFA726'))
                        f = it.font(); f.setBold(True); it.setFont(f)
                    it.setBackground(QColor('#111111'))
                    tbl.setItem(ri, c, it)
                tbl.setSpan(ri, 0, 1, 4)
                continue

            # separator
            if attr is None:
                vlabels.append('')
                for c in range(4):
                    it = QTableWidgetItem('')
                    it.setBackground(QColor('#0a0a0a'))
                    tbl.setItem(ri, c, it)
                continue

            vlabels.append(f'{label} ({unit})')
            for c, corner in enumerate(['FL', 'FR', 'RL', 'RR']):
                cl = loads.get(corner)
                val = getattr(cl, attr, 0) if cl else 0
                val = val + 0.0  # eliminate -0.0
                txt = f'{val:.{dec}f}'

                it = QTableWidgetItem(txt)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                # colour: axial → tension green / compression red
                if attr in _AXIAL_ATTRS:
                    if val > 10:
                        it.setForeground(QColor('#66BB6A'))
                    elif val < -10:
                        it.setForeground(QColor('#EF5350'))
                    else:
                        it.setForeground(QColor('#888888'))
                else:
                    it.setForeground(QColor('#e0e0e0'))

                it.setToolTip(label)
                tbl.setItem(ri, c, it)

        tbl.setVerticalHeaderLabels(vlabels)
        tbl.resizeRowsToContents()
        lay.addWidget(tbl)

        # ── build plain-text for copy / export ───────────────────────
        def _build_text():
            lines = []
            lines.append(f'Component Loads — {lat_g:.2f}g lateral{lon_desc}')
            lines.append('=' * 72)
            lines.append(f'{"":40s}  {"FL":>7s}  {"FR":>7s}  {"RL":>7s}  {"RR":>7s}')
            lines.append('-' * 72)
            for label, attr, unit, dec in rows:
                if label == 'header':
                    lines.append('')
                    lines.append(attr)
                    continue
                if attr is None:
                    continue
                vals = []
                for corner in ('FL', 'FR', 'RL', 'RR'):
                    cl = loads.get(corner)
                    v = getattr(cl, attr, 0) if cl else 0
                    vals.append(f'{v + 0.0:.{dec}f}')
                tag = f'{label} ({unit})'
                lines.append(f'{tag:40s}  {vals[0]:>7s}  {vals[1]:>7s}  {vals[2]:>7s}  {vals[3]:>7s}')
            return '\n'.join(lines)

        def _build_csv():
            csv_lines = [f'# Component Loads — {lat_g:.2f}g lateral{lon_desc}']
            csv_lines.append('Parameter,Unit,FL,FR,RL,RR')
            for label, attr, unit, dec in rows:
                if label == 'header' or attr is None:
                    continue
                vals = []
                for corner in ('FL', 'FR', 'RL', 'RR'):
                    cl = loads.get(corner)
                    v = getattr(cl, attr, 0) if cl else 0
                    vals.append(f'{v + 0.0:.{dec}f}')
                csv_lines.append(f'{label},{unit},{",".join(vals)}')
            return '\n'.join(csv_lines)

        # ── buttons ──────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        _btn_style = ('QPushButton { background: #333; color: white; padding: 6px 16px; '
                      'border-radius: 3px; } QPushButton:hover { background: #555; }')
        _btn_green = ('QPushButton { background: #2E7D32; color: white; padding: 6px 16px; '
                      'border-radius: 3px; font-weight: bold; } '
                      'QPushButton:hover { background: #388E3C; }')

        copy_btn = QPushButton('Copy to Clipboard')
        copy_btn.setStyleSheet(_btn_green)
        def _copy():
            QApplication.clipboard().setText(_build_text())
            copy_btn.setText('Copied!')
            copy_btn.setStyleSheet(
                'QPushButton { background: #1B5E20; color: #A5D6A7; padding: 6px 16px; '
                'border-radius: 3px; font-weight: bold; }')
        copy_btn.clicked.connect(_copy)
        btn_row.addWidget(copy_btn)

        export_btn = QPushButton('Export CSV')
        export_btn.setStyleSheet(_btn_style)
        def _export():
            path, _ = QFileDialog.getSaveFileName(
                dlg, 'Export Loads', 'component_loads.csv', 'CSV (*.csv)')
            if path:
                with open(path, 'w') as f:
                    f.write(_build_csv())
                export_btn.setText(f'Saved!')
        export_btn.clicked.connect(_export)
        btn_row.addWidget(export_btn)

        btn_row.addStretch()
        close_btn = QPushButton('Close')
        close_btn.setStyleSheet(_btn_style)
        close_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(close_btn)
        lay.addLayout(btn_row)

        dlg.exec()


# ══════════════════════════════════════════════════════════════════════════════
#  BRAKE CALCULATOR PANEL
# ══════════════════════════════════════════════════════════════════════════════

class BrakeCalcPanel(CollapsibleSection):
    """
    Brake system calculator.

    Computes line pressure, brake torque, caliper clamping force,
    lockup pedal force, and single-event rotor temperature rise
    for all 4 corners.
    """
    compute_requested = pyqtSignal()

    def __init__(self):
        super().__init__('Brake Calculator', header_color='#D32F2F')
        self.set_info(section_info.BRAKE)
        self._build()

    def _build(self):
        # ── Operating point ──────────────────────────────────────────
        op_grid = QGridLayout(); op_grid.setSpacing(4)
        op_grid.addWidget(QLabel('Lateral g:'), 0, 0)
        self._lat_g = _spin(-5, 5, 1.0, ' g', dec=2, step=0.1)
        op_grid.addWidget(self._lat_g, 0, 1)
        op_grid.addWidget(QLabel('Lon g:'), 0, 2)
        self._lon_g = _spin(-5, 5, 0.0, ' g', dec=2, step=0.1)
        op_grid.addWidget(self._lon_g, 0, 3)
        self.add_layout(op_grid)

        # ── System-level params ──────────────────────────────────────
        sys_grid = QGridLayout(); sys_grid.setSpacing(4)
        lbl = QLabel('Brake System')
        lbl.setStyleSheet('font-weight: bold; color: #FFA726;')
        sys_grid.addWidget(lbl, 0, 0, 1, 4)

        r = 1
        sys_grid.addWidget(QLabel('Pedal ratio:'), r, 0)
        self._pedal_ratio = _spin(1.0, 20.0, 5.0, ':1', dec=1, step=0.25)
        self._pedal_ratio.setToolTip('Mechanical advantage of brake pedal lever arm')
        sys_grid.addWidget(self._pedal_ratio, r, 1)

        sys_grid.addWidget(QLabel('Bias (front):'), r, 2)
        self._bias_front = _spin(0, 100, 65, ' %', dec=0, step=1)
        self._bias_front.setToolTip('Brake bias bar setting: % of total force to front')
        sys_grid.addWidget(self._bias_front, r, 3)

        r += 1
        sys_grid.addWidget(QLabel('MC bore F:'), r, 0)
        self._mc_bore_f = _spin(5.0, 40.0, 15.87, ' mm', dec=2, step=0.5)
        self._mc_bore_f.setToolTip('Front master cylinder bore diameter (5/8" = 15.87mm)')
        sys_grid.addWidget(self._mc_bore_f, r, 1)

        sys_grid.addWidget(QLabel('MC bore R:'), r, 2)
        self._mc_bore_r = _spin(5.0, 40.0, 15.87, ' mm', dec=2, step=0.5)
        self._mc_bore_r.setToolTip('Rear master cylinder bore diameter')
        sys_grid.addWidget(self._mc_bore_r, r, 3)

        self.add_layout(sys_grid)

        # ── Rotor thermal ────────────────────────────────────────────
        th_grid = QGridLayout(); th_grid.setSpacing(4)
        th_lbl = QLabel('Rotor Thermal')
        th_lbl.setStyleSheet('font-weight: bold; color: #FFA726;')
        th_grid.addWidget(th_lbl, 0, 0, 1, 4)

        r = 1
        th_grid.addWidget(QLabel('Rotor mass F:'), r, 0)
        self._rotor_mass_f = _spin(0.1, 20.0, 1.5, ' kg', dec=2, step=0.1)
        th_grid.addWidget(self._rotor_mass_f, r, 1)
        th_grid.addWidget(QLabel('Rotor mass R:'), r, 2)
        self._rotor_mass_r = _spin(0.1, 20.0, 1.2, ' kg', dec=2, step=0.1)
        th_grid.addWidget(self._rotor_mass_r, r, 3)

        r += 1
        th_grid.addWidget(QLabel('Cp (rotor):'), r, 0)
        self._rotor_cp = _spin(100, 2000, 460, ' J/kg·K', dec=0, step=10)
        self._rotor_cp.setToolTip('Specific heat capacity of rotor material\n'
                                   'Cast iron ≈ 460 J/kg·K\n'
                                   'Carbon-carbon ≈ 710 J/kg·K')
        th_grid.addWidget(self._rotor_cp, r, 1)
        th_grid.addWidget(QLabel('Ambient:'), r, 2)
        self._ambient_temp = _spin(-20, 60, 25, ' °C', dec=0, step=5)
        th_grid.addWidget(self._ambient_temp, r, 3)

        r += 1
        th_grid.addWidget(QLabel('Brake from:'), r, 0)
        self._brake_speed = _spin(0, 200, 60, ' mph', dec=0, step=5)
        self._brake_speed.setToolTip('Initial speed for single-event thermal calc')
        th_grid.addWidget(self._brake_speed, r, 1)
        th_grid.addWidget(QLabel('to:'), r, 2)
        self._brake_speed_end = _spin(0, 200, 0, ' mph', dec=0, step=5)
        th_grid.addWidget(self._brake_speed_end, r, 3)

        self.add_layout(th_grid)

        # ── Compute button ───────────────────────────────────────────
        self._compute_btn = QPushButton('Compute Brakes')
        self._compute_btn.setStyleSheet(
            'QPushButton { background: #B71C1C; color: white; padding: 6px 16px; '
            'border-radius: 3px; font-weight: bold; }'
            'QPushButton:hover { background: #D32F2F; }')
        self._compute_btn.clicked.connect(lambda: self.compute_requested.emit())
        self.add_widget(self._compute_btn)

        # ── Status ───────────────────────────────────────────────────
        self._status = QLabel('')
        self._status.setStyleSheet('color: #888; font-size: 11px;')
        self.add_widget(self._status)

    def get_system_params(self, tire_radius_m: float = 0.203):
        from vahan.loads import BrakeSystemParams
        return BrakeSystemParams(
            pedal_ratio=self._pedal_ratio.value(),
            mc_bore_front_mm=self._mc_bore_f.value(),
            mc_bore_rear_mm=self._mc_bore_r.value(),
            bias_pct_front=self._bias_front.value(),
            tire_radius_m=tire_radius_m,
        )

    def get_thermal_params(self) -> dict:
        return {
            'rotor_mass_f_kg': self._rotor_mass_f.value(),
            'rotor_mass_r_kg': self._rotor_mass_r.value(),
            'rotor_cp':        self._rotor_cp.value(),
            'ambient_C':       self._ambient_temp.value(),
            'speed_start_mph': self._brake_speed.value(),
            'speed_end_mph':   self._brake_speed_end.value(),
        }

    # ── Save / load ─────────────────────────────────────────────────
    def get_state(self) -> dict:
        return {
            'lat_g':          self._lat_g.value(),
            'lon_g':          self._lon_g.value(),
            'pedal_ratio':    self._pedal_ratio.value(),
            'mc_bore_f_mm':   self._mc_bore_f.value(),
            'mc_bore_r_mm':   self._mc_bore_r.value(),
            'bias_front_pct': self._bias_front.value(),
            'rotor_mass_f_kg': self._rotor_mass_f.value(),
            'rotor_mass_r_kg': self._rotor_mass_r.value(),
            'rotor_cp':        self._rotor_cp.value(),
            'ambient_C':       self._ambient_temp.value(),
            'speed_start_mph': self._brake_speed.value(),
            'speed_end_mph':   self._brake_speed_end.value(),
        }

    def set_state(self, d: dict) -> None:
        if not isinstance(d, dict):
            return
        _map = {
            'lat_g':          self._lat_g,
            'lon_g':          self._lon_g,
            'pedal_ratio':    self._pedal_ratio,
            'mc_bore_f_mm':   self._mc_bore_f,
            'mc_bore_r_mm':   self._mc_bore_r,
            'bias_front_pct': self._bias_front,
            'rotor_mass_f_kg': self._rotor_mass_f,
            'rotor_mass_r_kg': self._rotor_mass_r,
            'rotor_cp':        self._rotor_cp,
            'ambient_C':       self._ambient_temp,
            'speed_start_mph': self._brake_speed,
            'speed_end_mph':   self._brake_speed_end,
        }
        for key, sb in _map.items():
            if key in d:
                try:
                    sb.blockSignals(True)
                    sb.setValue(float(d[key]))
                finally:
                    sb.blockSignals(False)

    # ── Results display ─────────────────────────────────────────────
    def show_results(self, results: dict, Fz: dict,
                     lat_g: float = 0.0, lon_g: float = 0.0,
                     thermal: dict = None):
        """Display brake calc results in a popup dialog.

        thermal : dict, optional
            Per-corner thermal results from compute_brake_thermal().
            Keys: 'FL'...'RR' → dict with 'energy_J', 'delta_T_C', 'peak_T_C'.
        """
        self._status.setText(
            f'Computed at {lat_g:.2f}g lateral, {lon_g:.2f}g longitudinal')

        rows = [
            ('header', 'VERTICAL LOADS', None, None),
            ('Fz',                    'Fz_N',                    'N',   0),
            ('Tire μ (from TTC)',     'tire_mu',                 '',    2),
            ('', None, None, None),

            ('header', 'LOCKUP THRESHOLD', None, None),
            ('Lockup Fx',             'lockup_Fx_N',             'N',   0),
            ('Lockup brake torque',   'lockup_torque_Nm',        'Nm',  1),
            ('Lockup clamp force',    'lockup_clamp_N',          'N',   0),
            ('Lockup line pressure',  'lockup_line_pressure_MPa','MPa', 2),
            ('Lockup pedal force',    'lockup_pedal_force_N',    'N',   1),
        ]

        # Append thermal rows if available
        if thermal:
            rows.append(('', None, None, None))
            rows.append(('header', 'ROTOR THERMAL  (single braking event, adiabatic)', None, None))
            rows.append(('Energy absorbed',  '_thermal_energy_kJ',  'kJ',  2))
            rows.append(('Temperature rise', '_thermal_delta_T_C',  '°C',  0))
            rows.append(('Peak rotor temp',  '_thermal_peak_T_C',   '°C',  0))

        dlg = QDialog(self)
        dlg.setWindowTitle('Brake Calculator')
        dlg.setMinimumSize(700, 500 if thermal else 420)
        dlg.setStyleSheet('''
            QDialog { background: #000; color: #e0e0e0; }
            QLabel  { color: #e0e0e0; }
            QTableWidget { background: #0a0a0a; color: #e0e0e0;
                           gridline-color: #2a2a2a; border: none; font-size: 12px; }
            QHeaderView::section { background: #111; color: #ccc;
                                   border: 1px solid #2a2a2a; padding: 4px;
                                   font-weight: bold; font-size: 12px; }
        ''')
        lay = QVBoxLayout(dlg)

        # ── Header ───────────────────────────────────────────────────
        lon_desc = ''
        if lon_g < -0.01:
            lon_desc = f', {abs(lon_g):.2f}g braking'
        elif lon_g > 0.01:
            lon_desc = f', {lon_g:.2f}g acceleration'
        cond = QLabel(f'Operating point:  {lat_g:.2f}g lateral{lon_desc}')
        cond.setStyleSheet(
            'color: #FFA726; font-size: 14px; font-weight: bold; padding: 6px;')
        lay.addWidget(cond)

        # Show system summary
        sp = self.get_system_params()
        import math
        mc_f_area = sp.mc_area_front_mm2
        mc_r_area = sp.mc_area_rear_mm2
        sys_lbl = QLabel(
            f'Pedal ratio {sp.pedal_ratio:.1f}:1  |  '
            f'MC bore F/R {sp.mc_bore_front_mm:.1f}/{sp.mc_bore_rear_mm:.1f} mm  '
            f'({mc_f_area:.0f}/{mc_r_area:.0f} mm²)  |  '
            f'Bias {sp.bias_pct_front:.0f}% front  |  '
            f'Tire R {sp.tire_radius_m*1000:.0f} mm  |  '
            f'Tire μ from TTC (per corner)')
        sys_lbl.setStyleSheet('color: #aaa; font-size: 11px; padding: 2px 6px;')
        sys_lbl.setWordWrap(True)
        lay.addWidget(sys_lbl)

        # ── Table ────────────────────────────────────────────────────
        tbl = QTableWidget(len(rows), 4)
        tbl.setHorizontalHeaderLabels(['FL', 'FR', 'RL', 'RR'])
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        tbl.setSelectionMode(QTableWidget.SelectionMode.NoSelection)

        vlabels = []
        for ri, (label_txt, attr, unit, dec) in enumerate(rows):
            if label_txt == 'header':
                vlabels.append('')
                for c in range(4):
                    it = QTableWidgetItem(attr if c == 0 else '')
                    if c == 0:
                        it.setForeground(QColor('#FFA726'))
                        f = it.font(); f.setBold(True); it.setFont(f)
                    it.setBackground(QColor('#111111'))
                    tbl.setItem(ri, c, it)
                tbl.setSpan(ri, 0, 1, 4)
                continue

            if attr is None:
                vlabels.append('')
                for c in range(4):
                    it = QTableWidgetItem('')
                    it.setBackground(QColor('#0a0a0a'))
                    tbl.setItem(ri, c, it)
                continue

            vlabels.append(f'{label_txt} ({unit})' if unit else label_txt)
            for c, corner in enumerate(['FL', 'FR', 'RL', 'RR']):
                # Thermal pseudo-attrs start with '_thermal_'
                if attr.startswith('_thermal_') and thermal:
                    th = thermal.get(corner, {})
                    th_key = attr.replace('_thermal_', '')
                    val = th.get(th_key, 0)
                else:
                    cr = results.get(corner)
                    val = getattr(cr, attr, 0) if cr else 0
                val = val + 0.0
                txt = f'{val:.{dec}f}'
                it = QTableWidgetItem(txt)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                # Color thermal peak temp by severity
                if attr == '_thermal_peak_T_C' and val > 0:
                    if val > 600:
                        it.setForeground(QColor('#EF5350'))   # red — too hot
                    elif val > 400:
                        it.setForeground(QColor('#FFA726'))   # orange — warm
                    else:
                        it.setForeground(QColor('#66BB6A'))   # green — fine
                else:
                    it.setForeground(QColor('#e0e0e0'))
                tbl.setItem(ri, c, it)

        tbl.setVerticalHeaderLabels(vlabels)
        tbl.resizeRowsToContents()
        lay.addWidget(tbl)

        # ── Lockup summary ───────────────────────────────────────────
        # Which corner locks first? = lowest lockup pedal force
        lockup_forces = {}
        for corner in ['FL', 'FR', 'RL', 'RR']:
            cr = results.get(corner)
            if cr and cr.lockup_pedal_force_N > 0:
                lockup_forces[corner] = cr.lockup_pedal_force_N

        if lockup_forces:
            first = min(lockup_forces, key=lockup_forces.get)
            first_force = lockup_forces[first]
            first_lbl = QLabel(
                f'⚠  First lockup: {first} at {first_force:.0f} N pedal force  '
                f'({first_force / 4.448:.0f} lbf)')
            axle = 'Front' if first[0] == 'F' else 'Rear'
            first_lbl.setStyleSheet(
                'color: #EF5350; font-size: 13px; font-weight: bold; padding: 6px;')
            lay.addWidget(first_lbl)

            # Show front vs rear lockup balance
            front_avg = (lockup_forces.get('FL', 0) + lockup_forces.get('FR', 0)) / 2
            rear_avg = (lockup_forces.get('RL', 0) + lockup_forces.get('RR', 0)) / 2
            if front_avg > 0 and rear_avg > 0:
                if front_avg < rear_avg:
                    balance = f'Fronts lock first ({front_avg:.0f} vs {rear_avg:.0f} N)'
                    color = '#FFA726'
                else:
                    balance = f'Rears lock first ({rear_avg:.0f} vs {front_avg:.0f} N)'
                    color = '#EF5350'
                bal_lbl = QLabel(balance)
                bal_lbl.setStyleSheet(
                    f'color: {color}; font-size: 12px; padding: 2px 6px;')
                lay.addWidget(bal_lbl)

        # ── Thermal summary ───────────────────────────────────────────
        if thermal:
            peak_temps = {c: thermal[c]['peak_T_C'] for c in ['FL', 'FR', 'RL', 'RR']}
            hottest = max(peak_temps, key=peak_temps.get)
            hottest_T = peak_temps[hottest]
            tp = self.get_thermal_params()
            speed_str = (f'{tp["speed_start_mph"]:.0f} → '
                         f'{tp["speed_end_mph"]:.0f} mph')

            if hottest_T > 600:
                temp_color = '#EF5350'
                verdict = 'EXCEEDS SAFE LIMIT'
            elif hottest_T > 400:
                temp_color = '#FFA726'
                verdict = 'warm but acceptable'
            else:
                temp_color = '#66BB6A'
                verdict = 'within safe range'

            th_lbl = QLabel(
                f'Hottest rotor: {hottest} at {hottest_T:.0f}°C  '
                f'({verdict})  —  {speed_str} adiabatic stop')
            th_lbl.setStyleSheet(
                f'color: {temp_color}; font-size: 12px; font-weight: bold; padding: 4px 6px;')
            lay.addWidget(th_lbl)

        # ── Buttons ──────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        _btn_style = ('QPushButton { background: #333; color: white; padding: 6px 16px; '
                      'border-radius: 3px; } QPushButton:hover { background: #555; }')

        close_btn = QPushButton('Close')
        close_btn.setStyleSheet(_btn_style)
        close_btn.clicked.connect(dlg.accept)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        lay.addLayout(btn_row)

        dlg.exec()


# ══════════════════════════════════════════════════════════════════════════════
#  DYNAMICS OPTIMIZER PANEL
# ══════════════════════════════════════════════════════════════════════════════

# Metrics the user can target
_OPT_METRICS = [
    ('understeer_gradient_deg', 'Understeer gradient', '°'),
    ('roll_angle_deg',          'Roll angle',          '°'),
    ('pitch_angle_deg',         'Pitch angle',         '°'),
    ('lltd_pct',                'LLTD (TOTAL LT front)', '%'),
    ('utilization_max',         'Max utilization',     ''),
    ('utilization_spread',      'Util. spread',        ''),
    ('ideal_ackermann_pct',     'Ideal Ackermann',     '%'),
]


class DynamicsOptPanel(CollapsibleSection):
    """
    Dynamics sensitivity analyzer / optimizer.

    User sets operating g-levels, picks a metric to improve, sets a target,
    and the panel shows every way to get there — parameter changes AND
    kinematic changes — ranked by effectiveness with side effects.
    """
    analyze_requested = pyqtSignal(dict)  # {lateral_g, longitudinal_g}

    def __init__(self):
        super().__init__('Dynamics Optimizer', header_color='#FFA726')
        self.set_info(section_info.DYNAMICS_OPT)
        self._analysis = None
        self._build()

    def _build(self):
        # ── Operating point ──────────────────────────────────────────
        op_grid = QGridLayout(); op_grid.setSpacing(4)
        op_grid.addWidget(QLabel('Analyze at:'), 0, 0)
        self._opt_lat_g = _spin(-99, 99, 1.2, ' g lat', dec=2, step=0.1)
        op_grid.addWidget(self._opt_lat_g, 0, 1)
        self._opt_lon_g = _spin(-99, 99, 0.0, ' g lon', dec=2, step=0.1)
        op_grid.addWidget(self._opt_lon_g, 0, 2)
        op_grid.addWidget(QLabel('Turn radius:'), 1, 0)
        self._opt_turn_radius = _spin(1.0, 200.0, 7.5, ' m', dec=1, step=0.5)
        self._opt_turn_radius.setToolTip(
            'Turn radius for ideal Ackermann computation.\n'
            'FSAE typical: 4–10 m skidpad, 6–15 m endurance.')
        op_grid.addWidget(self._opt_turn_radius, 1, 1)
        self.add_layout(op_grid)

        # ── Analyze button ───────────────────────────────────────────
        self._analyze_btn = QPushButton('Analyze Sensitivities')
        self._analyze_btn.setStyleSheet(
            'QPushButton { background: #7B3F00; color: white; padding: 6px 16px; '
            'border-radius: 3px; font-weight: bold; }'
            'QPushButton:hover { background: #A0522D; }')
        self._analyze_btn.clicked.connect(self._on_analyze)
        self.add_widget(self._analyze_btn)

        # ── Status ───────────────────────────────────────────────────
        self._opt_status = QLabel('')
        self._opt_status.setStyleSheet('color: #888; font-size: 11px;')
        self._opt_status.setWordWrap(True)
        self.add_widget(self._opt_status)

        # ── Baseline readout ─────────────────────────────────────────
        self._baseline_label = QLabel('')
        self._baseline_label.setStyleSheet('color: #FFA726; font-size: 11px;')
        self._baseline_label.setWordWrap(True)
        self.add_widget(self._baseline_label)

        # ── Target selection ─────────────────────────────────────────
        tgt_grid = QGridLayout(); tgt_grid.setSpacing(4)
        tgt_grid.addWidget(QLabel('Target metric:'), 0, 0)
        self._target_combo = QComboBox()
        for key, label, unit in _OPT_METRICS:
            suffix = f' ({unit})' if unit else ''
            self._target_combo.addItem(f'{label}{suffix}', key)
        self._target_combo.setStyleSheet(
            'QComboBox { background: #1a1a1a; color: #e0e0e0; border: 1px solid #333; '
            'border-radius: 3px; padding: 2px 6px; }'
            'QComboBox QAbstractItemView { background: #1a1a1a; color: #e0e0e0; '
            'selection-background-color: #7B3F00; }')
        self._target_combo.currentIndexChanged.connect(self._on_target_changed)
        tgt_grid.addWidget(self._target_combo, 0, 1)

        tgt_grid.addWidget(QLabel('Change by:'), 1, 0)
        self._target_delta = _spin(-50, 50, -1.0, '', dec=2, step=0.5)
        self._target_delta.valueChanged.connect(self._on_target_changed)
        tgt_grid.addWidget(self._target_delta, 1, 1)

        # Direction label — shows exactly what the target means
        self._target_dir_label = QLabel('')
        self._target_dir_label.setStyleSheet('color: #FFA726; font-size: 11px; font-weight: bold;')
        self._target_dir_label.setWordWrap(True)
        tgt_grid.addWidget(self._target_dir_label, 2, 0, 1, 2)
        self.add_layout(tgt_grid)

        # ── Recommendation button ────────────────────────────────────
        self._recommend_btn = QPushButton('Show Recommendations')
        self._recommend_btn.setStyleSheet(
            'QPushButton { background: #1a5276; color: white; padding: 6px 16px; '
            'border-radius: 3px; font-weight: bold; }'
            'QPushButton:hover { background: #1f6da0; }')
        self._recommend_btn.clicked.connect(self._on_recommend)
        self._recommend_btn.setEnabled(False)
        self.add_widget(self._recommend_btn)

        # ── Sensitivity table ────────────────────────────────────────
        self._sens_table = QTableWidget(0, 5)
        self._sens_table.setHorizontalHeaderLabels([
            'Knob', 'Change', 'Side Effects', 'Category', 'How',
        ])
        self._sens_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self._sens_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Fixed)
        self._sens_table.horizontalHeader().resizeSection(1, 180)
        self._sens_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch)
        self._sens_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Fixed)
        self._sens_table.horizontalHeader().resizeSection(3, 70)
        self._sens_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.Stretch)
        self._sens_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._sens_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._sens_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._sens_table.setMinimumHeight(200)
        self._sens_table.setMaximumHeight(400)
        self.add_widget(self._sens_table)

        # ── Full sensitivity grid (all knobs x all metrics) ──────────
        self._full_sens_btn = QPushButton('Show Full Sensitivity Grid')
        self._full_sens_btn.setStyleSheet(
            'QPushButton { background: #333; color: #aaa; padding: 4px 12px; '
            'border-radius: 3px; font-size: 11px; }'
            'QPushButton:hover { background: #444; color: #ddd; }')
        self._full_sens_btn.clicked.connect(self._show_full_grid)
        self._full_sens_btn.setEnabled(False)
        self.add_widget(self._full_sens_btn)

    def _on_analyze(self):
        self.analyze_requested.emit({
            'lateral_g': self._opt_lat_g.value(),
            'longitudinal_g': self._opt_lon_g.value(),
            'turn_radius_m': self._opt_turn_radius.value(),
        })
        self._opt_status.setText('Analyzing...')
        self._analyze_btn.setEnabled(False)

    def show_analysis(self, analysis: dict):
        """Called by main_window after sensitivity analysis completes."""
        self._analysis = analysis
        self._analyze_btn.setEnabled(True)
        self._recommend_btn.setEnabled(True)
        self._full_sens_btn.setEnabled(True)

        bl = analysis['baseline']
        parts = []
        import math
        for key, label, unit in _OPT_METRICS:
            val = bl.get(key, 0)
            if math.isnan(val):
                parts.append(f'{label}: N/A')
            else:
                parts.append(f'{label}: {val:.2f}{unit}')
        self._baseline_label.setText('Baseline:  ' + '  |  '.join(parts))
        self._opt_status.setText(
            f'Done — {len(analysis["sensitivities"])} knobs analyzed')

        # Auto-show recommendations for current target
        self._update_target_dir_label()
        self._on_recommend()

    def _on_target_changed(self):
        self._update_target_dir_label()
        if self._analysis is not None:
            self._on_recommend()

    def _update_target_dir_label(self):
        delta = self._target_delta.value()
        key = self._target_combo.currentData()
        label = key
        unit = ''
        for k, l, u in _OPT_METRICS:
            if k == key:
                label = l; unit = u; break
        if self._analysis is not None:
            bl = self._analysis['baseline'].get(key, 0)
            if math.isnan(bl):
                self._target_dir_label.setText(
                    f'{label}: N/A (no tire model or turn radius)')
                return
            predicted = bl + delta
            self._target_dir_label.setText(
                f'Target: {label} {bl:.2f}{unit} -> {predicted:.2f}{unit}')
        elif abs(delta) > 1e-6:
            if delta > 0:
                self._target_dir_label.setText(f'Goal: INCREASE {label} by {delta:+.2f}{unit}')
            else:
                self._target_dir_label.setText(f'Goal: DECREASE {label} by {delta:.2f}{unit}')

    def _on_recommend(self):
        if self._analysis is None:
            return

        try:
            self._on_recommend_impl()
        except Exception:
            import traceback; traceback.print_exc()
            self._opt_status.setText('Recommendation error — see console')

    def _on_recommend_impl(self):
        from vahan.dynamics import DynamicsSensitivity
        target_key = self._target_combo.currentData()
        target_delta = self._target_delta.value()
        baseline = self._analysis['baseline']

        if abs(target_delta) < 1e-6:
            self._sens_table.setRowCount(0)
            return

        # Build recommendations
        sens = DynamicsSensitivity.__new__(DynamicsSensitivity)
        recs = sens.recommend(self._analysis, target_key, target_delta)

        if not recs:
            self._sens_table.setRowCount(1)
            item = QTableWidgetItem('No knobs affect this metric')
            item.setForeground(QColor('#888'))
            self._sens_table.setItem(0, 0, item)
            for c in range(1, 5):
                self._sens_table.setItem(0, c, QTableWidgetItem(''))
            return

        # Find target label/unit for display
        target_label = target_key
        target_unit = ''
        for k, l, u in _OPT_METRICS:
            if k == target_key:
                target_label = l
                target_unit = u
                break

        baseline_val = baseline.get(target_key, 0)
        predicted_val = baseline_val + target_delta

        self._sens_table.setRowCount(len(recs))
        for i, rec in enumerate(recs):
            # Knob name
            item = QTableWidgetItem(rec['knob'])
            cat_color = '#66BB6A' if rec['category'] == 'parameter' else '#4FC3F7'
            item.setForeground(QColor(cat_color))
            self._sens_table.setItem(i, 0, item)

            # Change needed + predicted target value
            change = rec['change_needed']
            unit = rec['unit']
            new_val = rec['new_value']
            if unit:
                change_str = f'{change:+.1f} {unit}'
                change_str += f'\n  {rec["current"]:.0f} -> {new_val:.0f} {unit}'
            else:
                change_str = f'{change:+.4f}'
                change_str += f'\n  {rec["current"]:.3f} -> {new_val:.3f}'
            # Show predicted target metric value
            change_str += f'\n{target_label}: {baseline_val:.2f} -> {predicted_val:.2f}{target_unit}'

            # Flag infeasible values
            feasible = True
            key = rec['key']
            if 'spring_rate' in key and new_val < 0:
                change_str += '  [infeasible]'
                feasible = False
            elif 'arb_rate' in key and new_val < 0:
                change_str += '  [infeasible]'
                feasible = False
            elif 'motion_ratio' in key and (new_val < 0.3 or new_val > 3.0):
                change_str += '  [infeasible]'
                feasible = False
            elif 'brake_bias' in key and (new_val < 30 or new_val > 90):
                change_str += '  [infeasible]'
                feasible = False
            elif 'cg_to_front' in key and (new_val < 500 or new_val > 2000):
                change_str += '  [infeasible]'
                feasible = False

            item = QTableWidgetItem(change_str)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if not feasible:
                item.setForeground(QColor('#555555'))
            self._sens_table.setItem(i, 1, item)

            # Side effects — show as absolute values (baseline -> predicted)
            se_parts = []
            for metric, delta in rec['side_effects'].items():
                if math.isnan(delta) or abs(delta) < 0.001:
                    continue
                for k, l, u in _OPT_METRICS:
                    if k == metric:
                        bl = baseline.get(metric, 0)
                        if math.isnan(bl):
                            continue
                        se_parts.append(f'{l}: {bl:.2f} -> {bl + delta:.2f}{u}')
                        break
            item = QTableWidgetItem('\n'.join(se_parts) if se_parts else 'None')
            item.setForeground(QColor('#999999'))
            self._sens_table.setItem(i, 2, item)

            # Category
            cat_str = 'Bolt-on' if rec['category'] == 'parameter' else 'Geometry'
            item = QTableWidgetItem(cat_str)
            item.setForeground(QColor(cat_color))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._sens_table.setItem(i, 3, item)

            # Implementation hints
            item = QTableWidgetItem('\n'.join(rec['implementations'][:3]))
            item.setForeground(QColor('#888888'))
            self._sens_table.setItem(i, 4, item)

        self._sens_table.resizeRowsToContents()

    def _show_full_grid(self):
        """Pop out the recommendation table into a larger dialog."""
        if self._analysis is None:
            return

        dlg = QDialog(self)
        dlg.setWindowTitle('Recommendations')
        dlg.resize(1100, 500)
        dlg.setStyleSheet(
            'QDialog { background: #0a0a0a; }'
            'QTableWidget { background: #0a0a0a; color: #e0e0e0; '
            'gridline-color: #222; font-size: 12px; }')
        lay = QVBoxLayout(dlg)

        # Copy baseline + target info
        lay.addWidget(QLabel(self._baseline_label.text()))
        lay.lastWidget = lay.itemAt(0).widget()
        lay.lastWidget.setStyleSheet('color: #FFA726; font-size: 12px;')
        lay.lastWidget.setWordWrap(True)

        dir_txt = self._target_dir_label.text()
        if dir_txt:
            dir_lbl = QLabel(dir_txt)
            dir_lbl.setStyleSheet('color: #FFA726; font-size: 12px; font-weight: bold;')
            lay.addWidget(dir_lbl)

        # Clone the recommendation table content into a bigger table
        src = self._sens_table
        n_rows = src.rowCount()
        n_cols = src.columnCount()
        tbl = QTableWidget(n_rows, n_cols)
        tbl.setHorizontalHeaderLabels([
            src.horizontalHeaderItem(c).text() for c in range(n_cols)])
        tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        tbl.horizontalHeader().resizeSection(1, 210)
        tbl.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        tbl.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        tbl.horizontalHeader().resizeSection(3, 75)
        tbl.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        tbl.verticalHeader().setVisible(False)

        for r in range(n_rows):
            for c in range(n_cols):
                src_item = src.item(r, c)
                if src_item:
                    item = QTableWidgetItem(src_item.text())
                    item.setForeground(src_item.foreground())
                    item.setTextAlignment(src_item.textAlignment())
                    tbl.setItem(r, c, item)

        tbl.resizeRowsToContents()
        lay.addWidget(tbl)

        close_btn = QPushButton('Close')
        close_btn.clicked.connect(dlg.accept)
        close_btn.setStyleSheet(
            'QPushButton { background: #1a5276; color: white; padding: 6px 20px; '
            'border-radius: 3px; font-weight: bold; }')
        lay.addWidget(close_btn)
        dlg.exec()


# ══════════════════════════════════════════════════════════════════════════════
#  SKIDPAD / TRANSIENT PANEL
# ══════════════════════════════════════════════════════════════════════════════

_TRANSIENT_TESTS = [
    ('skidpad',       'FSAE Skidpad (single circle)'),
    ('skidpad_full',  'FSAE Skidpad (full figure-8)'),
    ('step',          'Step steer'),
    ('ramp',          'Ramp steer'),
    ('sine',          'Sine (freq sweep)'),
]

_TRANSIENT_SIGNALS = [
    ('yaw_rate',    'Yaw rate (deg/s)'),
    ('ay',          'Lateral g'),
    ('roll',        'Roll angle (deg)'),
    ('beta',        'Body slip (deg)'),
    ('velocity',    'Velocity (m/s)'),
    ('slip_angle',  'Tire slip angles'),
    ('Fz',          'Per-corner Fz'),
    ('path',        'Trajectory (X-Y)'),
    ('steer',       'Steering input'),
]

# Solve-mode toggle for closed-loop skidpad tests.  Either the driver
# picks the speed (and the resulting lateral-g falls out of the sim), or
# they pick the target lateral-g (and the speed is derived from the
# steady-state relation v² = a_y · g · R).  Mutually exclusive — cannot
# "solve for both" because on a fixed-radius path they are linked.
_TRANSIENT_SOLVE_MODES = [
    ('target_speed', 'Target speed'),
    ('target_lat_g', 'Target lateral g'),
]


class SkidpadPanel(CollapsibleSection):
    """
    Time-domain transient dynamics — skidpad, step-steer, ramp-steer.

    Emits simulate_requested with a dict of all simulation parameters.
    Main window builds the TransientSolver, runs it in a worker,
    and routes results back via show_result().
    """
    simulate_requested = pyqtSignal(dict)
    signals_changed    = pyqtSignal(list)    # selected plot signals

    def __init__(self):
        super().__init__('Skidpad / Transient', header_color='#FFB74D')
        self.set_info(section_info.SKIDPAD)
        self._build()

    def _build(self):
        # ── Test type ───────────────────────────────────────────────────
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel('Test:'))
        self._test_combo = QComboBox()
        for key, label in _TRANSIENT_TESTS:
            self._test_combo.addItem(label, key)
        self._test_combo.setStyleSheet(
            'QComboBox { background: #1a1a1a; color: #e0e0e0; '
            'border: 1px solid #333; border-radius: 3px; padding: 2px 6px; }'
            'QComboBox QAbstractItemView { background: #1a1a1a; color: #e0e0e0; '
            'selection-background-color: #7B3F00; }')
        self._test_combo.currentIndexChanged.connect(self._on_test_changed)
        type_row.addWidget(self._test_combo, 1)
        self.add_layout(type_row)

        # ── Solve mode toggle ────────────────────────────────────────────
        # For closed-loop skidpad tests, the driver can either specify a
        # target speed (and read off the resulting lateral-g) OR specify
        # a target lateral-g (and have the speed derived from the
        # steady-state relation v = √(a_y · g · R)).  These are mutually
        # exclusive because on a fixed-radius path the two are linked —
        # picking one pins the other.
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel('Solve for:'))
        self._solve_mode_combo = QComboBox()
        for key, label in _TRANSIENT_SOLVE_MODES:
            self._solve_mode_combo.addItem(label, key)
        self._solve_mode_combo.setStyleSheet(
            'QComboBox { background: #1a1a1a; color: #e0e0e0; '
            'border: 1px solid #333; border-radius: 3px; padding: 2px 6px; }'
            'QComboBox QAbstractItemView { background: #1a1a1a; color: #e0e0e0; '
            'selection-background-color: #7B3F00; }')
        self._solve_mode_combo.currentIndexChanged.connect(self._on_solve_mode_changed)
        mode_row.addWidget(self._solve_mode_combo, 1)
        self.add_layout(mode_row)

        # ── Driving conditions ───────────────────────────────────────────
        # We keep a {key: (QLabel, QSpinBox)} map so redundant rows can be
        # hidden based on test type (FSAE skidpad fixes radius at 9.125 m,
        # derives peak steer from δ = L / R, sim duration from lap time).
        self._rows: dict = {}
        g = QGridLayout(); g.setSpacing(4)
        r = 0
        def row(key, label_txt, lo, hi, val, suf, r, dec=1, step=1.0):
            lbl = QLabel(label_txt)
            g.addWidget(lbl, r, 0)
            sb = _spin(lo, hi, val, suf, dec=dec, step=step)
            g.addWidget(sb, r, 1)
            self._rows[key] = (lbl, sb)
            return sb

        # Target speed in MPH — internally converted to m/s in get_params().
        # Default 23.3 mph ≈ 37.4 km/h ≈ typical FSAE skidpad entry speed.
        self._target_speed   = row('speed',   'Target speed:',  0.1, 1e6, 23.3, ' mph',  r, dec=1, step=1); r += 1
        # Target lateral-g: only used when the mode toggle is on "Target
        # lateral g".  Default 1.50 g ≈ typical FSAE skidpad cornering
        # limit on slicks.  Speed then falls out as v = √(a_y · g · R).
        self._target_lat_g   = row('lat_g',   'Target lat g:',  0.01, 5.0, 1.50, ' g',   r, dec=2, step=0.05); r += 1
        self._skidpad_radius = row('radius',  'Skidpad radius:', 0.01, 1e6, 9.125, ' m', r, dec=3, step=0.5); r += 1
        self._steer_deg      = row('steer',   'Peak steer:',    -1e4, 1e4, 12.0, ' deg', r, dec=1, step=1); r += 1
        self._ramp_duration  = row('ramp',    'Ramp duration:', 0.001, 1e6, 0.5, ' s',   r, dec=2, step=0.1); r += 1
        self._sim_duration   = row('sim',     'Sim duration:',  0.01, 1e6, 5.0, ' s',   r, dec=1, step=0.5); r += 1
        self._dt_ms          = row('dt',      'Time step:',     0.001, 1e6, 2.0, ' ms', r, dec=1, step=0.5); r += 1
        self.add_layout(g)

        # Direction (left/right) — only meaningful for skidpad, but useful elsewhere
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel('Direction:'))
        self._dir_combo = QComboBox()
        self._dir_combo.addItems(['Left', 'Right'])
        self._dir_combo.setStyleSheet(
            'QComboBox { background: #1a1a1a; color: #e0e0e0; '
            'border: 1px solid #333; border-radius: 3px; padding: 2px 6px; }'
            'QComboBox QAbstractItemView { background: #1a1a1a; color: #e0e0e0; '
            'selection-background-color: #7B3F00; }')
        self._dir_combo.setMaximumWidth(100)
        dir_row.addWidget(self._dir_combo)
        dir_row.addStretch()
        self.add_layout(dir_row)

        # ── Transient-specific params ────────────────────────────────────
        # Inertias (Ixx, Izz) and Ackermann % are auto-computed from the
        # car + hardpoint geometry in the dispatch.  Body roll damping is
        # **derived** from these damper bump/rebound coefficients via
        #     c_phi = Σ_axle (c_bump+c_rebound)·MR²·t²/4
        # using the motion-ratio and track-width fields already in the
        # car spec.  The derived c_phi is echoed back below in the
        # _auto_info label.  Defaults below are placeholders only — set
        # them from the actual damper dyno curve (e.g. low-frequency
        # bump/rebound slopes from your "Springs, Dampers, Anti-roll"
        # workbook).
        gg = QGroupBox('Damper rates & steer lag')
        gg.setStyleSheet(
            'QGroupBox { color: #FFB74D; font-size: 11px; border: 1px solid #333; '
            'border-radius: 3px; margin-top: 6px; padding: 6px; }'
            'QGroupBox::title { left: 6px; padding: 0 4px; }')
        gl = QGridLayout(gg); gl.setSpacing(4)
        def grow(label, lo, hi, val, suf, r, dec=1, step=1.0):
            gl.addWidget(QLabel(label), r, 0)
            sb = _spin(lo, hi, val, suf, dec=dec, step=step)
            gl.addWidget(sb, r, 1)
            return sb
        r = 0
        # Damper coefficients are at the SHOCK (force/velocity slope at
        # the damper body, not at the wheel).  Conversion to wheel rate
        # uses the kinematic motion ratio from the car spec.
        self._dmp_F_bump = grow('Damper F bump:',    0, 1e6, 1305.0, ' N·s/m', r, dec=0, step=25); r += 1
        self._dmp_F_reb  = grow('Damper F rebound:', 0, 1e6, 2936.0, ' N·s/m', r, dec=0, step=25); r += 1
        self._dmp_R_bump = grow('Damper R bump:',    0, 1e6, 1483.0, ' N·s/m', r, dec=0, step=25); r += 1
        self._dmp_R_reb  = grow('Damper R rebound:', 0, 1e6, 3338.0, ' N·s/m', r, dec=0, step=25); r += 1
        self._steer_tau  = grow('Steer lag:',        0, 1e6,    0.02, ' s',     r, dec=3, step=0.005); r += 1
        self._steer_tau.setToolTip(
            'First-order lag between commanded and actual steer angle.\n'
            '• 0.01–0.02 s for closed-loop path-following (Stanley) — '
            'fast actuator, controller is stable.\n'
            '• 0.10–0.25 s only for human-driven open-loop tests — '
            'will destabilise closed-loop controllers.\n'
            '• 0 is invalid (ODE goes singular); the solver clamps '
            'silently to 1 ms.')
        self.add_widget(gg)

        # Auto-computed readout — populated by main_window before each sim.
        self._auto_info = QLabel(
            'Izz, Ixx, Ackermann — auto-computed from car + kinematics at run time.')
        self._auto_info.setStyleSheet(
            'color: #888; font-size: 10px; font-style: italic; '
            'background: #0a0a0a; padding: 4px 6px; border: 1px solid #1a1a1a; '
            'border-radius: 3px;')
        self._auto_info.setWordWrap(True)
        self.add_widget(self._auto_info)

        # ── Simulate button ──────────────────────────────────────────────
        self._sim_btn = QPushButton('Simulate')
        self._sim_btn.setStyleSheet(
            'QPushButton { background: #7B3F00; color: white; padding: 6px 16px; '
            'border-radius: 3px; font-weight: bold; }'
            'QPushButton:hover { background: #A0522D; }'
            'QPushButton:disabled { background: #333; color: #666; }')
        self._sim_btn.clicked.connect(self._on_simulate)
        self.add_widget(self._sim_btn)

        # ── Status ────────────────────────────────────────────────────────
        self._status = QLabel('')
        self._status.setStyleSheet(f'color: {C_SUB}; font-size: 11px;')
        self._status.setWordWrap(True)
        self.add_widget(self._status)

        # ── Results readout ──────────────────────────────────────────────
        self._results_lbl = QLabel('')
        self._results_lbl.setStyleSheet(
            'color: #FFB74D; font-size: 11px; font-family: monospace; '
            'background: #0a0a0a; padding: 6px; border: 1px solid #1a1a1a; '
            'border-radius: 3px;')
        self._results_lbl.setWordWrap(True)
        self.add_widget(self._results_lbl)

        # ── Plot signal picker ───────────────────────────────────────────
        sig_lbl = QLabel('Plots:')
        sig_lbl.setStyleSheet('color: #e0e0e0; font-size: 11px; font-weight: bold;')
        self.add_widget(sig_lbl)

        self._sig_list = QListWidget()
        self._sig_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self._sig_list.setMaximumHeight(140)
        self._sig_list.setStyleSheet(
            'QListWidget { background: #0a0a0a; color: #e0e0e0; '
            'border: 1px solid #333; font-size: 11px; }'
            'QListWidget::item { padding: 2px 6px; }'
            'QListWidget::item:selected { background: #7B3F00; color: white; }')
        for key, label in _TRANSIENT_SIGNALS:
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, key)
            self._sig_list.addItem(item)
        # Default selections
        for i in range(self._sig_list.count()):
            key = self._sig_list.item(i).data(Qt.ItemDataRole.UserRole)
            if key in ('yaw_rate', 'ay', 'roll'):
                self._sig_list.item(i).setSelected(True)
        self._sig_list.itemSelectionChanged.connect(self._on_signals_changed)
        self.add_widget(self._sig_list)

        # Initialize visibility based on default test
        self._on_test_changed()

    # ── Test type dependent visibility ───────────────────────────────────

    def _show_rows(self, visible_keys: set):
        """Show only the (label, spinbox) rows whose keys are in visible_keys;
        hide the rest so redundant inputs don't clutter the UI."""
        for key, (lbl, sb) in self._rows.items():
            vis = key in visible_keys
            lbl.setVisible(vis)
            sb.setVisible(vis)

    def _on_test_changed(self):
        key = self._test_combo.currentData()
        # FSAE skidpad fixes radius at 9.125 m, peak steer = L/R, sim
        # duration = lap count × lap time.  Only the speed (and timestep)
        # are real user inputs — hide everything else.
        # The target-lat-g row is shown for skidpad tests only (they have
        # a fixed radius so v = √(a_y·g·R) is well-defined); for open
        # straight-line tests (step/ramp/sine) lateral-g isn't a setpoint
        # so the mode toggle is forced back to "Target speed".
        if key == 'skidpad_full':
            self._show_rows({'speed', 'lat_g', 'dt'})
            self._skidpad_radius.setValue(9.125)        # FSAE regulation
            self._solve_mode_combo.setEnabled(True)
            # Time step stays at whatever user picked.
        elif key == 'skidpad':
            # Single-circle legacy mode — let the user pick radius + ramp.
            self._show_rows({'speed', 'lat_g', 'radius', 'ramp', 'sim', 'dt'})
            self._solve_mode_combo.setEnabled(True)
        elif key == 'ramp':
            self._show_rows({'speed', 'steer', 'ramp', 'sim', 'dt'})
            self._force_target_speed_mode()
        elif key == 'step':
            self._show_rows({'speed', 'steer', 'sim', 'dt'})
            self._force_target_speed_mode()
        elif key == 'sine':
            self._show_rows({'speed', 'steer', 'ramp', 'sim', 'dt'})
            self._force_target_speed_mode()
        else:
            self._show_rows(set(self._rows.keys()))
            self._solve_mode_combo.setEnabled(True)

        # For sine, peak steer becomes amplitude and ramp becomes frequency (Hz)
        if key == 'sine':
            self._ramp_duration.setSuffix(' Hz')
            self._ramp_duration.setValue(0.8)   # 0.8 Hz default
        else:
            self._ramp_duration.setSuffix(' s')

        # Re-apply the solve-mode enable/disable so the visible row
        # actually matches the current mode.
        self._on_solve_mode_changed()

    def _force_target_speed_mode(self):
        """For open tests (step/ramp/sine) there's no fixed radius, so
        target-lat-g mode is meaningless.  Snap the combo back to
        target_speed and grey it out."""
        for i in range(self._solve_mode_combo.count()):
            if self._solve_mode_combo.itemData(i) == 'target_speed':
                self._solve_mode_combo.blockSignals(True)
                self._solve_mode_combo.setCurrentIndex(i)
                self._solve_mode_combo.blockSignals(False)
                break
        self._solve_mode_combo.setEnabled(False)

    def _on_solve_mode_changed(self):
        """Enable the active setpoint spin and grey out the other so it
        reads as obviously non-authoritative.  The visible rows are
        still controlled by ``_on_test_changed`` — we only tweak the
        enabled state here, never the visibility."""
        mode = self._solve_mode_combo.currentData()
        speed_active = (mode == 'target_speed')
        # Target speed spin is authoritative in target_speed mode;
        # lat-g spin is authoritative in target_lat_g mode.
        self._target_speed.setEnabled(speed_active)
        self._target_lat_g.setEnabled(not speed_active)
        # Visual cue: labels lose their contrast when the field isn't
        # in use, so users don't enter a value that gets ignored.
        sp_lbl = self._rows['speed'][0]
        lg_lbl = self._rows['lat_g'][0]
        sp_lbl.setStyleSheet('' if speed_active else 'color: #555;')
        lg_lbl.setStyleSheet('color: #555;' if speed_active else '')

    # ── Public API ───────────────────────────────────────────────────────

    def get_params(self) -> dict:
        # Target speed is in mph on the UI; convert to SI for the solver.
        # 1 mph = 0.44704 m/s (exact).
        # ``solve_mode`` controls whether ``target_speed_ms`` is used
        # directly, or whether the dispatch derives it from
        # ``target_lat_g`` via v = √(a_y · g · R).
        return {
            'test_type':        self._test_combo.currentData(),
            'solve_mode':       self._solve_mode_combo.currentData(),
            'target_speed_ms':  self._target_speed.value() * 0.44704,
            'target_lat_g':     self._target_lat_g.value(),
            'skidpad_radius_m': self._skidpad_radius.value(),
            'peak_steer_deg':   self._steer_deg.value(),
            'ramp_duration_s':  self._ramp_duration.value(),
            'sim_duration_s':   self._sim_duration.value(),
            'dt_s':             self._dt_ms.value() / 1000.0,
            'direction':        self._dir_combo.currentText().lower(),
            # Damper bump/rebound coefficients (N·s/m at the shock).
            # main_window combines these with the car's motion ratio and
            # track width to derive `roll_damping_Nms_rad`; nothing
            # downstream reads `roll_damping` directly any more.
            'damper_F_bump_Nspm':    self._dmp_F_bump.value(),
            'damper_F_rebound_Nspm': self._dmp_F_reb.value(),
            'damper_R_bump_Nspm':    self._dmp_R_bump.value(),
            'damper_R_rebound_Nspm': self._dmp_R_reb.value(),
            'steer_tau_s':      self._steer_tau.value(),
            # Ixx, Izz, Ackermann %, and derived roll damping are
            # auto-filled by main_window and pushed back via
            # set_auto_info() for display.
        }

    # ── Save / load (every input on the panel) ───────────────────────────

    def get_state(self) -> dict:
        """JSON-friendly snapshot of every user-facing input on the
        skidpad/transient panel.  Stores the *raw* widget values (mph,
        ms, deg) rather than SI so a save file is human-readable and
        consistent with what the user typed.  The auto-info readout
        (Izz, Ixx, Ackermann %, derived roll damping) is recomputed on
        every solve so it isn't worth saving.
        """
        return {
            'test_type':           self._test_combo.currentData(),
            'solve_mode':          self._solve_mode_combo.currentData(),
            'direction':           self._dir_combo.currentText(),
            'target_speed_mph':    float(self._target_speed.value()),
            'target_lat_g':        float(self._target_lat_g.value()),
            'skidpad_radius_m':    float(self._skidpad_radius.value()),
            'peak_steer_deg':      float(self._steer_deg.value()),
            'ramp_duration':       float(self._ramp_duration.value()),
            'sim_duration_s':      float(self._sim_duration.value()),
            'dt_ms':               float(self._dt_ms.value()),
            'damper_F_bump_Nspm':  float(self._dmp_F_bump.value()),
            'damper_F_rebound_Nspm': float(self._dmp_F_reb.value()),
            'damper_R_bump_Nspm':  float(self._dmp_R_bump.value()),
            'damper_R_rebound_Nspm': float(self._dmp_R_reb.value()),
            'steer_tau_s':         float(self._steer_tau.value()),
            'plot_signals':        self.get_selected_signals(),
        }

    def set_state(self, d: dict) -> None:
        """Restore every input from a dict produced by :meth:`get_state`.

        The combobox values are restored by their ``UserRole`` data
        (``currentData()``) — same way :meth:`get_params` reads them —
        so they survive renames of the visible label without breaking
        the save format.
        """
        if not isinstance(d, dict):
            return

        def _set_spin(sb, key):
            if key in d:
                try:
                    sb.setValue(float(d[key]))
                except (TypeError, ValueError):
                    pass

        def _set_combo_by_data(combo, key):
            if key not in d:
                return
            target = d[key]
            for i in range(combo.count()):
                if combo.itemData(i) == target:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(i)
                    combo.blockSignals(False)
                    return

        widgets = [
            self._test_combo, self._solve_mode_combo, self._dir_combo,
            self._target_speed, self._target_lat_g, self._skidpad_radius,
            self._steer_deg, self._ramp_duration, self._sim_duration,
            self._dt_ms, self._dmp_F_bump, self._dmp_F_reb,
            self._dmp_R_bump, self._dmp_R_reb, self._steer_tau,
        ]
        for w in widgets:
            w.blockSignals(True)
        try:
            # 1) Combo selections first — _on_test_changed reads them.
            _set_combo_by_data(self._test_combo,       'test_type')
            _set_combo_by_data(self._solve_mode_combo, 'solve_mode')
            if 'direction' in d:
                idx = self._dir_combo.findText(str(d['direction']))
                if idx >= 0:
                    self._dir_combo.setCurrentIndex(idx)

            # 2) Apply test-type visibility / enable rules NOW (before
            # writing spinbox values) — for the skidpad_full test type
            # this forces the radius to the FSAE regulation 9.125 m,
            # which is correct behaviour for that mode but would clobber
            # the saved value if we wrote the radius first.  Writing
            # spinboxes *after* _on_test_changed lets the saved values
            # win for any field that the test type doesn't lock.
            try:
                self._on_test_changed()
            except Exception:
                pass

            # 3) Spinboxes — saved values overwrite any test-type forcing.
            _set_spin(self._target_speed,    'target_speed_mph')
            _set_spin(self._target_lat_g,    'target_lat_g')
            _set_spin(self._skidpad_radius,  'skidpad_radius_m')
            _set_spin(self._steer_deg,       'peak_steer_deg')
            _set_spin(self._ramp_duration,   'ramp_duration')
            _set_spin(self._sim_duration,    'sim_duration_s')
            _set_spin(self._dt_ms,           'dt_ms')
            _set_spin(self._dmp_F_bump,      'damper_F_bump_Nspm')
            _set_spin(self._dmp_F_reb,       'damper_F_rebound_Nspm')
            _set_spin(self._dmp_R_bump,      'damper_R_bump_Nspm')
            _set_spin(self._dmp_R_reb,       'damper_R_rebound_Nspm')
            _set_spin(self._steer_tau,       'steer_tau_s')

            # Plot-signal selections (multi-select list)
            if isinstance(d.get('plot_signals'), list):
                wanted = set(d['plot_signals'])
                self._sig_list.blockSignals(True)
                for i in range(self._sig_list.count()):
                    it = self._sig_list.item(i)
                    key = it.data(Qt.ItemDataRole.UserRole)
                    it.setSelected(key in wanted)
                self._sig_list.blockSignals(False)
        finally:
            for w in widgets:
                w.blockSignals(False)

    def set_auto_info(self, yaw_Izz: float, sprung_Ixx: float,
                      ackermann_pct: float,
                      sim_duration_s: float = float('nan'),
                      peak_steer_deg: float = float('nan'),
                      derived_speed_ms: float = float('nan'),
                      derived_roll_damping: float = float('nan')):
        """Echo the auto-computed transient parameters back to the UI
        so the user can see what the dispatch used.  ``derived_speed_ms``
        is only populated when the panel is in target-lat-g mode — it's
        the speed that came out of v = √(a_y · g · R).
        ``derived_roll_damping`` is the c_phi the dispatch built from
        the damper bump/rebound rates × MR² × t²/4.
        """
        parts = [
            f'Izz = {yaw_Izz:,.1f} kg·m²',
            f'Ixx = {sprung_Ixx:,.1f} kg·m²',
            f'Ackermann = {ackermann_pct:+.0f} %',
        ]
        if not np.isnan(derived_roll_damping):
            parts.append(f'c_phi = {derived_roll_damping:,.0f} N·m·s/rad')
        if not np.isnan(peak_steer_deg):
            parts.append(f'Peak steer = {peak_steer_deg:.2f}°')
        if not np.isnan(sim_duration_s):
            parts.append(f'Sim duration = {sim_duration_s:.1f} s')
        if not np.isnan(derived_speed_ms):
            # Echo both SI and driver-friendly units so the lat-g value
            # is reconcilable with the Target speed field.
            mph = derived_speed_ms / 0.44704
            parts.append(f'Derived speed = {derived_speed_ms:.2f} m/s ({mph:.1f} mph)')
        self._auto_info.setText('Auto: ' + '   '.join(parts))

    def get_selected_signals(self) -> list:
        out = []
        for i in range(self._sig_list.count()):
            it = self._sig_list.item(i)
            if it.isSelected():
                out.append(it.data(Qt.ItemDataRole.UserRole))
        return out

    def set_solving(self, busy: bool):
        self._sim_btn.setEnabled(not busy)
        self._sim_btn.setText('Simulating...' if busy else 'Simulate')

    def set_status(self, msg: str):
        self._status.setText(msg)

    def show_result(self, result):
        """Display the TransientResult metrics in the readout label."""
        if result is None:
            self._results_lbl.setText('')
            return
        lines = [
            f'Peak lateral g     : {result.peak_lateral_g:6.3f}',
            f'Steady lateral g   : {result.steady_state_lateral_g:6.3f}',
            f'Peak roll          : {result.peak_roll_deg:6.2f} deg',
            f'Steady roll        : {result.steady_state_roll_deg:6.2f} deg',
            f'Yaw rise time (10-90): {result.yaw_rate_rise_time_s*1000:6.0f} ms',
            f'Yaw overshoot      : {result.yaw_rate_overshoot_pct:6.1f} %',
            f'Yaw settling time  : {result.yaw_rate_settling_time_s*1000:6.0f} ms',
            f'Peak US (alpha_F-R): {result.peak_understeer_deg:6.2f} deg',
        ]
        self._results_lbl.setText('\n'.join(lines))

    # ── Signals ──────────────────────────────────────────────────────────

    def _on_simulate(self):
        self.simulate_requested.emit(self.get_params())

    def _on_signals_changed(self):
        self.signals_changed.emit(self.get_selected_signals())


# -----------------------------------------------------------------------------
class VehicleConstantsPanel(CollapsibleSection):
    """Button that opens a popup with all DERIVED vehicle constants
    (wheel rates, roll stiffness, ride freq, LLTD, etc.).
    Inputs (mass, track, springs) are NOT shown — only computed outputs.
    Intermediate values are included so each result can be traced to its
    inputs and cross-checked against a spreadsheet.
    """

    def __init__(self):
        super().__init__('Vehicle Constants', header_color='#4FC3F7')
        self.set_info(section_info.VEHICLE_CONSTANTS)
        self._last_veh = None
        self._dlg: QDialog | None = None
        self._label: QLabel | None = None

        btn = QPushButton('Show Computed Constants')
        btn.setStyleSheet(
            'QPushButton { background: #1a1a1a; color: #4FC3F7; '
            'border: 1px solid #444; padding: 6px 16px; border-radius: 3px; '
            'font-weight: bold; }'
            'QPushButton:hover { background: #2a2a2a; }')
        btn.clicked.connect(self._show_popup)
        self.add_widget(btn)

    def update(self, veh):
        """Cache the vehicle params and refresh the popup if open."""
        self._last_veh = veh
        if self._dlg is not None and self._dlg.isVisible():
            self._refresh()

    def _show_popup(self):
        if self._dlg is not None and self._dlg.isVisible():
            self._dlg.raise_()
            return
        self._dlg = QDialog(self)
        self._dlg.setWindowTitle('Vehicle Constants — derived from inputs')
        self._dlg.resize(680, 780)
        self._dlg.setStyleSheet('QDialog { background: #0a0a0a; }')

        scroll = QScrollArea(self._dlg)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._label = QLabel('Build dynamics solver to populate')
        self._label.setStyleSheet(
            'color: #e0e0e0; font-size: 14px; font-family: "Consolas","Courier New",monospace;'
            'background: #0a0a0a; padding: 16px;')
        self._label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self._label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._label.setWordWrap(False)
        scroll.setWidget(self._label)

        lay = QVBoxLayout(self._dlg)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(scroll)

        self._refresh()
        self._dlg.show()

    # ---------------------------------------------------------------
    def _refresh(self):
        if self._last_veh is None or self._label is None:
            return
        import math
        veh = self._last_veh
        L = 175.127      # N/m → lbf/in
        g = 9.81
        lines = []

        # ── Aligned-column helpers ─────────────────────────────────
        LBLW = 22       # label width
        VALW = 9        # value width per axle
        def hdr(title, formula=''):
            lines.append('')
            lines.append('═' * 64)
            lines.append(f'  {title}')
            if formula:
                lines.append(f'    {formula}')
            lines.append('─' * 64)
            lines.append(f'  {"":<{LBLW}}{"Front":>{VALW}}  {"Rear":>{VALW}}   Units')
            lines.append('─' * 64)
        def row(label, fv, rv, unit=''):
            lines.append(f'  {label:<{LBLW}}{fv:>{VALW}}  {rv:>{VALW}}   {unit}')
        def note(s):
            lines.append(f'  ‣ {s}')

        # ═══ Wheel rate (spring × MR²) ═════════════════════════════
        mr_f, mr_r = veh.motion_ratio_front, veh.motion_ratio_rear
        sr_f, sr_r = veh.spring_rate_front_Npm, veh.spring_rate_rear_Npm
        wr_f, wr_r = veh.wheel_rate_front_Npm, veh.wheel_rate_rear_Npm
        hdr('WHEEL RATE', 'K_w  =  K_spring · MR²')
        row('Motion ratio (geom)', f'{mr_f:.4f}', f'{mr_r:.4f}')
        row('Spring rate',  f'{sr_f/L:.1f}',  f'{sr_r/L:.1f}',  'lbf/in')
        row('→ Wheel rate', f'{wr_f/L:.1f}',  f'{wr_r/L:.1f}',  'lbf/in')
        row('',             f'{wr_f/1000:.2f}', f'{wr_r/1000:.2f}', 'N/mm')

        # ═══ ARB wheel rate ════════════════════════════════════════
        arb_f, arb_r = veh.arb_rate_front_Npm, veh.arb_rate_rear_Npm
        hdr('ARB WHEEL RATE')
        row('K_ARB at wheel',  f'{arb_f/L:.1f}',  f'{arb_r/L:.1f}',  'lbf/in')
        row('',                f'{arb_f/1000:.2f}', f'{arb_r/1000:.2f}', 'N/mm')

        # ═══ Ride rate (wheel ⊕ tire in series) ════════════════════
        kt = veh.tire_rate_Npm
        rr_f, rr_r = veh.ride_rate_front_Npm, veh.ride_rate_rear_Npm
        hdr('RIDE RATE', 'K_ride  =  (K_w · K_tire) / (K_w + K_tire)')
        note(f'Tire rate = {kt/L:.0f} lbf/in   ({kt/1000:.1f} N/mm)')
        row('→ Ride rate',  f'{rr_f/L:.1f}', f'{rr_r/L:.1f}', 'lbf/in')

        # ═══ Ride frequency ════════════════════════════════════════
        m_f = veh.sprung_mass_kg * veh.front_weight_fraction / 2
        m_r = veh.sprung_mass_kg * veh.rear_weight_fraction / 2
        f_f = math.sqrt(rr_f / m_f) / (2 * math.pi) if m_f > 0 and rr_f > 0 else 0
        f_r = math.sqrt(rr_r / m_r) / (2 * math.pi) if m_r > 0 and rr_r > 0 else 0
        hdr('RIDE FREQUENCY', 'f  =  (1/2π) · √(K_ride / m_corner)')
        row('Sprung mass / corner', f'{m_f:.1f}', f'{m_r:.1f}', 'kg')
        row('→ Ride freq',          f'{f_f:.2f}', f'{f_r:.2f}', 'Hz')
        if f_f > 0 and f_r > 0:
            ratio = f_r / f_f
            n = ('flat ride' if 0.95 <= ratio <= 1.05 else
                 'R faster → nose-down pitch' if ratio > 1.05 else
                 'F faster → nose-up pitch (Olley flat-ride target)')
            note(f'freq ratio R/F = {ratio:.3f}   ({n})')

        # ═══ Roll stiffness ════════════════════════════════════════
        tf = veh.front_track_m if hasattr(veh, 'front_track_m') else getattr(veh, 'track_front_m', 0)
        tr = veh.rear_track_m  if hasattr(veh, 'rear_track_m')  else getattr(veh, 'track_rear_m', 0)
        Kphi_spring_f = wr_f * tf**2 / 2
        Kphi_spring_r = wr_r * tr**2 / 2
        Kphi_arb_f    = arb_f * tf**2 / 2
        Kphi_arb_r    = arb_r * tr**2 / 2
        rs_f = veh.roll_stiffness_front_Npm_rad
        rs_r = veh.roll_stiffness_rear_Npm_rad
        rs_t = rs_f + rs_r
        hdr('ROLL STIFFNESS', 'K_φ  =  (K_w + K_ARB) · t² / 2')
        row('From springs', f'{Kphi_spring_f:.0f}', f'{Kphi_spring_r:.0f}', 'Nm/rad')
        row('From ARB',     f'{Kphi_arb_f:.0f}',    f'{Kphi_arb_r:.0f}',    'Nm/rad')
        row('→ Total K_φ',  f'{rs_f:.0f}',          f'{rs_r:.0f}',          'Nm/rad')
        note(f'Σ K_φ  =  {rs_t:.0f}  Nm/rad   =   {rs_t*math.pi/180:.1f}  Nm/°')

        # ═══ LLTD ───────────────────────────────────────────────────
        lltd = rs_f / rs_t * 100 if rs_t > 0 else 50
        wf_pct = veh.front_weight_fraction * 100
        lltd_minus_w = lltd - wf_pct        # the meaningful number
        hdr('LATERAL LOAD-TRANSFER DISTRIBUTION  (elastic only)',
            'LLTD_F  =  K_φ,F / (K_φ,F + K_φ,R)')
        note(f'LLTD          = {lltd:5.1f}%  front')
        note(f'Weight dist   = {wf_pct:5.1f}%  front')
        note(f'LLTD − weight = {lltd_minus_w:+5.1f}  pts')
        if abs(lltd_minus_w) < 2:
            verdict = 'elastic-LT-neutral (within ±2 pts of weight)'
        elif lltd_minus_w > 0:
            verdict = ('more elastic LT at F than weight share → '
                       'front axle loses grip first → understeer bias '
                       '(elastic only — see below)')
        else:
            verdict = ('more elastic LT at R than weight share → '
                       'rear axle loses grip first → oversteer bias '
                       '(elastic only — see below)')
        note(verdict)
        lines.append('')
        note('CAVEAT: this is the SPRING+ARB part only.')
        note('Real balance also depends on geometric LT (roll-centre')
        note('height), aero, tyre compound, static toe, dampers, brake')
        note('bias on entry. Use the MMD or skidpad sweep for the full')
        note('picture — this number is just the elastic baseline.')

        # ═══ Roll gradient ─────────────────────────────────────────
        hdr('ROLL GRADIENT', 'φ / A_y  =  m·g·h / K_φ,total')
        if rs_t > 0:
            mgh = veh.total_mass_kg * g * veh.cg_height_m
            roll_per_g_deg = math.degrees(mgh / rs_t)
            note(f'Roll moment per g  =  {mgh:.0f}  Nm')
            note(f'→ Roll gradient     =  {roll_per_g_deg:.3f}  °/g')
            target = ('soft (>1.5°/g)' if roll_per_g_deg > 1.5 else
                      'race-typical (0.5–1.5°/g)' if roll_per_g_deg > 0.5 else
                      'stiff (<0.5°/g)')
            note(f'Bucket: {target}')

        # ═══ Damping (optional, only if defined) ───────────────────
        zeta = getattr(veh, 'damping_ratio', None)
        if zeta is not None and zeta > 0:
            hdr('DAMPING COEFFICIENT  (target)', 'c  =  2·ζ · √(K_w · m_corner)')
            c_f = 2 * zeta * math.sqrt(wr_f * m_f)
            c_r = 2 * zeta * math.sqrt(wr_r * m_r)
            note(f'Target ζ = {zeta:.2f}')
            row('c at wheel', f'{c_f:.0f}', f'{c_r:.0f}', 'N·s/m')

        self._label.setText('\n'.join(lines))


# -----------------------------------------------------------------------------
class AnalysisPlotsPanel(CollapsibleSection):
    """Buttons for higher-level validation plots.

    Each button emits a signal. MainWindow connects each signal to a handler
    that builds a matplotlib Figure (from vahan.analysis_plots) and shows it
    in a PlotDialog (with copy/save context menu).
    """
    brake_capacity_requested    = pyqtSignal()
    ride_freq_requested         = pyqtSignal()
    rc_vs_roll_requested        = pyqtSignal()
    steering_torque_requested   = pyqtSignal()
    ackermann_demand_requested  = pyqtSignal()
    ackermann_fzfy_requested    = pyqtSignal()
    rack_zero_ackermann_requested = pyqtSignal()
    mmd_requested               = pyqtSignal()
    wheel_rate_linearity_requested = pyqtSignal()
    llt_requested               = pyqtSignal()

    def __init__(self):
        super().__init__('Analysis & Validation Plots', header_color='#FFD600')
        self.set_info(section_info.ANALYSIS_PLOTS)
        self._build()

    def _build(self):
        # Brake capacity inputs
        g = QGridLayout(); g.setSpacing(4)
        lbl = QLabel('Brake-torque capacity')
        lbl.setStyleSheet('font-weight:bold;color:#FFA726;')
        g.addWidget(lbl, 0, 0, 1, 4)
        g.addWidget(QLabel('Pad mu:'), 1, 0)
        self._pad_mu = _spin(0.1, 0.7, 0.45, '', dec=2, step=0.01)
        g.addWidget(self._pad_mu, 1, 1)
        g.addWidget(QLabel('Piston area:'), 1, 2)
        self._piston_area = _spin(50, 5000, 1583, ' mm^2', dec=0, step=10)
        self._piston_area.setToolTip('Total piston area per caliper (one side, mm^2). '
                                      'GP200 = 2 x 1.25" pistons = 1583 mm^2')
        g.addWidget(self._piston_area, 1, 3)
        g.addWidget(QLabel('Rotor OD:'), 2, 0)
        self._rotor_od = _spin(4.0, 14.0, 7.5, ' in', dec=2, step=0.1)
        g.addWidget(self._rotor_od, 2, 1)
        g.addWidget(QLabel('Pad height:'), 2, 2)
        self._pad_height = _spin(0.3, 2.0, 0.85, ' in', dec=2, step=0.05)
        g.addWidget(self._pad_height, 2, 3)
        g.addWidget(QLabel('Max line P:'), 3, 0)
        self._max_line_psi = _spin(500, 5000, 1500, ' psi', dec=0, step=50)
        g.addWidget(self._max_line_psi, 3, 1)
        self.add_layout(g)
        btn = QPushButton('Plot brake-torque capacity vs deceleration')
        btn.clicked.connect(lambda: self.brake_capacity_requested.emit())
        self.add_widget(btn)
        self.add_widget(_panel_sep())

        # Wheel-rate linearity inputs
        g = QGridLayout(); g.setSpacing(4)
        lbl = QLabel('Wheel-rate linearity across travel')
        lbl.setStyleSheet('font-weight:bold;color:#FFA726;')
        g.addWidget(lbl, 0, 0, 1, 4)
        g.addWidget(QLabel('Travel ±:'), 1, 0)
        self._wr_travel_mm = _spin(10, 100, 50, ' mm', dec=0, step=5)
        self._wr_travel_mm.setToolTip('Half-range of wheel travel swept (mm)')
        g.addWidget(self._wr_travel_mm, 1, 1)
        g.addWidget(QLabel('Operating ±:'), 1, 2)
        self._wr_op_mm = _spin(5, 60, 25, ' mm', dec=0, step=5)
        self._wr_op_mm.setToolTip('Typical operating travel band (highlighted)')
        g.addWidget(self._wr_op_mm, 1, 3)
        g.addWidget(QLabel('Resolution:'), 2, 0)
        self._wr_n = _spin(11, 121, 41, ' pts', dec=0, step=2)
        g.addWidget(self._wr_n, 2, 1)
        self.add_layout(g)
        btn = QPushButton('Plot wheel-rate linearity across travel')
        btn.clicked.connect(lambda: self.wheel_rate_linearity_requested.emit())
        self.add_widget(btn)
        self.add_widget(_panel_sep())

        # Lateral load transfer distribution
        btn = QPushButton('Plot lateral load transfer distribution')
        btn.clicked.connect(lambda: self.llt_requested.emit())
        self.add_widget(btn)
        self.add_widget(_panel_sep())

        # Ride freq inputs
        g = QGridLayout(); g.setSpacing(4)
        lbl = QLabel('Ride frequency - pitch over a bump')
        lbl.setStyleSheet('font-weight:bold;color:#FFA726;')
        g.addWidget(lbl, 0, 0, 1, 4)
        g.addWidget(QLabel('zeta (damping):'), 1, 0)
        self._damping_ratio = _spin(0.1, 1.5, 0.5, '', dec=2, step=0.05)
        self._damping_ratio.setToolTip('Critical damping ratio. ~0.3-0.5 lively. ~0.5-0.7 race typical.')
        g.addWidget(self._damping_ratio, 1, 1)
        g.addWidget(QLabel('Bump:'), 1, 2)
        self._bump_h = _spin(5, 50, 20, ' mm', dec=0, step=5)
        g.addWidget(self._bump_h, 1, 3)
        g.addWidget(QLabel('Speeds (m/s):'), 2, 0)
        sp = QLabel('5, 10, 15, 20')
        sp.setStyleSheet('color:#888;font-size:10px;')
        g.addWidget(sp, 2, 1, 1, 3)
        self.add_layout(g)
        btn = QPushButton('Plot pitch response over a bump at varying speeds')
        btn.clicked.connect(lambda: self.ride_freq_requested.emit())
        self.add_widget(btn)
        self.add_widget(_panel_sep())

        # RC vs body roll
        g = QGridLayout(); g.setSpacing(4)
        lbl = QLabel('Roll-centre height vs body roll')
        lbl.setStyleSheet('font-weight:bold;color:#FFA726;')
        g.addWidget(lbl, 0, 0, 1, 4)
        g.addWidget(QLabel('Max lat g:'), 1, 0)
        self._rc_max_g = _spin(0.5, 4.0, 2.0, ' g', dec=1, step=0.1)
        g.addWidget(self._rc_max_g, 1, 1)
        self.add_layout(g)
        btn = QPushButton('Plot RC height vs body roll')
        btn.clicked.connect(lambda: self.rc_vs_roll_requested.emit())
        self.add_widget(btn)
        self.add_widget(_panel_sep())

        # Steering torque
        g = QGridLayout(); g.setSpacing(4)
        lbl = QLabel('Steering-wheel torque vs steer angle')
        lbl.setStyleSheet('font-weight:bold;color:#FFA726;')
        g.addWidget(lbl, 0, 0, 1, 4)
        g.addWidget(QLabel('Steering ratio:'), 1, 0)
        self._steer_ratio = _spin(2.0, 25.0, 6.0, ':1', dec=1, step=0.5)
        self._steer_ratio.setToolTip('Steering-wheel to road-wheel angle ratio')
        g.addWidget(self._steer_ratio, 1, 1)
        g.addWidget(QLabel('Speeds:'), 1, 2)
        sp = QLabel('10, 15, 20 m/s')
        sp.setStyleSheet('color:#888;font-size:10px;')
        g.addWidget(sp, 1, 3)
        self.add_layout(g)
        btn = QPushButton('Plot steering torque vs steer angle')
        btn.clicked.connect(lambda: self.steering_torque_requested.emit())
        self.add_widget(btn)
        self.add_widget(_panel_sep())

        # Ackermann demand vs supply
        g = QGridLayout(); g.setSpacing(4)
        lbl = QLabel('Ackermann slip-angle budget')
        lbl.setStyleSheet('font-weight:bold;color:#FFA726;')
        g.addWidget(lbl, 0, 0, 1, 4)
        g.addWidget(QLabel('Ackermann %:'), 1, 0)
        self._ack_demand_pct = _spin(-100, 100, 33, ' %', dec=0, step=1)
        self._ack_demand_pct.setToolTip('Car\'s actual Ackermann percentage '
                                         '(from kinematic steer sweep)')
        g.addWidget(self._ack_demand_pct, 1, 1)
        g.addWidget(QLabel('Speed:'), 1, 2)
        self._ack_demand_speed = _spin(3, 40, 13.4, ' m/s', dec=1, step=0.5)
        g.addWidget(self._ack_demand_speed, 1, 3)
        self.add_layout(g)
        btn = QPushButton('Plot slip-angle demand vs Ackermann supply')
        btn.clicked.connect(lambda: self.ackermann_demand_requested.emit())
        self.add_widget(btn)
        btn2 = QPushButton('Plot Fz–Fy operating map (demand vs Ackermann)')
        btn2.clicked.connect(lambda: self.ackermann_fzfy_requested.emit())
        self.add_widget(btn2)
        btn3 = QPushButton('Find rack fore-aft position for 0% Ackermann')
        btn3.clicked.connect(lambda: self.rack_zero_ackermann_requested.emit())
        self.add_widget(btn3)
        self.add_widget(_panel_sep())

        # MMD
        g = QGridLayout(); g.setSpacing(4)
        lbl = QLabel('Milliken Moment Diagram (pure cornering)')
        lbl.setStyleSheet('font-weight:bold;color:#FFA726;')
        g.addWidget(lbl, 0, 0, 1, 4)
        g.addWidget(QLabel('Speed:'), 1, 0)
        self._mmd_speed = _spin(5, 40, 13.4, ' m/s', dec=1, step=0.5)
        g.addWidget(self._mmd_speed, 1, 1)
        g.addWidget(QLabel('β range:'), 1, 2)
        self._mmd_beta = _spin(2, 20, 8, ' deg', dec=0, step=1)
        g.addWidget(self._mmd_beta, 1, 3)
        g.addWidget(QLabel('δ range:'), 2, 0)
        self._mmd_delta = _spin(5, 30, 20, ' deg', dec=0, step=1)
        g.addWidget(self._mmd_delta, 2, 1)
        # ARB lever arm deltas (mm; positive = longer arm = softer)
        lbl_arm = QLabel('ARB lever arm Δ  (+ longer = softer):')
        lbl_arm.setStyleSheet('color:#888;font-size:10px;')
        g.addWidget(lbl_arm, 3, 0, 1, 4)
        g.addWidget(QLabel('ΔArm front:'), 4, 0)
        self._mmd_darm_f = _spin(-80, 80, 0, ' mm', dec=0, step=5.0)
        g.addWidget(self._mmd_darm_f, 4, 1)
        g.addWidget(QLabel('ΔArm rear:'), 4, 2)
        self._mmd_darm_r = _spin(-80, 80, 0, ' mm', dec=0, step=5.0)
        g.addWidget(self._mmd_darm_r, 4, 3)
        # Static toe (deg, + = toe-in)
        lbl_toe = QLabel('Static toe  (+ = toe-in):')
        lbl_toe.setStyleSheet('color:#888;font-size:10px;')
        g.addWidget(lbl_toe, 5, 0, 1, 4)
        g.addWidget(QLabel('Toe front:'), 6, 0)
        self._mmd_toe_f = _spin(-3.0, 3.0, 0.0, ' °', dec=1, step=0.1)
        g.addWidget(self._mmd_toe_f, 6, 1)
        g.addWidget(QLabel('Toe rear:'), 6, 2)
        self._mmd_toe_r = _spin(-3.0, 3.0, 0.0, ' °', dec=1, step=0.1)
        g.addWidget(self._mmd_toe_r, 6, 3)
        self.add_layout(g)
        btn = QPushButton('Plot Milliken Moment Diagram')
        btn.clicked.connect(lambda: self.mmd_requested.emit())
        self.add_widget(btn)

    # Getters
    def brake_inputs(self):
        return dict(pad_mu=self._pad_mu.value(),
                    piston_area_per_caliper_mm2=self._piston_area.value(),
                    rotor_outer_dia_in=self._rotor_od.value(),
                    pad_height_in=self._pad_height.value(),
                    typical_max_line_psi=self._max_line_psi.value())

    def ride_freq_inputs(self):
        return dict(damping_ratio=self._damping_ratio.value(),
                    bump_height_m=self._bump_h.value() / 1000,
                    speeds_mps=(5, 10, 15, 20))

    def wheel_rate_linearity_inputs(self):
        tr = float(self._wr_travel_mm.value())
        op = float(self._wr_op_mm.value())
        return dict(
            travel_range_mm=(-tr, tr),
            operating_range_mm=(-op, op),
            n_points=int(self._wr_n.value()),
        )

    def rc_vs_roll_inputs(self):
        return dict(max_lat_g=self._rc_max_g.value())

    def steering_torque_inputs(self):
        return dict(steering_ratio=self._steer_ratio.value(),
                    speeds_mps=(10, 15, 20))

    def ackermann_demand_inputs(self):
        return dict(ackermann_pct=self._ack_demand_pct.value(),
                    velocity_mps=self._ack_demand_speed.value())

    def mmd_inputs(self):
        return dict(
            velocity_mps=self._mmd_speed.value(),
            beta_range_deg=(-self._mmd_beta.value(), self._mmd_beta.value()),
            beta_n=int(self._mmd_beta.value()) + 1,
            delta_range_deg=(-self._mmd_delta.value(), self._mmd_delta.value()),
            delta_n=int(self._mmd_delta.value()) // 2 + 1,
            # Lever arm deltas (mm); stiffness scaling done in _on_plot_mmd
            delta_arm_front_mm=self._mmd_darm_f.value(),
            delta_arm_rear_mm=self._mmd_darm_r.value(),
            # Static toe (degrees)
            toe_front_deg=self._mmd_toe_f.value(),
            toe_rear_deg=self._mmd_toe_r.value(),
        )


def _panel_sep():
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet('color:#2a2a2a;background:#2a2a2a;max-height:1px;')
    return f


# ══════════════════════════════════════════════════════════════════════════════
#  FRAME / INTERFERENCE PANEL  (own panel so the toggle is always visible)
# ══════════════════════════════════════════════════════════════════════════════
class FrameInterferencePanel(CollapsibleSection):
    """Show the suspension at REAL part thickness in 3D and flag interferences.

    A pure view aid.  Tick the box to draw control arms / pushrod / tie-rod /
    rocker / ARB as solid tubes at the diameters entered here; parts on the
    same corner that overlap are highlighted RED with the overlap depth listed
    below, and a yellow cylinder shows the rocker-pivot bearing clearance.
    Lives in its own panel (not buried in Direct Edit) so it is always at hand.
    """

    frame_changed = pyqtSignal()   # toggle or any thickness changed -> re-render

    def __init__(self, parent=None):
        super().__init__('FRAME / INTERFERENCE CHECK', parent,
                         header_color='#FFD600')
        self._frame_chk = QCheckBox('Show frame at real thickness + flag clashes')
        self._frame_chk.setStyleSheet('QCheckBox { font-weight: bold; }')
        self._frame_chk.setToolTip(
            'Draw control arms / pushrod / tie-rod / rocker / ARB at their real '
            'thickness, plus a bearing-clearance cylinder at the rocker pivot. '
            'Parts on the same corner that overlap are highlighted RED with the '
            'overlap depth shown below. The rocker plate is bisected by the '
            'rocker plane (extrude radius = thickness / 2).')
        self._frame_chk.toggled.connect(lambda _=False: self.frame_changed.emit())
        self.add_widget(self._frame_chk)

        self._frame_dim_spins = {}

        def _dim(key, label, default, lo, hi):
            row = QHBoxLayout(); row.setSpacing(4)
            row.addWidget(QLabel(label))
            sb = QDoubleSpinBox(); sb.setRange(lo, hi); sb.setDecimals(1)
            sb.setSingleStep(0.5); sb.setValue(default); sb.setSuffix(' mm')
            sb.valueChanged.connect(lambda _=0.0: self.frame_changed.emit())
            row.addWidget(sb); row.addStretch(1)
            self.add_layout(row)
            self._frame_dim_spins[key] = sb

        _dim('ctrl_arm_od', 'Control-arm tube OD:', 19.0, 3.0, 60.0)
        _dim('pushrod_od', 'Pushrod / tie-rod OD:', 16.0, 3.0, 50.0)
        _dim('arb_od', 'ARB / drop-link OD:', 14.0, 3.0, 50.0)
        _dim('rocker_th', 'Rocker plate thickness:', 8.0, 1.0, 40.0)
        _dim('bearing_od', 'Rocker-pivot bearing OD:', 38.1, 5.0, 120.0)
        _dim('bearing_len', 'Bearing length:', 25.4, 3.0, 80.0)

        self._frame_readout = QLabel('—')
        self._frame_readout.setStyleSheet('color:#FF5252; font-size:10px;'
                                          ' padding:2px 2px 6px 2px;')
        self._frame_readout.setWordWrap(True)
        self.add_widget(self._frame_readout)

    def frame_enabled(self) -> bool:
        return self._frame_chk.isChecked()

    def frame_dims(self) -> dict:
        """Frame part thicknesses (mm), keyed for _frame_overlay."""
        return {k: sb.value() for k, sb in self._frame_dim_spins.items()}

    def set_frame_readout(self, text: str):
        self._frame_readout.setText(text)


# ══════════════════════════════════════════════════════════════════════════════
#  DIRECT EDIT MODE PANEL
# ══════════════════════════════════════════════════════════════════════════════
class DirectEditPanel(CollapsibleSection):
    """Keyboard-driven hardpoint editor.

    Lets the user pick any hardpoint by name (no need to click in the 3D
    viewport) and nudge it with WASD/QE in world axes.  Keys 1-6 set the
    step size.  Position display updates live.  Optional F↔R mirroring
    propagates the same delta to the matching hardpoint on the other axle.
    """

    enabled_changed     = pyqtSignal(bool)
    hp_selected         = pyqtSignal(str, str)   # (hp_name, corner)
    step_changed        = pyqtSignal(float)      # mm
    mirror_axle_changed = pyqtSignal(bool)
    apply_clicked       = pyqtSignal()
    discard_clicked     = pyqtSignal()
    # Plane-edit: rotate the entire actuation plane (pushrod + rocker +
    # spring + ARB drop-link, on the chosen axle) about an axis through a
    # pivot hardpoint.  Payload = (axle, axis_xyz, deg, pivot_key).
    plane_tilt_requested = pyqtSignal(str, str, float, str)
    # Snap rocker_axis_pt so the pin axis is exactly normal to the rocker
    # plane (defined by rocker_pivot + rocker_spring_pt + pushrod_inner).
    # Payload = (axle,) — 'front' or 'rear'.
    snap_axis_to_normal_requested = pyqtSignal(str)
    # Snap the WHOLE actuation chain coplanar: project pushrod_outer /
    # spring_chassis_pt / damper ends / ARB drop link onto the rocker plate
    # plane and snap the rocker pin normal.  Payload = (axle,).
    snap_actuation_to_plane_requested = pyqtSignal(str)
    # Set a LENGTH directly (coordinates stay for position).
    rack_length_changed = pyqtSignal(float)        # mm, inner-pickup tip-to-tip
    shock_length_changed = pyqtSignal(str, float)  # (axle, mm) mount-to-mount
    # Fired when the plane-tilt axle selector changes, so MainWindow can
    # repopulate the pivot dropdown with the pivots that actually exist on
    # that axle's topology (e.g. a DIRECT axle offers damper endpoints; a
    # pushrod axle offers rocker/pushrod points).  Payload = 'front'/'rear'.
    plane_axle_changed = pyqtSignal(str)
    # Translate a whole sub-assembly together on one axle.
    # Payload = (group, axle, axis_index, mm_signed); group in
    # {'spring','arb','arms'}, axis 0=X 1=Y 2=Z.
    group_move_requested = pyqtSignal(str, str, int, float)
    # User typed an exact coordinate for the selected point (mm, world).
    position_typed = pyqtSignal(float, float, float)
    # Nudge-constraint mode changed: 'free' | 'link' | 'plane'.
    constraint_mode_changed = pyqtSignal(str)
    # Snapshot current geometry as the baseline (ghost + delta readout).
    baseline_set_requested = pyqtSignal()
    ghost_toggled = pyqtSignal(bool)

    _STEPS_MM = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    def __init__(self):
        super().__init__('Direct Edit Mode', header_color='#FFD600')
        self.set_info(section_info.DIRECT_EDIT)

        # Row 1: Enable + Mirror toggles
        row1 = QHBoxLayout(); row1.setSpacing(8)
        self._chk_enable = QCheckBox('Enable (WASD/QE keys)')
        self._chk_enable.setToolTip(
            'When ON, the 3D viewer captures keyboard input.\n'
            'WASD = X/Y plane · QE = Z axis · 1-6 = step · Ctrl+Z = undo')
        # Use explicit lambda forwarder — PyQt6 can drop signal.emit slots.
        self._chk_enable.toggled.connect(
            lambda v: self.enabled_changed.emit(bool(v)))
        row1.addWidget(self._chk_enable)

        self._chk_mirror = QCheckBox('Mirror F↔R')
        self._chk_mirror.setToolTip(
            'Apply the same delta to the matching hardpoint on the other axle\n'
            '(e.g. editing front uca_outer also moves rear uca_outer).')
        self._chk_mirror.toggled.connect(
            lambda v: self.mirror_axle_changed.emit(bool(v)))
        row1.addWidget(self._chk_mirror)
        self.add_layout(row1)

        # Row 2: Corner selector (FL FR RL RR)
        cl = QHBoxLayout(); cl.setSpacing(2)
        cl.addWidget(QLabel('Corner:'))
        self._corner_group = QButtonGroup(self)
        self._corner_radios: dict[str, QRadioButton] = {}
        for corner in ('FL', 'FR', 'RL', 'RR'):
            r = QRadioButton(corner)
            r.toggled.connect(
                lambda checked, c=corner: self._on_corner_toggled(c, checked))
            self._corner_group.addButton(r)
            self._corner_radios[corner] = r
            cl.addWidget(r)
        cl.addStretch(1)
        self._corner_radios['FL'].setChecked(True)
        self._current_corner = 'FL'
        self.add_layout(cl)

        # Row 3: Hardpoint list
        self.add_widget(QLabel('Hardpoint:'))
        self._hp_list = QListWidget()
        self._hp_list.setMaximumHeight(220)
        self._hp_list.setMinimumHeight(140)
        for name in HP_NAMES:
            self._hp_list.addItem(name)
        self._hp_list.currentItemChanged.connect(self._on_hp_pick)
        self._hp_list.setStyleSheet("""
            QListWidget {
                background-color: #0e0e0e;
                border: 1px solid #2a2a2a;
                outline: 0;
            }
            QListWidget::item {
                color: #d6d6d6;
                font-family: Consolas, 'SF Mono', monospace;
                font-size: 11.5px;
                padding: 4px 8px;
                border: 0;
            }
            QListWidget::item:hover {
                background-color: #1d1d1d;
                color: #ffffff;
            }
            QListWidget::item:selected,
            QListWidget::item:selected:active,
            QListWidget::item:selected:!active {
                background-color: #FFD600;
                color: #0a0a0a;
                font-weight: bold;
            }
        """)
        self.add_widget(self._hp_list)

        # Row 4: Position — EDITABLE.  Type an exact coordinate and the point
        # goes there (committed through the same delta path as WASD, so undo /
        # damper-bounds guard / live graphs all apply).
        pos_row = QHBoxLayout(); pos_row.setSpacing(4)
        self._pos_spins = []
        for axis in ('X', 'Y', 'Z'):
            pos_row.addWidget(QLabel(axis))
            sb = _NoScrollSpin()
            sb.setRange(-5000.0, 5000.0)
            sb.setDecimals(2)
            sb.setSingleStep(1.0)
            sb.setSuffix(' mm')
            sb.setEnabled(False)
            sb.setKeyboardTracking(False)   # commit on Enter/focus-out only
            sb.editingFinished.connect(self._on_pos_typed)
            self._pos_spins.append(sb)
            pos_row.addWidget(sb, 1)
        self.add_layout(pos_row)

        # Δ from baseline (set via the Baseline button below)
        self._delta_label = QLabel('Δ baseline:  —')
        self._delta_label.setStyleSheet(
            f"color:#FFD600; font-family:'Consolas','SF Mono',monospace;"
            f"font-size:11px; padding:2px 2px;")
        self._delta_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.add_widget(self._delta_label)

        # Row 4b: nudge constraint — Free / Link / Plane (keys F / L / P)
        con_row = QHBoxLayout(); con_row.setSpacing(4)
        con_row.addWidget(QLabel('Constrain:'))
        self._constraint_group = QButtonGroup(self)
        self._constraint_btns = {}
        for label, mode, tip in (
            ('Free (F)', 'free', 'WASD/QE move in world axes'),
            ('Link (L)', 'link', 'Movement projected onto the LINK this point '
                                 'belongs to (slides along the member — length '
                                 'changes, direction preserved)'),
            ('Plane (P)', 'plane', 'Movement projected onto the rocker / '
                                   'bellcrank plane (keeps the mechanism planar)'),
        ):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setStyleSheet(
                'QPushButton { background:#1a1a1a; color:#999; padding:3px 8px;'
                ' border:1px solid #2a2a2a; border-radius:3px; font-size:11px; }'
                'QPushButton:checked { background:#FFD600; color:#0a0a0a;'
                ' font-weight:bold; }')
            b.clicked.connect(lambda _=False, m=mode: self._emit_constraint(m))
            self._constraint_group.addButton(b)
            self._constraint_btns[mode] = b
            con_row.addWidget(b)
        self._constraint_btns['free'].setChecked(True)
        con_row.addStretch(1)
        self.add_layout(con_row)

        # Row 4c: baseline ghost controls
        bl_row = QHBoxLayout(); bl_row.setSpacing(4)
        self._btn_baseline = QPushButton('Set baseline')
        self._btn_baseline.setToolTip(
            'Snapshot the CURRENT geometry.  The 3D shows it as a grey ghost '
            'and the Δ readout above measures every point against it.')
        self._btn_baseline.clicked.connect(
            lambda: self.baseline_set_requested.emit())
        bl_row.addWidget(self._btn_baseline)
        self._chk_ghost = QCheckBox('Show ghost')
        self._chk_ghost.setChecked(True)
        self._chk_ghost.toggled.connect(
            lambda v: self.ghost_toggled.emit(bool(v)))
        bl_row.addWidget(self._chk_ghost)
        bl_row.addStretch(1)
        self.add_layout(bl_row)

        # Row 5: Step buttons (1-6)
        self.add_widget(QLabel('Step (key 1-6):'))
        step_row = QHBoxLayout(); step_row.setSpacing(2)
        self._step_buttons: list[tuple[QToolButton, float]] = []
        for i, step in enumerate(self._STEPS_MM):
            btn = QToolButton()
            btn.setText(f'{i+1}\n{step:g} mm')
            btn.setCheckable(True)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.setStyleSheet("""
                QToolButton { background:#1a1a1a; color:#aaa; border:1px solid #2a2a2a;
                              border-radius:3px; padding:4px 0; font-size:10.5px; }
                QToolButton:hover   { background:#222; color:#fff; }
                QToolButton:checked { background:#FFD600; color:#111; font-weight:bold; }
            """)
            btn.clicked.connect(lambda _=None, s=step: self._on_step_clicked(s))
            self._step_buttons.append((btn, step))
            step_row.addWidget(btn)
        # default = 1.0 mm
        self._step_buttons[2][0].setChecked(True)
        self.add_layout(step_row)

        # Hint
        hint = QLabel(
            'WASD = ±Y forward/back   AD = ±X out/inboard   QE = ±Z up/down\n'
            '1-6 = step  ·  Ctrl+Z = undo')
        hint.setStyleSheet(f'color:{C_SUB}; font-size:10px; padding:4px 2px;')
        hint.setWordWrap(True)
        self.add_widget(hint)

        # ── Plane-tilt mini-section ──────────────────────────────────────
        # Rotates the entire actuation plane (pushrod / rocker / spring /
        # drop-link) of one axle as a rigid body about an axis through the
        # chosen pivot hardpoint.  Use this to skew the plane toward the
        # driver, level it, etc. — saves nudging 6-8 hardpoints individually.
        self.add_widget(_panel_sep())
        plane_hdr = QLabel('PLANE TILT  (rotate rocker plane as one piece)')
        plane_hdr.setStyleSheet('color:#FFD600; font-weight:bold; font-size:11px;'
                                ' padding-top:4px;')
        self.add_widget(plane_hdr)

        prow1 = QHBoxLayout(); prow1.setSpacing(4)
        prow1.addWidget(QLabel('Axle:'))
        self._cmb_plane_axle = QComboBox()
        self._cmb_plane_axle.addItems(['Front', 'Rear'])
        # Repopulate the pivot list when the user switches axle (a mixed car
        # can have a pushrod front and a direct rear, with different pivots).
        self._cmb_plane_axle.currentTextChanged.connect(
            lambda t: self.plane_axle_changed.emit(t.strip().lower()))
        prow1.addWidget(self._cmb_plane_axle)
        prow1.addWidget(QLabel('Axis:'))
        self._cmb_plane_axis = QComboBox()
        # Axis labels map to a unit vector inside main_window's handler:
        #   PUSHROD = pushrod_inner - pushrod_outer (skew plane about the
        #             pushrod, keeping wheel attach fixed — this is how
        #             we design it: damper/rocker/drop-link tilt as one
        #             rigid plane around the pushrod line)
        #   X       = +X world (lateral, outboard)
        #   Y       = +Y world (longitudinal, fwd)
        #   Z       = +Z world (vertical, up)
        self._cmb_plane_axis.addItems(
            ['Pushrod', 'X (lat)', 'Y (long)', 'Z (vert)'])
        prow1.addWidget(self._cmb_plane_axis)
        prow1.addStretch(1)
        self.add_layout(prow1)

        prow2 = QHBoxLayout(); prow2.setSpacing(4)
        prow2.addWidget(QLabel('Pivot:'))
        self._cmb_plane_pivot = QComboBox()
        self._cmb_plane_pivot.addItems(
            ['rocker_pivot', 'rocker_spring_pt', 'pushrod_inner',
             'pushrod_outer', 'spring_chassis_pt'])
        prow2.addWidget(self._cmb_plane_pivot)
        prow2.addWidget(QLabel('Tilt:'))
        self._spn_plane_deg = QDoubleSpinBox()
        self._spn_plane_deg.setRange(-90.0, 90.0)
        self._spn_plane_deg.setDecimals(2)
        self._spn_plane_deg.setSingleStep(0.5)
        self._spn_plane_deg.setSuffix(' deg')
        self._spn_plane_deg.setValue(1.0)
        prow2.addWidget(self._spn_plane_deg)
        prow2.addStretch(1)
        self.add_layout(prow2)

        prow3 = QHBoxLayout(); prow3.setSpacing(6)
        btn_minus = QPushButton('– Tilt')
        btn_minus.setToolTip('Rotate plane by –tilt about chosen axis through pivot.')
        btn_minus.clicked.connect(lambda: self._emit_plane_tilt(-1.0))
        btn_plus = QPushButton('+ Tilt')
        btn_plus.setToolTip('Rotate plane by +tilt about chosen axis through pivot.')
        btn_plus.clicked.connect(lambda: self._emit_plane_tilt(+1.0))
        for b in (btn_minus, btn_plus):
            b.setStyleSheet("""
                QPushButton { background:#2a2a2a; color:#FFD600;
                              border:1px solid #3a3a3a; border-radius:4px;
                              padding:5px 10px; font-weight:bold; }
                QPushButton:hover  { background:#3a3a3a; color:#FFEB3B; }
                QPushButton:pressed{ background:#1a1a1a; }
            """)
        prow3.addWidget(btn_minus, 1)
        prow3.addWidget(btn_plus,  1)
        self.add_layout(prow3)

        # Snap rocker pin axis to plane normal (one button per axle)
        prow4 = QHBoxLayout(); prow4.setSpacing(6)
        btn_snap = QPushButton('Snap pin-axis ⟂ plane')
        btn_snap.setToolTip(
            'Re-compute rocker_axis_pt as rocker_pivot + L·n, where n is '
            'the unit normal to the rocker plane (defined by rocker_pivot, '
            'rocker_spring_pt, pushrod_inner) and L is the current pin '
            'length.\nUse this after manually editing rocker_axis_pt if the '
            'pin direction drifted off normal — the rocker physically '
            'pivots about a pin that MUST be normal to its plate.')
        btn_snap.clicked.connect(self._emit_snap_axis)
        btn_snap.setStyleSheet("""
            QPushButton { background:#2a2a2a; color:#90CAF9;
                          border:1px solid #3a3a3a; border-radius:4px;
                          padding:5px 10px; }
            QPushButton:hover { background:#3a3a3a; color:#BBDEFB; }
        """)
        prow4.addWidget(btn_snap, 1)
        self.add_layout(prow4)

        # Snap the whole actuation chain coplanar (pushrod_outer, spring chassis
        # pt, damper ends, ARB drop link projected onto the rocker plate plane).
        prow4b = QHBoxLayout(); prow4b.setSpacing(6)
        btn_snap_act = QPushButton('Snap actuation ⟂ plane')
        btn_snap_act.setToolTip(
            'Project every actuation-chain point that should lie in the rocker '
            'plate plane (pushrod_outer, spring_chassis_pt, damper ends, ARB '
            'drop-link top) ONTO that plane, and snap the rocker pin normal.\n'
            'Use after nudging a point off the plane — a bellcrank is a planar '
            'mechanism, so these points must be coplanar.')
        btn_snap_act.clicked.connect(self._emit_snap_actuation)
        btn_snap_act.setStyleSheet("""
            QPushButton { background:#2a2a2a; color:#FFD600;
                          border:1px solid #3a3a3a; border-radius:4px;
                          padding:5px 10px; }
            QPushButton:hover { background:#3a3a3a; color:#FFEA00; }
        """)
        prow4b.addWidget(btn_snap_act, 1)
        self.add_layout(prow4b)

        plane_hint = QLabel(
            'Moves all rocker-plane hardpoints together (pushrod, rocker,\n'
            'spring chassis pt, drop link) so the plane tilts as one rigid\n'
            'body.  Pivot point stays fixed.  Undoable as ONE Ctrl+Z step.')
        plane_hint.setStyleSheet(f'color:{C_SUB}; font-size:10px;'
                                 ' padding:2px 2px 6px 2px;')
        plane_hint.setWordWrap(True)
        self.add_widget(plane_hint)

        # ── Group moves (shift a whole sub-assembly) ─────────────────────
        self.add_widget(_panel_sep())
        grp_hdr = QLabel('GROUP MOVE  (shift a whole sub-assembly)')
        grp_hdr.setStyleSheet('color:#FFD600; font-weight:bold; font-size:11px;'
                              ' padding-top:4px;')
        self.add_widget(grp_hdr)

        grow = QHBoxLayout(); grow.setSpacing(4)
        grow.addWidget(QLabel('Step:'))
        self._grp_step = QDoubleSpinBox()
        self._grp_step.setRange(0.1, 100.0)
        self._grp_step.setDecimals(1)
        self._grp_step.setSingleStep(1.0)
        self._grp_step.setValue(5.0)
        self._grp_step.setSuffix(' mm')
        grow.addWidget(self._grp_step)
        grow.addWidget(QLabel('(uses Axle selector above)'))
        grow.addStretch(1)
        self.add_layout(grow)

        def _gbtn(text, group, axis, sign, tip):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(
                lambda _=False, g=group, a=axis, s=sign: self._emit_group_move(g, a, s))
            b.setStyleSheet("""
                QPushButton { background:#2a2a2a; color:#FFD600;
                              border:1px solid #3a3a3a; border-radius:4px;
                              padding:5px 8px; font-weight:bold; }
                QPushButton:hover  { background:#3a3a3a; color:#FFEB3B; }
                QPushButton:pressed{ background:#1a1a1a; }
            """)
            return b

        srow = QHBoxLayout(); srow.setSpacing(6)
        srow.addWidget(QLabel('Spring set:'))
        srow.addWidget(_gbtn('▲ up', 'spring', 2, +1.0,
            'Move the inboard spring/actuation hardware (rocker, spring-chassis, '
            'pushrod inner, heave / decoupled cradle) UP. Pushrod OUTER stays on '
            'the wishbone.'), 1)
        srow.addWidget(_gbtn('▼ down', 'spring', 2, -1.0,
            'Move the inboard spring/actuation hardware DOWN.'), 1)
        self.add_layout(srow)

        arow = QHBoxLayout(); arow.setSpacing(6)
        arow.addWidget(QLabel('ARB:'))
        arow.addWidget(_gbtn('▲ up', 'arb', 2, +1.0,
            'Move the ARB body UP. The drop-link top stays bolted to the '
            'rocker/upright.'), 1)
        arow.addWidget(_gbtn('▼ down', 'arb', 2, -1.0,
            'Move the ARB body DOWN (drop-link top stays put).'), 1)
        self.add_layout(arow)

        crow = QHBoxLayout(); crow.setSpacing(6)
        crow.addWidget(QLabel('Inboard arms:'))
        crow.addWidget(_gbtn('◄ in', 'arms', 0, -1.0,
            'Move inboard control-arm + toe-link pickups INBOARD (toward '
            'centreline). Front steering-rack inboard point is NOT moved.'), 1)
        crow.addWidget(_gbtn('out ►', 'arms', 0, +1.0,
            'Move inboard control-arm + toe-link pickups OUTBOARD. Front '
            'steering-rack inboard point is NOT moved.'), 1)
        self.add_layout(crow)

        grp_hint = QLabel(
            'Shifts a whole sub-assembly on the selected axle as ONE undo step.\n'
            'Geometry stays the source of truth — track / dynamics / loads follow.')
        grp_hint.setStyleSheet(f'color:{C_SUB}; font-size:10px;'
                               ' padding:2px 2px 6px 2px;')
        grp_hint.setWordWrap(True)
        self.add_widget(grp_hint)

        # (Frame / interference check moved to its own always-visible
        #  FrameInterferencePanel so the toggle is never buried here.)

        # ── DIMENSIONS: set a LENGTH directly (coords stay for position) ──
        self.add_widget(_panel_sep())
        dhdr = QLabel('DIMENSIONS  (set length, not coordinates)')
        dhdr.setStyleSheet('color:#FFD600; font-weight:bold; font-size:11px;'
                           ' padding-top:4px;')
        self.add_widget(dhdr)

        def _len(label, lo, hi, tip):
            row = QHBoxLayout(); row.setSpacing(4)
            lab = QLabel(label); lab.setToolTip(tip); row.addWidget(lab)
            sb = QDoubleSpinBox(); sb.setRange(lo, hi); sb.setDecimals(1)
            sb.setSingleStep(1.0); sb.setSuffix(' mm'); sb.setToolTip(tip)
            row.addWidget(sb); row.addStretch(1)
            self.add_layout(row)
            return sb

        self._rack_len = _len('Rack length (front):', 50.0, 900.0,
            'Tip-to-tip spacing of the two inner tie-rod pickups. Sets the X of '
            'tie_rod_inner symmetrically; Y/Z (position) untouched.')
        self._rack_len.editingFinished.connect(
            lambda: self.rack_length_changed.emit(self._rack_len.value()))
        self._shock_len_f = _len('Shock length (front):', 50.0, 600.0,
            'Mount-to-mount length of the front damper. Moves the chassis end '
            'along the damper axis; the rocker/spring end stays put.')
        self._shock_len_f.editingFinished.connect(
            lambda: self.shock_length_changed.emit('front', self._shock_len_f.value()))
        self._shock_len_r = _len('Shock length (rear):', 50.0, 600.0,
            'Mount-to-mount length of the rear damper. Moves the chassis end '
            'along the damper axis; the rocker/spring end stays put.')
        self._shock_len_r.editingFinished.connect(
            lambda: self.shock_length_changed.emit('rear', self._shock_len_r.value()))

        # Apply / Discard buttons
        btn_row = QHBoxLayout(); btn_row.setSpacing(6)
        self._btn_apply = QPushButton('Apply Changes')
        self._btn_apply.setToolTip(
            'Commit the current edits — clears the undo history so this geometry\n'
            'becomes the new baseline.  Use this when you\'re happy with the change.')
        self._btn_apply.setStyleSheet("""
            QPushButton { background:#FFD600; color:#111; border:0;
                          border-radius:4px; padding:8px 12px; font-weight:bold; }
            QPushButton:hover    { background:#FFEB3B; }
            QPushButton:pressed  { background:#FBC02D; }
            QPushButton:disabled { background:#3a3a3a; color:#777; }
        """)
        self._btn_apply.clicked.connect(lambda: self.apply_clicked.emit())
        btn_row.addWidget(self._btn_apply, 2)

        self._btn_discard = QPushButton('Discard')
        self._btn_discard.setToolTip(
            'Revert every edit since the last Apply (or since edit mode started).')
        self._btn_discard.setStyleSheet("""
            QPushButton { background:#2a2a2a; color:#ccc; border:1px solid #3a3a3a;
                          border-radius:4px; padding:8px 12px; }
            QPushButton:hover    { background:#3a3a3a; color:#fff; }
            QPushButton:pressed  { background:#1a1a1a; }
            QPushButton:disabled { color:#555; border-color:#2a2a2a; }
        """)
        self._btn_discard.clicked.connect(lambda: self.discard_clicked.emit())
        btn_row.addWidget(self._btn_discard, 1)
        self.add_layout(btn_row)

        # Edit counter
        self._edit_count_label = QLabel('No pending edits')
        self._edit_count_label.setStyleSheet(
            f'color:{C_SUB}; font-size:10.5px; padding:2px 2px;')
        self._edit_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.add_widget(self._edit_count_label)
        self._set_buttons_enabled(False)

    # ── External setters ─────────────────────────────────────────────────
    def set_step(self, mm: float):
        """Sync the step buttons when keys 1-6 are pressed in the 3D view."""
        for btn, s in self._step_buttons:
            btn.setChecked(abs(s - mm) < 1e-6)

    def set_position(self, xyz_mm):
        if xyz_mm is None:
            for sb in self._pos_spins:
                sb.blockSignals(True); sb.setValue(0.0)
                sb.setEnabled(False); sb.blockSignals(False)
            return
        for sb, v in zip(self._pos_spins, xyz_mm):
            sb.blockSignals(True)
            sb.setValue(float(v))
            sb.setEnabled(True)
            sb.blockSignals(False)

    def _on_pos_typed(self):
        """A coordinate spinbox was committed (Enter / focus-out) — emit the
        absolute target.  Guard: only when a point is actually selected."""
        if not self._pos_spins[0].isEnabled():
            return
        self.position_typed.emit(*(float(sb.value()) for sb in self._pos_spins))

    def _emit_constraint(self, mode: str):
        self._constraint_btns[mode].setChecked(True)
        self.constraint_mode_changed.emit(mode)

    def set_constraint_mode(self, mode: str):
        """Sync the buttons when F/L/P is pressed in the 3D view."""
        if mode in self._constraint_btns:
            self._constraint_btns[mode].setChecked(True)

    def set_baseline_delta(self, dxyz_mm):
        if dxyz_mm is None:
            self._delta_label.setText('Δ baseline:  —')
        else:
            dx, dy, dz = dxyz_mm
            self._delta_label.setText(
                f'Δ baseline:  {dx:+7.2f}  {dy:+7.2f}  {dz:+7.2f}  mm')

    def set_selection(self, hp_name: str | None, corner: str | None):
        """Sync the panel when the user clicks a hardpoint in the 3D view."""
        if corner and corner in self._corner_radios:
            self._corner_radios[corner].blockSignals(True)
            self._corner_radios[corner].setChecked(True)
            self._corner_radios[corner].blockSignals(False)
            self._current_corner = corner
        if hp_name:
            for i in range(self._hp_list.count()):
                if self._hp_list.item(i).text() == hp_name:
                    self._hp_list.blockSignals(True)
                    self._hp_list.setCurrentRow(i)
                    self._hp_list.blockSignals(False)
                    return
        else:
            self._hp_list.blockSignals(True)
            self._hp_list.setCurrentRow(-1)
            self._hp_list.blockSignals(False)

    def is_enabled(self) -> bool:
        return self._chk_enable.isChecked()

    def is_mirror_to_other_axle(self) -> bool:
        return self._chk_mirror.isChecked()

    def current_corner(self) -> str:
        return self._current_corner

    def set_hp_names(self, names: list[str]):
        """Replace the hardpoint list contents (used when topology changes
        and the active hp dict has different keys, e.g. direct-damper).
        Preserves the selected name if it still exists in the new list.
        """
        current = self._hp_list.currentItem().text() if self._hp_list.currentItem() else None
        self._hp_list.blockSignals(True)
        self._hp_list.clear()
        for nm in names:
            self._hp_list.addItem(nm)
        if current and current in names:
            self._hp_list.setCurrentRow(names.index(current))
        else:
            self._hp_list.setCurrentRow(-1)
        self._hp_list.blockSignals(False)

    def current_plane_axle(self) -> str:
        """'front' or 'rear' — the axle the plane-tilt controls target."""
        return self._cmb_plane_axle.currentText().strip().lower()

    def set_plane_pivots(self, names: list[str]):
        """Replace the plane-tilt pivot dropdown with the pivots that actually
        exist on the current axle's topology (poka-yoke: the user can never
        pick a pivot the axle doesn't have).  Preserves the selection if it
        survives the topology change.  Falls back to a single 'rocker_pivot'
        placeholder if the axle somehow has no plane pivots so the widget is
        never empty."""
        names = list(names) or ['rocker_pivot']
        current = self._cmb_plane_pivot.currentText()
        self._cmb_plane_pivot.blockSignals(True)
        self._cmb_plane_pivot.clear()
        self._cmb_plane_pivot.addItems(names)
        if current in names:
            self._cmb_plane_pivot.setCurrentIndex(names.index(current))
        self._cmb_plane_pivot.blockSignals(False)

    def set_edit_count(self, n: int):
        """Called externally to reflect the pending-edit count + button state."""
        if n <= 0:
            self._edit_count_label.setText('No pending edits')
        else:
            self._edit_count_label.setText(
                f'{n} pending edit step{"s" if n != 1 else ""}  '
                f'(Ctrl+Z to undo one)')
        self._set_buttons_enabled(n > 0)

    def _set_buttons_enabled(self, enabled: bool):
        self._btn_apply.setEnabled(enabled)
        self._btn_discard.setEnabled(enabled)

    # ── Plane-tilt internals ─────────────────────────────────────────────
    def _emit_plane_tilt(self, sign: float):
        """Send the (axle, axis_key, deg, pivot) tuple up to MainWindow.

        axis_key is a short stable token (not localised text):
            'PUSHROD' | 'X' | 'Y' | 'Z'
        MainWindow resolves it to a unit vector at the time of rotation.
        """
        axle = self._cmb_plane_axle.currentText().lower()        # 'front'/'rear'
        axis_label = self._cmb_plane_axis.currentText().strip()
        if axis_label.upper().startswith('PUSHROD'):
            axis_key = 'PUSHROD'
        else:
            # First letter of "X / Y / Z (...)" maps to canonical world axis
            axis_key = axis_label[0].upper()
        deg = float(self._spn_plane_deg.value()) * sign
        pivot = self._cmb_plane_pivot.currentText()
        # For rotation about the pushrod, the pivot MUST lie on the
        # pushrod line or you get a screw motion (rotation + translation).
        # Auto-snap to pushrod_inner — that's the canonical chassis end
        # and what we use during design.
        if axis_key == 'PUSHROD':
            pivot = 'pushrod_inner'
        # Only emit if user has selected a non-zero rotation
        if abs(deg) < 1e-9:
            return
        self.plane_tilt_requested.emit(axle, axis_key, deg, pivot)

    def _emit_group_move(self, group: str, axis: int, sign: float):
        """Send a (group, axle, axis, mm_signed) group-move up to MainWindow.
        Axle comes from the same selector the plane-tilt uses."""
        axle = self.current_plane_axle()
        mm = float(self._grp_step.value()) * float(sign)
        self.group_move_requested.emit(group, axle, int(axis), mm)

    def _emit_snap_axis(self):
        """Snap rocker_axis_pt to be normal to the rocker plane on the
        currently-chosen axle."""
        axle = self._cmb_plane_axle.currentText().lower()
        self.snap_axis_to_normal_requested.emit(axle)

    def _emit_snap_actuation(self):
        """Snap the whole actuation chain coplanar on the chosen axle."""
        self.snap_actuation_to_plane_requested.emit(
            self._cmb_plane_axle.currentText().lower())

    def set_dimensions(self, rack_mm, shock_f_mm, shock_r_mm):
        """Populate the length boxes with current geometry (no signal)."""
        for sb, v in ((self._rack_len, rack_mm),
                      (self._shock_len_f, shock_f_mm),
                      (self._shock_len_r, shock_r_mm)):
            sb.blockSignals(True)
            if v is not None and v > 0:
                sb.setValue(float(v)); sb.setEnabled(True)
            else:
                sb.setEnabled(False)
            sb.blockSignals(False)

    # ── Internal handlers ────────────────────────────────────────────────
    def _on_corner_toggled(self, corner: str, checked: bool):
        if not checked:
            return
        self._current_corner = corner
        # Guard against the radio firing during __init__ before _hp_list exists
        if not hasattr(self, '_hp_list'):
            return
        item = self._hp_list.currentItem()
        if item is not None:
            self.hp_selected.emit(item.text(), corner)

    def _on_hp_pick(self, current, previous):
        if current is None:
            return
        self.hp_selected.emit(current.text(), self._current_corner)

    def _on_step_clicked(self, step: float):
        for btn, s in self._step_buttons:
            btn.setChecked(abs(s - step) < 1e-6)
        self.step_changed.emit(step)
