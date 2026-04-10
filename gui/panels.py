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
    QDialog, QDialogButtonBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont

from vahan.metrics_catalog import CATALOG, CATALOG_MAP, DEFAULT_Y_KEYS

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
    """A titled section that can be toggled open/closed."""

    def __init__(self, title: str, parent=None, header_color: str = '#cccccc'):
        super().__init__(parent)
        self._title = title
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 2)
        layout.setSpacing(0)

        # header toggle button
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
        layout.addWidget(self._btn)

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
        damper_params_changed(dict)   — {'stroke_mm', 'sag_pct'}
    """
    motion_changed        = pyqtSignal(str)
    range_changed         = pyqtSignal(float, float)
    position_changed      = pyqtSignal(float)
    damper_params_changed = pyqtSignal(dict)

    def __init__(self):
        super().__init__('Motion')
        self._motion  = 'heave'
        self._min_val = -50.0
        self._max_val =  50.0
        self._pos     =   0.0
        self._building = False
        self._build()

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
        self._min_spin = _spin(-300, 0, -50, ' mm'); self._min_spin.valueChanged.connect(self._on_range)
        grid.addWidget(self._min_spin, 0, 1)
        grid.addWidget(QLabel('Max:'), 0, 2)
        self._max_spin = _spin(0, 300, 50, ' mm');   self._max_spin.valueChanged.connect(self._on_range)
        grid.addWidget(self._max_spin, 0, 3)
        self.add_layout(grid)

        # Damper limits
        dlim = QGridLayout(); dlim.setSpacing(4)
        dlim.addWidget(QLabel('Stroke:'), 0, 0)
        self._stroke = _spin(10, 300, 55, ' mm'); self._stroke.valueChanged.connect(self._on_damper)
        dlim.addWidget(self._stroke, 0, 1)
        dlim.addWidget(QLabel('Static sag:'), 0, 2)
        self._sag = _spin(0, 80, 35, ' %'); self._sag.valueChanged.connect(self._on_damper)
        dlim.addWidget(self._sag, 0, 3)
        self.add_layout(dlim)

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
            'stroke_mm': self._stroke.value(),
            'sag_pct':   self._sag.value(),
        })

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
            'max_rack_travel_in':     self._rack_max_in.value(),
        }

    def _build(self):
        grid = QGridLayout(); grid.setSpacing(4)

        grid.addWidget(QLabel('Rack travel/rev:'), 0, 0)
        self._rack_ratio = _spin(10, 200, 60, ' mm/rev')
        self._rack_ratio.valueChanged.connect(
            lambda _: self.steering_changed.emit(self.get_params()))
        grid.addWidget(self._rack_ratio, 0, 1)

        grid.addWidget(QLabel('Total rack travel:'), 1, 0)
        self._rack_total = _spin(20, 300, 100, ' mm')
        self._rack_total.valueChanged.connect(
            lambda _: self.steering_changed.emit(self.get_params()))
        grid.addWidget(self._rack_total, 1, 1)

        grid.addWidget(QLabel('Max rack travel:'), 2, 0)
        self._rack_max_in = _spin(0.1, 10.0, 2.0, ' in', dec=2, step=0.1)
        self._rack_max_in.valueChanged.connect(
            lambda _: self.steering_changed.emit(self.get_params()))
        grid.addWidget(self._rack_max_in, 2, 1)
        self.add_layout(grid)

        info = QLabel('Use Motion > Steer mode to simulate steering.')
        info.setWordWrap(True)
        info.setStyleSheet('color: #888888; font-size: 11px;')
        self.add_widget(info)


# ══════════════════════════════════════════════════════════════════════════════
#  CAR PARAMETERS PANEL
# ══════════════════════════════════════════════════════════════════════════════

class CarParamsPanel(CollapsibleSection):
    """Track width, wheelbase, tire dimensions, ground grid toggle."""
    params_changed = pyqtSignal(dict)

    def __init__(self):
        super().__init__('Car Parameters')
        self._build()

    def get_params(self) -> dict:
        return {
            'track_mm':              self._track.value(),
            'wheelbase_mm':          self._wb.value(),
            'tire_outer_dia_mm':     self._t_outer.value(),
            'tire_rim_dia_mm':       self._t_rim.value(),
            'tire_width_mm':         self._t_width.value(),
            'show_ground':           self._show_ground.isChecked(),
            'cg_x_mm':              self._cg_x.value(),
            'cg_y_mm':              self._cg_y.value(),
            'cg_z_mm':              self._cg_z.value(),
            'front_brake_bias_pct':  self._brake_bias.value(),
        }

    def _build(self):
        g = QGridLayout(); g.setSpacing(4)
        def row(label, lo, hi, val, suf, r, dec=0, step=10):
            g.addWidget(QLabel(label), r, 0)
            sb = _spin(lo, hi, val, suf, dec=dec, step=step)
            sb.valueChanged.connect(lambda _: self.params_changed.emit(self.get_params()))
            g.addWidget(sb, r, 1)
            return sb
        self._track      = row('Track width:',      800,  2000, 1220, ' mm', 0)
        self._wb         = row('Wheelbase:',         1500, 4000, 2400, ' mm', 1)
        self._t_outer    = row('Tyre OD:',           300,  700,  406,  ' mm', 2)
        self._t_rim      = row('Rim dia:',           200,  600,  330,  ' mm', 3)
        self._t_width    = row('Tyre width:',        100,  400,  200,  ' mm', 4)
        self._cg_x       = row('CG X (lateral):',   -300, 300,    0,  ' mm', 5, dec=1, step=1)
        self._cg_y       = row('CG Y (longitudinal):', 500, 3000, 1100, ' mm', 6, dec=1, step=5)
        self._cg_z       = row('CG Z (height):',     100, 600,  280,  ' mm', 7, dec=1, step=1)
        self._brake_bias = row('Front Brake Bias:',  30,   90,   65,   ' %',  8,
                               dec=0, step=1)
        self.add_layout(g)

        self._show_ground = QCheckBox('Show ground grid')
        self._show_ground.setChecked(True)
        self._show_ground.stateChanged.connect(
            lambda _: self.params_changed.emit(self.get_params()))
        self.add_widget(self._show_ground)

    def set_params(self, d: dict):
        """Populate widgets from a dict (e.g. loaded project)."""
        _map = {
            'track_mm': self._track, 'wheelbase_mm': self._wb,
            'tire_outer_dia_mm': self._t_outer, 'tire_rim_dia_mm': self._t_rim,
            'tire_width_mm': self._t_width,
            'cg_x_mm': self._cg_x, 'cg_y_mm': self._cg_y, 'cg_z_mm': self._cg_z,
            'front_brake_bias_pct': self._brake_bias,
        }
        # backward compat: old files have cg_height_mm → map to cg_z_mm
        if 'cg_height_mm' in d and 'cg_z_mm' not in d:
            d['cg_z_mm'] = d.pop('cg_height_mm')
        for key, sb in _map.items():
            if key in d:
                sb.blockSignals(True)
                sb.setValue(d[key])
                sb.blockSignals(False)
        if 'show_ground' in d:
            self._show_ground.blockSignals(True)
            self._show_ground.setChecked(d['show_ground'])
            self._show_ground.blockSignals(False)


# ══════════════════════════════════════════════════════════════════════════════
#  HARDPOINT TABLE (one per axle)
# ══════════════════════════════════════════════════════════════════════════════

class HardpointPanel(CollapsibleSection):
    """
    Editable hardpoint table. Columns: Name | X | Y | Z (all in mm).
    Also hosts ARB hardpoints in an appended section.

    Signals:
        hp_changed(dict)   — full hp dict (metres)
        row_selected(str)  — hp name when row selected
    """
    hp_changed   = pyqtSignal(dict)
    row_selected = pyqtSignal(str)

    def __init__(self, title: str, hp_dict: dict, arb_dict: dict | None = None):
        super().__init__(title)
        self._hp    = {k: v.copy() for k, v in hp_dict.items()}
        self._arb   = {k: v.copy() for k, v in (arb_dict or {}).items()}
        self._names = list(hp_dict.keys())
        self._all_names = self._names + list(self._arb.keys())
        self._updating = False
        self._build()

    def refresh(self, hp_dict: dict, arb_dict: dict | None = None):
        self._hp  = {k: v.copy() for k, v in hp_dict.items()}
        if arb_dict:
            self._arb = {k: v.copy() for k, v in arb_dict.items()}
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
            if name in self._hp:
                vals_mm = self._hp[name] * 1000.0
                is_chassis = name in CHASSIS_PTS
                col_str = C_BLUE if is_chassis else C_RED
            else:
                vals_mm = self._arb[name] * 1000.0
                col_str = '#FFB300'   # amber for ARB

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
            if name in self._hp:
                pt = self._hp[name].copy()
                pt[col - 1] = val_m
                self._hp[name] = pt
            else:
                pt = self._arb[name].copy()
                pt[col - 1] = val_m
                self._arb[name] = pt
            self.hp_changed.emit({**{k: v.copy() for k, v in self._hp.items()},
                                   **{k: v.copy() for k, v in self._arb.items()}})
        except ValueError:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE VALUES TABLE
# ══════════════════════════════════════════════════════════════════════════════

class ValuesPanel(CollapsibleSection):
    """Read-only live metric values for the FL corner."""

    def __init__(self):
        super().__init__('Live Values — FL corner')
        self._build()

    def update_values(self, values: dict):
        self._updating = True
        for row, entry in enumerate(CATALOG):
            val = values.get(entry['key'], float('nan'))
            item = self._table.item(row, 1)
            if item:
                item.setText(f'{val:.4f}' if not np.isnan(val) else '—')
        self._updating = False

    def _build(self):
        self._table = QTableWidget(len(CATALOG), 3)
        self._table.setHorizontalHeaderLabels(['Metric', 'Value', 'Unit'])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col, w in ((1, 82), (2, 44)):
            self._table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
            self._table.setColumnWidth(col, w)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._updating = False

        # Slightly larger font for readability
        tbl_font = QFont()
        tbl_font.setPointSize(10)
        self._table.setFont(tbl_font)

        # Category header color: light gray
        prev_cat = None
        for row, entry in enumerate(CATALOG):
            cat = entry['category']
            ni = QTableWidgetItem(entry['label'])
            ni.setFlags(Qt.ItemFlag.ItemIsEnabled)
            if cat != prev_cat:
                ni.setForeground(QColor('#aaaaaa'))
            prev_cat = cat
            vi = QTableWidgetItem('—')
            vi.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            vi.setFlags(Qt.ItemFlag.ItemIsEnabled)
            ui = QTableWidgetItem(entry['unit'])
            ui.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            ui.setFlags(Qt.ItemFlag.ItemIsEnabled)
            ui.setForeground(QColor(C_SUB))
            self._table.setItem(row, 0, ni)
            self._table.setItem(row, 1, vi)
            self._table.setItem(row, 2, ui)

        self._table.resizeRowsToContents()
        self.add_widget(self._table)


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH PICKER
# ══════════════════════════════════════════════════════════════════════════════

class GraphPickerPanel(CollapsibleSection):
    """Select which metrics to plot and which corners to show."""
    selection_changed = pyqtSignal(list)
    corners_changed   = pyqtSignal(list)

    def __init__(self):
        super().__init__('Graph Selection', header_color='#cccccc')
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
        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.setMaximumHeight(200)

        prev_cat = None
        for entry in CATALOG:
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
                Qt.CheckState.Checked if entry['key'] in DEFAULT_Y_KEYS
                else Qt.CheckState.Unchecked)
            self._list.addItem(item)

        self._list.itemChanged.connect(
            lambda _: self.selection_changed.emit(self.get_selected_keys()))
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
        self._list.blockSignals(False)
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
        self._ft = _spin(-5.0, 5.0, 0.0, ' deg', dec=2, step=0.1)
        grid.addWidget(self._ft, 0, 1)

        grid.addWidget(QLabel('Front camber:'), 1, 0)
        self._fc = _spin(-10.0, 5.0, 0.0, ' deg', dec=2, step=0.1)
        grid.addWidget(self._fc, 1, 1)

        grid.addWidget(QLabel('Rear toe:'),     2, 0)
        self._rt = _spin(-5.0, 5.0, 0.0, ' deg', dec=2, step=0.1)
        grid.addWidget(self._rt, 2, 1)

        grid.addWidget(QLabel('Rear camber:'),  3, 0)
        self._rc = _spin(-10.0, 5.0, 0.0, ' deg', dec=2, step=0.1)
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
    sb.setRange(lo, hi)
    sb.setValue(val)
    sb.setSuffix(suffix)
    sb.setDecimals(dec)
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
    ('rc_height',    'Roll Centre Height', 'mm', -20, 80),
    ('caster',       'Caster Angle',       '°',   0, 15),
    ('trail',        'Caster Trail',       'mm', -10, 60),
    ('motion_ratio', 'Motion Ratio',       '-',   0.3, 1.5),
]

# Hardpoints the user can select (inboard chassis points that matter)
IK_HARDPOINTS = [
    'uca_front', 'uca_rear', 'uca_outer',
    'lca_front', 'lca_rear', 'lca_outer',
    'tie_rod_inner', 'tie_rod_outer',
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
        self._range_lo = _spin(-120, 120, -18, ' mm', dec=0, step=5)
        grid.addWidget(self._range_lo, r, 1); r += 1
        grid.addWidget(QLabel('Max:'), r, 0)
        self._range_hi = _spin(-120, 120, 42, ' mm', dec=0, step=5)
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

        # Bound (how far points can move)
        grid.addWidget(QLabel('Max Movement:'), r, 0)
        self._bound = _spin(1, 100, 10, ' mm', dec=1, step=1)
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
        self._lock_tol = _spin(0.1, 100, 5.0, '', dec=1, step=1.0)
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
                          'rc_height', 'caster', 'trail', 'motion_ratio'}
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
            sp = _spin(0, 150, default_mm, ' mm', dec=1, step=1.0)
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

    def _auto_select(self):
        """Auto-select hardpoints based on the chosen metric."""
        from vahan.optimizer import PRESETS
        key = self._metric.currentData()
        preset = PRESETS.get(key, [])
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
            'bound_mm':     self._bound.value(),
            'method':       self._method.currentText(),
            'hp_names':     hp_names,
            'coords':       coords,
            'lock_metrics': lock_metrics,
            'lock_tol':     self._lock_tol.value(),
            'tube_od':      tube_od,
        }

    def set_damper_limits(self, stroke_mm: float, sag_pct: float):
        """Clamp the IK sweep range to physical damper travel.

        stroke_mm : total damper stroke length in mm
        sag_pct   : percentage of stroke used as static sag (0-100)

        Effective range:
            min travel = -(stroke_mm * sag_pct / 100)   (bump / compression)
            max travel = stroke_mm * (1 - sag_pct/100)  (droop / extension)
        """
        sag_frac = max(0.0, min(1.0, sag_pct / 100.0))
        lo = -stroke_mm * sag_frac
        hi = stroke_mm * (1 - sag_frac)
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
